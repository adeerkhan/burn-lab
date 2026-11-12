//! Cached versus uncached generation on the same model and prompt.

use burn::prelude::*;
use glm53::generate::{Decoding, GroupConfig, generate_group};
use glm53::{ByteTokenizer, GLM53Flash};
use std::time::Instant;

fn uncached_logits(model: &GLM53Flash, rows: &[Vec<u32>], device: &Device) -> Tensor<3> {
    let batch = rows.len();
    let width = rows[0].len();
    let flat: Vec<i64> = rows
        .iter()
        .flat_map(|row| row.iter().map(|token| *token as i64))
        .collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(flat, [batch, width]), device);
    model.forward(input).0
}

fn main() {
    let which = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "flex".to_string());
    let device = if which == "wgpu" {
        Device::wgpu(Default::default())
    } else {
        Device::flex()
    };

    let group = 16usize;
    let new_tokens = 24usize;

    let config = glm53::ModelConfig::new()
        .with_dim(192)
        .with_heads(6)
        .with_layers(4)
        .with_experts(8)
        .with_expert_hidden(256)
        .with_streams(4)
        .with_max_sequence_length(128);
    let model = glm53::seeding::seeded_model(&config, 42, &device).valid();

    let base = "# Return x plus one.\n# Double the input.\n# Return x squared.\n";
    let mut prompt: Vec<u32> = ByteTokenizer::new().encode(base, true, false);
    while prompt.len() < 48 {
        prompt.extend(ByteTokenizer::new().encode(base, false, false));
    }
    prompt.truncate(48);

    // Warm the single-token shapes before timing.
    let mut warm = glm53::kv::Cache::new(model.config());
    for (position, token) in prompt.iter().take(4).enumerate() {
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(vec![*token as i64; group], [group, 1]),
            &device,
        );
        let _ = model.forward_step(input, position, &mut warm);
    }
    let expected = group * new_tokens;

    // Cached path (what `generate_group` uses now).
    let task = glm53::tasks::frozen_tasks("dev", 1)[0].clone();
    let started = Instant::now();
    let _ = generate_group(
        &model,
        &task,
        &GroupConfig {
            group_size: group,
            max_new_tokens: new_tokens,
            decoding: Decoding::Greedy,
            seed: 1,
            stop_on_parseable: false,
        },
        &device,
    );
    let cached = started.elapsed().as_secs_f64();

    // Uncached: rebuild the whole prefix each step, as the reference does.
    let started = Instant::now();
    let vocab = config.vocab_size;
    let mut rows: Vec<Vec<u32>> = vec![prompt.clone(); group];
    for _ in 0..new_tokens {
        let logits = uncached_logits(&model, &rows, &device);
        let [_, time, _] = logits.dims();
        let last = logits.slice([0..group, time - 1..time, 0..vocab]);
        let chosen = last
            .argmax(2)
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap();
        for (row, token) in rows.iter_mut().zip(chosen) {
            row.push(token as u32);
        }
    }
    let uncached = started.elapsed().as_secs_f64();

    println!(
        "{which}: cached {cached:.2} s  uncached {uncached:.2} s  speedup {:.2}x  ({:.0} vs {:.0} tok/s)",
        uncached / cached,
        expected as f64 / cached,
        expected as f64 / uncached
    );
}
