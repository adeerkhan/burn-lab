//! Probe: does resuming from a checkpoint reproduce an uninterrupted run, and
//! what happens at the sequence-length boundary?

use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use glm53::training::{TrainConfig, save_checkpoint, train};
use glm53::{GLM53Flash, ModelConfig};

#[derive(Default)]
struct Collector {
    flat: Vec<f32>,
}

impl ModuleVisitor for Collector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.flat.extend(
            param
                .val()
                .to_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap(),
        );
    }
}

fn params(model: &GLM53Flash) -> Vec<f32> {
    let mut collector = Collector::default();
    model.visit(&mut collector);
    collector.flat
}

fn tiny() -> ModelConfig {
    ModelConfig::new()
        .with_dim(32)
        .with_heads(2)
        .with_layers(2)
        .with_experts(2)
        .with_expert_hidden(16)
        .with_streams(2)
        .with_max_sequence_length(64)
}

fn train_config(steps: usize) -> TrainConfig {
    TrainConfig {
        start_step: 0,
        steps,
        batch_size: 2,
        sequence_length: 12,
        learning_rate: 1e-3,
        seed: 7,
        corpus_size: 32,
        log_every: 1,
        checkpoints: Vec::new(),
        artifact_dir: None,
    }
}

// The CLI has no resume path: every `train` call starts from a fresh model, a
// fresh AdamW and a fresh RNG. This test pins the requirement for when one is
// added. It fails today (max |delta| ~= 2.8e-3 over 26k parameters) because the
// checkpoint carries weights only: the optimizer moments and the data-order RNG
// both restart, so "continue from checkpoint" is a different trajectory rather
// than the rest of the same one.
#[test]
#[ignore = "checkpoints carry weights only; optimizer and RNG state are not saved"]
fn resuming_from_a_checkpoint_matches_an_uninterrupted_run() {
    let device = Device::flex().autodiff();
    let config = tiny();
    let start = GLM53Flash::new(&config, &device);

    // Four steps in one go.
    let uninterrupted = params(&train(start.clone(), &train_config(4), &device, |_| {}));

    // Two steps, save, reload, two more.
    let first = train(start, &train_config(2), &device, |_| {});
    let directory = std::env::temp_dir().join("glm53-resume-probe");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.to_str().unwrap().to_string();
    save_checkpoint(&first, Some(&path), 2);

    let record =
        burn::store::ModuleRecord::load(format!("{path}/checkpoint-0002/model")).expect("load");
    let reloaded = GLM53Flash::new(&config, &device).load_record(record);
    let mut continued = train_config(2);
    continued.start_step = 2;
    let resumed = params(&train(reloaded, &continued, &device, |_| {}));

    let max_difference = uninterrupted
        .iter()
        .zip(&resumed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_difference = uninterrupted
        .iter()
        .zip(&resumed)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / uninterrupted.len() as f64;

    println!(
        "resume: max |delta| = {max_difference:.6e}, mean |delta| = {mean_difference:.6e}, params = {}",
        uninterrupted.len()
    );

    assert!(
        max_difference < 1e-6,
        "resumed run diverged from the uninterrupted one by {max_difference:.6e}"
    );
}

#[test]
fn the_sequence_length_boundary_is_explicit() {
    let device = Device::flex();
    let config = tiny();
    let model = GLM53Flash::new(&config, &device).valid();
    let max = config.max_sequence_length;

    for width in [1, max, max + 1] {
        let tokens: Vec<i64> = (0..width).map(|index| 10 + index as i64 % 50).collect();
        let input = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, width]), &device);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (logits, _) = model.clone().forward(input);
            logits.dims()
        }));
        match outcome {
            Ok(dims) => println!("width {width:>3}/{max} -> ok, dims {dims:?}"),
            Err(_) => println!("width {width:>3}/{max} -> panicked"),
        }
    }
}
