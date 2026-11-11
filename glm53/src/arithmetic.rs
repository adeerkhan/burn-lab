//! The miniature arithmetic corpus the pretraining ablations use.
//!
//! Mirrors `experiments/pretraining_data_diversity.py`: three-operation
//! compositional structures over add/sub/mul/neg/abs, rendered as instructions
//! plus a Python expression.

use burn::prelude::*;

use crate::{tasks::Rng, tokenizer::ByteTokenizer};

/// The primitive operations.
pub const OPS: [&str; 5] = ["add", "sub", "mul", "neg", "abs"];

/// An operation structure: three primitives, at least two distinct.
pub type Structure = [usize; 3];

fn phrases(op: usize) -> (&'static str, &'static str) {
    match op {
        0 => ("Add {c}.", "Increase it by {c}."),
        1 => ("Subtract {c}.", "Decrease it by {c}."),
        2 => ("Multiply by {c}.", "Scale it by {c}."),
        3 => ("Negate it.", "Flip its sign."),
        4 => ("Take its absolute value.", "Make the result nonnegative."),
        _ => panic!("unknown operation {op}"),
    }
}

/// Every balanced structure, shuffled with the reference's seed.
pub fn all_structures() -> Vec<Structure> {
    let mut values = Vec::new();
    for first in 0..OPS.len() {
        for second in 0..OPS.len() {
            for third in 0..OPS.len() {
                let structure = [first, second, third];
                let mut distinct = structure.to_vec();
                distinct.sort_unstable();
                distinct.dedup();
                if distinct.len() >= 2 {
                    values.push(structure);
                }
            }
        }
    }

    let mut rng = Rng::new(1701);
    for index in (1..values.len()).rev() {
        let swap = rng.below(index + 1);
        values.swap(index, swap);
    }
    values
}

/// Render one example, returning the text and the prompt's token length.
///
/// The prompt length is what the evaluator needs to know where the answer
/// starts; it is measured with the byte tokenizer exactly as the reference does.
pub fn render_example(structure: &Structure, index: usize, seed: u64) -> (String, usize) {
    let structure_key: usize = structure
        .iter()
        .enumerate()
        .map(|(position, operation)| (position + 1) * (operation + 1) * 101)
        .sum();

    let mut rng = Rng::new(seed * 1_000_003 + index as u64 * 9_176 + structure_key as u64);

    let mut instructions = Vec::with_capacity(structure.len());
    let mut expression = "x".to_string();
    let mut value: i64 = rng.below(11) as i64 - 5;

    for operation in structure {
        let constant = [2, 3, 4, 5][rng.below(4)];
        let (first, second) = phrases(*operation);
        let phrase = if rng.below(2) == 0 { first } else { second };
        instructions.push(phrase.replace("{c}", &constant.to_string()));

        match operation {
            0 => {
                expression = format!("({expression} + {constant})");
                value += constant;
            }
            1 => {
                expression = format!("({expression} - {constant})");
                value -= constant;
            }
            2 => {
                expression = format!("({expression} * {constant})");
                value *= constant;
            }
            3 => {
                expression = format!("(-{expression})");
                value = -value;
            }
            4 => {
                expression = format!("abs({expression})");
                value = value.abs();
            }
            _ => unreachable!(),
        }
    }
    let _ = value;

    let prompt = format!("Task: start with x. {}\nPython: ", instructions.join(" "));
    let completion = format!("{expression}\n");

    let tokenizer = ByteTokenizer::new();
    let prompt_length = tokenizer.encode(&prompt, true, false).len();
    (format!("{prompt}{completion}"), prompt_length)
}

/// A padded, next-byte batch.
pub struct Batch {
    pub inputs: Tensor<2, Int>,
    pub labels: Tensor<2, Int>,
    pub tokens: usize,
}

/// Build one batch of `batch_size` examples.
pub fn make_batch(
    structures: &[Structure],
    step: usize,
    batch_size: usize,
    sequence_length: usize,
    seed: u64,
    device: &Device,
) -> Batch {
    let tokenizer = ByteTokenizer::new();
    let width = sequence_length + 1;
    let mut rows: Vec<i64> = Vec::with_capacity(batch_size * width);
    let mut valid = 0usize;

    for offset in 0..batch_size {
        let index = step * batch_size + offset;
        let structure = &structures[index % structures.len()];
        let (text, _) = render_example(structure, index, seed);
        let mut ids: Vec<i64> = tokenizer
            .encode(&text, true, true)
            .into_iter()
            .map(|token| token as i64)
            .collect();
        assert!(
            ids.len() <= width,
            "example is {} tokens, limit is {width}",
            ids.len()
        );
        ids.resize(width, 0);
        valid += ids.iter().filter(|token| **token != 0).count();
        rows.extend(ids);
    }

    // Inputs are every token but the last; labels are every token but the first.
    let mut inputs = Vec::with_capacity(batch_size * sequence_length);
    let mut labels = Vec::with_capacity(batch_size * sequence_length);
    for row in 0..batch_size {
        let base = row * width;
        inputs.extend_from_slice(&rows[base..base + sequence_length]);
        for column in 0..sequence_length {
            let token = rows[base + column + 1];
            labels.push(if token == 0 { -100 } else { token });
        }
    }

    Batch {
        inputs: Tensor::<2, Int>::from_data(
            TensorData::new(inputs, [batch_size, sequence_length]),
            device,
        ),
        labels: Tensor::<2, Int>::from_data(
            TensorData::new(labels, [batch_size, sequence_length]),
            device,
        ),
        tokens: valid,
    }
}

/// Held-out metrics on the evaluation pool.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Score {
    pub target_loss: f64,
    pub target_perplexity: f64,
    pub target_byte_accuracy: f64,
    pub target_exact_accuracy: f64,
}

/// Teacher-forced evaluation over `structures`.
pub fn evaluate(
    model: &crate::GLM53Flash,
    structures: &[Structure],
    seed: u64,
    device: &Device,
) -> Score {
    let tokenizer = ByteTokenizer::new();
    let mut loss_sum = 0.0f64;
    let mut loss_count = 0usize;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut exact = 0usize;

    for (index, structure) in structures.iter().enumerate() {
        let (text, prompt_length) = render_example(structure, 100_000 + index, seed + 999);
        let ids: Vec<i64> = tokenizer
            .encode(&text, true, true)
            .into_iter()
            .map(|token| token as i64)
            .collect();
        let width = ids.len();
        assert!(width > 1, "an example needs at least two tokens");

        let input = Tensor::<2, Int>::from_data(
            TensorData::new(ids[..width - 1].to_vec(), [1, width - 1]),
            device,
        );
        let (logits, _) = model.clone().forward(input);

        // The answer starts at the prompt boundary.
        let vocab = model.config().vocab_size;
        let start = prompt_length - 1;
        let targets: Vec<i64> = ids[start + 1..].to_vec();
        let answer = logits.slice([0..1, start..width - 1, 0..vocab]);
        let answer = answer.reshape::<2, _>([width - 1 - start, vocab]);

        let loss = crate::vision_train::masked_cross_entropy(
            answer
                .clone()
                .reshape::<3, _>([1, width - 1 - start, vocab]),
            Tensor::<2, Int>::from_data(
                TensorData::new(targets.clone(), [1, targets.len()]),
                device,
            ),
        )
        .into_scalar::<f32>();
        loss_sum += loss as f64 * targets.len() as f64;
        loss_count += targets.len();

        let predicted = answer.argmax(1);
        let predicted = predicted.into_data().convert::<i32>();
        let predicted = predicted.as_slice::<i32>().expect("int predictions");
        let mut row_correct = 0usize;
        for (prediction, target) in predicted.iter().zip(&targets) {
            if *prediction as i64 == *target {
                row_correct += 1;
            }
        }
        correct += row_correct;
        total += targets.len();
        exact += usize::from(row_correct == targets.len());
    }

    let mean = loss_sum / loss_count as f64;
    Score {
        target_loss: mean,
        target_perplexity: mean.min(20.0).exp(),
        target_byte_accuracy: correct as f64 / total as f64,
        target_exact_accuracy: exact as f64 / structures.len() as f64,
    }
}

/// The miniature model the ablations use.
pub fn miniature_config(sequence_length: usize) -> crate::ModelConfig {
    crate::ModelConfig::new()
        .with_dim(48)
        .with_layers(3)
        .with_heads(3)
        .with_expert_hidden(96)
        .with_experts(4)
        .with_top_k(1)
        .with_streams(2)
        .with_sparse_window(24)
        .with_sparse_stride(24)
        .with_max_sequence_length(sequence_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structures_are_balanced_three_step_compositions() {
        let structures = all_structures();
        // 5^3 - 5 structures with all three distinct = 120.
        assert_eq!(structures.len(), 120);
        for structure in &structures {
            let mut distinct = structure.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                distinct.len() >= 2,
                "structure {structure:?} repeats one op"
            );
        }
    }

    #[test]
    fn rendering_is_deterministic_and_well_formed() {
        let structures = all_structures();
        let (first, prompt_length) = render_example(&structures[0], 7, 11);
        let (second, again) = render_example(&structures[0], 7, 11);
        assert_eq!(first, second);
        assert_eq!(prompt_length, again);
        assert!(first.starts_with("Task: start with x. "));
        assert!(first.contains("Python: "));
        assert!(prompt_length > 0);
    }

    #[test]
    fn distinct_structures_render_distinct_programs() {
        let structures = all_structures();
        let (first, _) = render_example(&structures[0], 0, 1);
        let (second, _) = render_example(&structures[1], 0, 1);
        assert_ne!(first, second);
    }

    #[test]
    fn a_batch_has_the_requested_shape() {
        let device = Device::flex();
        let structures = all_structures();
        let batch = make_batch(&structures, 0, 4, 128, 11, &device);
        assert_eq!(batch.inputs.dims(), [4, 128]);
        assert_eq!(batch.labels.dims(), [4, 128]);
        assert!(batch.tokens > 0);
    }
}
