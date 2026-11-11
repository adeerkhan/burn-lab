//! The bounded full-model vision pilot.
//!
//! Mirrors `experiments/full_25m_vision_digit_pilot.py`: attach the vision tower
//! to the complete 25.7M language model, train only the vision tower, the final
//! block, the final norm and the tied embedding, and never save weights.
//!
//! The language model's vocabulary has to grow from 260 to 263 for the image
//! span. [`extend_vocab`] copies every matching parameter across and keeps the
//! new rows' initialisation, so the pretrained text behaviour survives.

use std::time::Instant;

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::{AdamWConfig, GradientsParams, grad_clipping::GradientClippingConfig},
    prelude::*,
};
use serde::Serialize;

use crate::{
    model::GLM53Flash,
    receipt,
    vision::{REQUIRED_VOCAB, VisionLanguageModel, answer_only_labels},
    vision_data,
    vision_train::{HeldOut, VisionTrainConfig, masked_cross_entropy, vision_config},
};

/// Settings for the pilot.
#[derive(Clone, Debug)]
pub struct PilotConfig {
    /// Steps to train.
    pub steps: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Held-out examples.
    pub eval_examples: usize,
    /// Seed.
    pub seed: u64,
    /// Wall-clock cap in seconds. `None` runs to completion.
    pub max_seconds: Option<f64>,
}

impl Default for PilotConfig {
    fn default() -> Self {
        Self {
            steps: 40,
            batch_size: 10,
            learning_rate: 1e-3,
            eval_examples: 100,
            seed: 73,
            max_seconds: Some(540.0),
        }
    }
}

/// The pilot's receipt.
#[derive(Clone, Debug, Serialize)]
pub struct PilotReport {
    pub schema_version: String,
    pub status: String,
    pub stage: String,
    pub checkpoint: String,
    pub checkpoint_sha256: Option<String>,
    pub vision_tokens: usize,
    pub language_parameters: usize,
    pub trainable_parameters: usize,
    pub visual_parameters: usize,
    pub steps_completed: usize,
    pub before: HeldOut,
    pub after: HeldOut,
    pub final_training_loss: Option<f64>,
    pub wall_time_seconds: f64,
    pub trainable_scope: String,
    pub weights_saved: bool,
}

#[derive(Default)]
struct Collector {
    chunks: Vec<Vec<f32>>,
}

impl ModuleVisitor for Collector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let data = param.val().into_data();
        self.chunks
            .push(data.convert::<f32>().as_slice::<f32>().unwrap().to_vec());
    }
}

struct Writer<'a> {
    chunks: &'a [Vec<f32>],
    cursor: usize,
}

impl ModuleMapper for Writer<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dims = param.val().dims();
        let count: usize = dims.iter().product();
        let device = param.val().device();
        let source = &self.chunks[self.cursor];
        self.cursor += 1;

        // Start from the target's own initialisation, then overwrite every
        // element the source covers. Only the embedding grows, so every other
        // parameter is overwritten completely.
        let mut values: Vec<f32> = param
            .val()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert!(
            source.len() <= count,
            "source parameter is larger than target"
        );
        values[..source.len()].copy_from_slice(source);

        let tensor = Tensor::<D>::from_data(TensorData::new(values, dims), &device);
        param.map(|_| tensor)
    }
}

/// Copy `source`'s parameters into `target`, which may have a larger vocabulary.
///
/// Both models must have the same architecture apart from the vocabulary, which
/// is what makes traversal order a valid pairing.
pub fn extend_vocab(source: &GLM53Flash, target: GLM53Flash) -> GLM53Flash {
    let mut collector = Collector::default();
    source.visit(&mut collector);

    let mut writer = Writer {
        chunks: &collector.chunks,
        cursor: 0,
    };
    let target = target.map(&mut writer);
    assert_eq!(
        writer.cursor,
        collector.chunks.len(),
        "both models must have the same parameter count"
    );
    target
}

/// Run the pilot against a text checkpoint.
pub fn run_pilot(
    checkpoint: &str,
    config: &PilotConfig,
    device: &Device,
    on_step: &mut dyn FnMut(usize, f64),
) -> Result<PilotReport, String> {
    let started = Instant::now();

    // The text checkpoint is 260-token; the vision path needs 263.
    let text_config = crate::ModelConfig::new();
    let record = burn::store::ModuleRecord::load(format!("{checkpoint}/model"))
        .map_err(|error| format!("could not load `{checkpoint}`: {error}"))?;
    let text_model = GLM53Flash::new(&text_config, device).load_record(record);

    let vision_language = crate::ModelConfig::new()
        .with_vocab_size(REQUIRED_VOCAB)
        .with_max_sequence_length(192);
    assert_eq!(
        vision_language.dim, text_config.dim,
        "dimension must not change"
    );
    let widened = GLM53Flash::new(&vision_language, device);
    let widened = extend_vocab(&text_model, widened);

    let mut model = VisionLanguageModel::new(widened, vision_config(), device);

    // Freeze everything, then re-open the vision tower and the surfaces the
    // reference trains: the last block, the final norm and the tied embedding.
    let vision = model.vision_group();
    let last_block = model.last_block_group();
    let final_norm = model.final_norm_group();
    let embedding = model.embedding_group();
    model = model.set_require_grad(false);
    model = model.set_require_grad_group(vision, true);
    model = model.set_require_grad_group(last_block, true);
    model = model.set_require_grad_group(final_norm, true);
    model = model.set_require_grad_group(embedding, true);

    let trainable = receipt::num_trainable_params(&model);
    let language_parameters = model.language_model().num_params();
    let visual_tokens = model.visual_token_count();
    let inserted = model.inserted_token_count();
    let visual_parameters = model.num_params() - language_parameters;

    let settings = VisionTrainConfig {
        steps: config.steps,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        eval_examples: config.eval_examples,
        seed: config.seed,
        noise: 0.06,
    };

    let before = held_out(&model, &settings);

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let mut final_loss = None;
    let mut steps_completed = 0;

    for step in 0..config.steps {
        if let Some(limit) = config.max_seconds
            && started.elapsed().as_secs_f64() > limit
        {
            break;
        }

        let labels: Vec<u8> = (0..config.batch_size)
            .map(|index| ((index + step) % 10) as u8)
            .collect();
        let images = vision_data::render_digits(&labels, config.seed + step as u64, 0.06, device);
        let (tokens, prompt_length) = vision_data::text_batch(&labels, device);
        let targets = answer_only_labels(
            tokens.clone(),
            prompt_length,
            visual_tokens,
            inserted - visual_tokens,
            device,
        );

        let [batch, time] = tokens.dims();
        let input = tokens.slice([0..batch, 0..time - 1]);
        let (logits, _) = model.forward(images, input);

        let loss = masked_cross_entropy(logits, targets);
        let value = loss.clone().into_scalar::<f32>();
        if !value.is_finite() {
            return Err(format!("non-finite loss at step {step}"));
        }
        final_loss = Some(value as f64);
        steps_completed = step + 1;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if step % 10 == 0 || step + 1 == config.steps {
            on_step(step + 1, value as f64);
        }
    }

    let after = held_out(&model, &settings);

    Ok(PilotReport {
        schema_version: "1.0".to_string(),
        status: "complete".to_string(),
        stage: "full_25m_vision_digit_pilot".to_string(),
        checkpoint: checkpoint.to_string(),
        checkpoint_sha256: receipt::read_receipt(std::path::Path::new(checkpoint))
            .map(|receipt| receipt.weights_sha256),
        vision_tokens: visual_tokens,
        language_parameters,
        trainable_parameters: trainable,
        visual_parameters,
        steps_completed,
        before,
        after,
        final_training_loss: final_loss,
        wall_time_seconds: started.elapsed().as_secs_f64(),
        trainable_scope: "vision + last block + final norm + tied embedding".to_string(),
        weights_saved: false,
    })
}

fn held_out(model: &VisionLanguageModel, config: &VisionTrainConfig) -> HeldOut {
    let labels: Vec<u8> = (0..config.eval_examples)
        .map(|index| (index % 10) as u8)
        .collect();
    let device = model
        .language_model()
        .devices()
        .first()
        .cloned()
        .unwrap_or_else(Device::flex);
    let images = vision_data::render_digits(&labels, 10_000 + config.seed, config.noise, &device);
    let (prompts, _) = vision_data::prompt_batch(labels.len(), &device);

    let (logits, _) = model.forward(images, prompts);
    let [batch, time, vocab] = logits.dims();
    let answer = logits
        .slice([0..batch, time - 1..time, 0..vocab])
        .reshape::<2, _>([batch, vocab]);

    let targets: Vec<i64> = labels
        .iter()
        .map(|label| (crate::tokenizer::BYTE_OFFSET + b'0' as u32 + *label as u32) as i64)
        .collect();
    let targets = Tensor::<1, Int>::from_data(TensorData::new(targets, [batch]), &device);

    let predicted = answer.clone().argmax(1);
    let predicted = predicted.into_data().convert::<i32>();
    let predicted = predicted.as_slice::<i32>().expect("int predictions");
    let expected = targets.clone().into_data().convert::<i32>();
    let expected = expected.as_slice::<i32>().expect("int targets");

    let correct = predicted
        .iter()
        .zip(expected)
        .filter(|(prediction, target)| prediction == target)
        .count();

    let loss = masked_cross_entropy(
        answer.reshape::<3, _>([batch, 1, vocab]),
        targets.reshape::<2, _>([batch, 1]),
    )
    .into_scalar::<f32>();

    HeldOut {
        accuracy: correct as f64 / batch as f64,
        loss: loss as f64,
        correct,
        examples: batch,
    }
}

/// Load a text model from a checkpoint directory.
pub fn load_text_model(
    config: &crate::ModelConfig,
    device: &Device,
    directory: &str,
) -> Result<GLM53Flash, String> {
    let record = burn::store::ModuleRecord::load(format!("{directory}/model"))
        .map_err(|error| format!("could not load `{directory}`: {error}"))?;
    Ok(GLM53Flash::new(config, device).load_record(record))
}

/// Write the pilot receipt, refusing to overwrite.
pub fn write_report(path: &str, report: &PilotReport) -> Result<(), String> {
    if std::path::Path::new(path).exists() {
        return Err(format!("{path} already exists"));
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let text = serde_json::to_string_pretty(report).map_err(|error| error.to_string())?;
    std::fs::write(path, text + "\n").map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extending_the_vocabulary_preserves_the_shared_rows() {
        let device = Device::flex();
        let source_config = crate::ModelConfig::new()
            .with_dim(32)
            .with_heads(4)
            .with_layers(1)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_max_sequence_length(64);
        let target_config = source_config.clone().with_vocab_size(REQUIRED_VOCAB);

        let source = GLM53Flash::new(&source_config, &device).valid();
        let target = GLM53Flash::new(&target_config, &device).valid();
        let widened = extend_vocab(&source, target);

        // Every shared parameter must match; the new vocabulary rows must not.
        let mut collector = Collector::default();
        source.visit(&mut collector);
        let mut widened_collector = Collector::default();
        widened.visit(&mut widened_collector);

        assert_eq!(collector.chunks.len(), widened_collector.chunks.len());
        for (index, (before, after)) in collector
            .chunks
            .iter()
            .zip(&widened_collector.chunks)
            .enumerate()
        {
            let shared = before.len().min(after.len());
            assert_eq!(
                &before[..shared],
                &after[..shared],
                "parameter {index} did not carry over"
            );
        }
    }

    #[test]
    fn the_widened_model_keeps_the_text_logits() {
        let device = Device::flex();
        let source_config = crate::ModelConfig::new()
            .with_dim(32)
            .with_heads(4)
            .with_layers(1)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_max_sequence_length(64);
        let target_config = source_config.clone().with_vocab_size(REQUIRED_VOCAB);

        let source = GLM53Flash::new(&source_config, &device).valid();
        let target = GLM53Flash::new(&target_config, &device).valid();
        let widened = extend_vocab(&source, target);

        let ids: Vec<i64> = vec![1, 14, 40, 20, 30];
        let text = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, 5]), &device);
        let (small, _) = source.clone().forward(text.clone());
        let (large, _) = widened.forward(text);

        // The first 260 logits of every position must agree.
        let small = small.into_data();
        let small = small.as_slice::<f32>().unwrap();
        let large = large.into_data();
        let large = large.as_slice::<f32>().unwrap();
        for position in 0..5 {
            for token in 0..260 {
                let a = small[position * 260 + token];
                let b = large[position * REQUIRED_VOCAB + token];
                assert!(
                    (a - b).abs() < 1e-5,
                    "position {position} token {token}: {a} vs {b}"
                );
            }
        }
    }
}
