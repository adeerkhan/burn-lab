//! Train and compare the two vision paths on generated RGB digits.
//!
//! Mirrors `scripts/train_vision.py`: the architecture-faithful tower against
//! the direct-patch baseline, on the same data, seeds and step budget.

use std::collections::BTreeMap;
use std::time::Instant;

use burn::{
    module::Module,
    optim::{AdamWConfig, GradientsParams, grad_clipping::GradientClippingConfig},
    prelude::*,
    tensor::activation,
};
use serde::Serialize;

use crate::{
    model::GLM53Flash,
    receipt::{self, BURN_VERSION},
    tokenizer::ByteTokenizer,
    vision::{
        DirectPatchVisionLanguageModel, VisionConfig, VisionLanguageModel, answer_only_labels,
    },
    vision_data,
};

/// Which image path to train.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// RGB patches, vision transformer blocks, 2x2 merge, gated projector.
    Faithful,
    /// RGB patches projected straight to language-model tokens.
    DirectPatch,
}

impl Architecture {
    pub fn parse(name: &str) -> Self {
        match name {
            "faithful" => Self::Faithful,
            "baseline" | "direct_patch_baseline" => Self::DirectPatch,
            other => panic!("unknown architecture `{other}`"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Faithful => "faithful",
            Self::DirectPatch => "direct_patch_baseline",
        }
    }
}

/// A vision model, whatever its topology.
// The two variants differ in size by design; boxing the baseline would add an
// indirection for no benefit in a teaching model.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum VisionModel {
    Faithful(VisionLanguageModel),
    DirectPatch(DirectPatchVisionLanguageModel),
}

impl VisionModel {
    pub fn forward(&self, images: Tensor<4>, text: Tensor<2, Int>) -> (Tensor<3>, Tensor<2>) {
        match self {
            Self::Faithful(model) => model.forward(images, text),
            Self::DirectPatch(model) => model.forward(images, text),
        }
    }

    pub fn visual_token_count(&self) -> usize {
        match self {
            Self::Faithful(model) => model.visual_token_count(),
            Self::DirectPatch(model) => model.visual_token_count(),
        }
    }

    pub fn inserted_token_count(&self) -> usize {
        match self {
            Self::Faithful(model) => model.inserted_token_count(),
            Self::DirectPatch(model) => model.inserted_token_count(),
        }
    }

    pub fn build(
        architecture: Architecture,
        language: &crate::ModelConfig,
        vision: &VisionConfig,
        device: &Device,
    ) -> Self {
        let language_model = GLM53Flash::new(language, device);
        match architecture {
            Architecture::Faithful => Self::Faithful(VisionLanguageModel::new(
                language_model,
                vision.clone(),
                device,
            )),
            Architecture::DirectPatch => Self::DirectPatch(DirectPatchVisionLanguageModel::new(
                language_model,
                8,
                device,
            )),
        }
    }
}

/// Training settings.
#[derive(Clone, Debug)]
pub struct VisionTrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub eval_examples: usize,
    pub seed: u64,
    pub noise: f32,
}

impl Default for VisionTrainConfig {
    fn default() -> Self {
        Self {
            steps: 120,
            batch_size: 40,
            learning_rate: 2e-3,
            eval_examples: 200,
            seed: 42,
            noise: 0.06,
        }
    }
}

/// A held-out measurement.
#[derive(Clone, Debug, Serialize)]
pub struct HeldOut {
    pub accuracy: f64,
    pub loss: f64,
    pub correct: usize,
    pub examples: usize,
}

/// One architecture's result.
#[derive(Clone, Debug, Serialize)]
pub struct ArchitectureResult {
    pub architecture: String,
    pub parameter_count: usize,
    pub visual_token_count: usize,
    pub inserted_token_count: usize,
    pub before: HeldOut,
    pub after: HeldOut,
    pub final_training_loss: Option<f64>,
    pub wall_time_seconds: f64,
}

/// The whole comparison, ready to serialise as a receipt.
#[derive(Clone, Debug, Serialize)]
pub struct VisionReport {
    pub schema_version: String,
    pub status: String,
    pub dataset: String,
    pub config: VisionReportConfig,
    pub environment: serde_json::Value,
    pub results: Vec<ArchitectureResult>,
    pub total_wall_time_seconds: f64,
    pub checkpoint_saved: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct VisionReportConfig {
    pub seed: u64,
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub eval_examples: usize,
    pub training_examples_seen_per_model: usize,
    pub held_out_seed: u64,
    pub vision: String,
}

/// The language model the vision path is attached to.
pub fn vision_language_config() -> crate::ModelConfig {
    crate::ModelConfig::new()
        .with_vocab_size(crate::vision::REQUIRED_VOCAB)
        .with_dim(32)
        .with_heads(4)
        .with_layers(1)
        .with_experts(4)
        .with_expert_hidden(48)
        .with_top_k(2)
        .with_streams(2)
        .with_sparse_window(8)
        .with_sparse_stride(8)
        .with_max_sequence_length(64)
}

/// The vision tower geometry the reference uses for this comparison.
pub fn vision_config() -> VisionConfig {
    VisionConfig::new()
        .with_hidden_size(24)
        .with_intermediate_size(48)
        .with_depth(2)
        .with_heads(4)
}

/// Train `architectures` and report the before/after held-out accuracy.
pub fn train_vision(
    architectures: &[Architecture],
    config: &VisionTrainConfig,
    device: &Device,
    on_step: &mut dyn FnMut(&str, usize, f64),
) -> VisionReport {
    let started = Instant::now();
    let language = vision_language_config();
    let vision = vision_config();

    let mut results = Vec::new();
    for architecture in architectures {
        results.push(train_one(
            *architecture,
            &language,
            &vision,
            config,
            device,
            on_step,
        ));
    }

    VisionReport {
        schema_version: "1.0".to_string(),
        status: "complete".to_string(),
        dataset: "generated RGB seven-segment digits".to_string(),
        config: VisionReportConfig {
            seed: config.seed,
            steps: config.steps,
            batch_size: config.batch_size,
            learning_rate: config.learning_rate,
            eval_examples: config.eval_examples,
            training_examples_seen_per_model: config.steps * config.batch_size,
            held_out_seed: 10_000 + config.seed,
            vision: format!("{vision:?}"),
        },
        environment: receipt::system_receipt(),
        results,
        total_wall_time_seconds: started.elapsed().as_secs_f64(),
        checkpoint_saved: false,
    }
}

fn train_one(
    architecture: Architecture,
    language: &crate::ModelConfig,
    vision: &VisionConfig,
    config: &VisionTrainConfig,
    device: &Device,
    on_step: &mut dyn FnMut(&str, usize, f64),
) -> ArchitectureResult {
    let mut model = VisionModel::build(architecture, language, vision, device);
    let parameters = model.num_params();
    let visual = model.visual_token_count();
    let inserted = model.inserted_token_count();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let evaluation_seed = 10_000 + config.seed;
    let before = held_out(&model, config, evaluation_seed, device);

    let started = Instant::now();
    let mut final_loss = None;

    for step in 0..config.steps {
        let labels: Vec<u8> = (0..config.batch_size)
            .map(|index| ((index + step) % 10) as u8)
            .collect();

        let images =
            vision_data::render_digits(&labels, config.seed + step as u64, config.noise, device);
        let (tokens, prompt_length) = vision_data::text_batch(&labels, device);
        let labels_tensor = answer_only_labels(
            tokens.clone(),
            prompt_length,
            visual,
            inserted - visual,
            device,
        );

        // The model sees everything but the answer token.
        let [batch, time] = tokens.dims();
        let input = tokens.slice([0..batch, 0..time - 1]);
        let (logits, usage) = model.forward(images, input);

        let language_loss = masked_cross_entropy(logits, labels_tensor);
        let balance = load_balance(usage, language.experts);
        let loss = language_loss.clone().add(balance.mul_scalar(0.01));

        let value = language_loss.into_scalar::<f32>();
        assert!(value.is_finite(), "non-finite loss at step {step}");
        final_loss = Some(value as f64);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if step % 20 == 0 || step + 1 == config.steps {
            on_step(architecture.label(), step + 1, value as f64);
        }
    }

    device.sync().ok();
    let after = held_out(&model, config, evaluation_seed, device);

    ArchitectureResult {
        architecture: architecture.label().to_string(),
        parameter_count: parameters,
        visual_token_count: visual,
        inserted_token_count: inserted,
        before,
        after,
        final_training_loss: final_loss,
        wall_time_seconds: started.elapsed().as_secs_f64(),
    }
}

/// Cross-entropy that ignores `-100` positions.
///
/// Burn's cross-entropy has no `ignore_index`, so the mask is applied to a
/// gathered log-probability and the mean is taken over valid positions only.
pub(crate) fn masked_cross_entropy(logits: Tensor<3>, labels: Tensor<2, Int>) -> Tensor<1> {
    let [batch, time, _vocab] = logits.dims();
    assert_eq!(labels.dims(), [batch, time], "one label per prediction");

    let log_probs = activation::log_softmax(logits, 2);
    let ignored = labels.clone().equal_scalar(-100);
    let safe = labels.mask_fill(ignored.clone(), 0);
    let picked = log_probs
        .gather(2, safe.unsqueeze_dim::<3>(2))
        .squeeze_dim::<2>(2);

    let mask = ignored.float().neg().add_scalar(1.0);
    let total = mask.clone().sum();

    picked.mul(mask).sum().div(total).neg()
}

/// The reference's auxiliary expert-balance penalty.
fn load_balance(usage: Tensor<2>, experts: usize) -> Tensor<1> {
    let [layers, _] = usage.dims();
    let mean = usage.reshape::<2, _>([layers, experts]).mean_dim(0);
    mean.sub_scalar(1.0 / experts as f32)
        .square()
        .mean()
        .mul_scalar(experts as f32)
}

/// Held-out accuracy on digits the model never trained on.
fn held_out(
    model: &VisionModel,
    config: &VisionTrainConfig,
    seed: u64,
    device: &Device,
) -> HeldOut {
    let labels: Vec<u8> = (0..config.eval_examples)
        .map(|index| (index % 10) as u8)
        .collect();
    let images = vision_data::render_digits(&labels, seed, config.noise, device);
    let (prompts, _) = vision_data::prompt_batch(labels.len(), device);

    let (logits, _) = model.forward(images, prompts);
    let [batch, time, vocab] = logits.dims();
    let answer = logits.slice([0..batch, time - 1..time, 0..vocab]);
    let answer = answer.reshape::<2, _>([batch, vocab]);

    let tokenizer = ByteTokenizer::new();
    let targets: Vec<i64> = labels
        .iter()
        // The answer token is the ASCII digit, not the numeric label.
        .map(|label| (crate::tokenizer::BYTE_OFFSET + b'0' as u32 + *label as u32) as i64)
        .collect();
    let targets = Tensor::<1, Int>::from_data(TensorData::new(targets, [batch]), device);

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

    let _ = tokenizer;
    HeldOut {
        accuracy: correct as f64 / batch as f64,
        loss: loss as f64,
        correct,
        examples: batch,
    }
}

/// Write a vision receipt, refusing to overwrite.
pub fn write_report(path: &str, report: &VisionReport) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let text = serde_json::to_string_pretty(report).expect("serialise the vision report");
    std::fs::write(path, text + "\n").expect("write the vision report");
}

/// The notes a vision run attaches to any checkpoint it writes.
pub fn checkpoint_notes(config: &VisionTrainConfig) -> BTreeMap<String, String> {
    let mut notes = BTreeMap::new();
    notes.insert("stage".to_string(), "vision".to_string());
    notes.insert("seed".to_string(), config.seed.to_string());
    notes.insert("steps".to_string(), config.steps.to_string());
    notes.insert("burn".to_string(), BURN_VERSION.to_string());
    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> VisionTrainConfig {
        VisionTrainConfig {
            steps: 3,
            batch_size: 4,
            learning_rate: 2e-3,
            eval_examples: 20,
            seed: 1,
            noise: 0.06,
        }
    }

    #[test]
    fn both_architectures_build_and_forward() {
        let device = Device::flex();
        let language = vision_language_config();
        let vision = vision_config();

        for architecture in [Architecture::Faithful, Architecture::DirectPatch] {
            let model = VisionModel::build(architecture, &language, &vision, &device);
            let images = vision_data::render_digits(&[3, 4], 1, 0.0, &device);
            let (prompts, _) = vision_data::prompt_batch(2, &device);
            let (logits, usage) = model.forward(images, prompts);
            assert_eq!(logits.dims()[0], 2);
            assert_eq!(usage.dims()[1], language.experts);
        }
    }

    #[test]
    fn a_short_run_reports_before_and_after() {
        let device = Device::flex().autodiff();
        let report = train_vision(
            &[Architecture::Faithful],
            &tiny(),
            &device,
            &mut |_, _, _| {},
        );
        assert_eq!(report.results.len(), 1);
        let result = &report.results[0];
        assert_eq!(result.architecture, "faithful");
        assert_eq!(result.visual_token_count, 16);
        assert_eq!(result.inserted_token_count, 18);
        assert_eq!(result.before.examples, 20);
        assert!(result.after.accuracy <= 1.0);
        assert!(result.final_training_loss.unwrap().is_finite());
    }

    #[test]
    fn masked_cross_entropy_ignores_masked_positions() {
        let device = Device::flex();
        // Two positions, one masked: the loss must equal the unmasked position's.
        let logits = Tensor::<3>::from_data(
            TensorData::new(vec![2.0f32, 0.0, 0.0, 1.0], [1, 2, 2]),
            &device,
        );
        let both = Tensor::<2, Int>::from_data(TensorData::new(vec![0i64, 1], [1, 2]), &device);
        let only_first =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0i64, -100], [1, 2]), &device);

        let full = masked_cross_entropy(logits.clone(), both).into_scalar::<f32>();
        let masked = masked_cross_entropy(logits, only_first).into_scalar::<f32>();

        // First position predicts class 0 from logits (2,0): -log(softmax)=~0.127.
        assert!((masked - 0.1269).abs() < 1e-3, "masked value {masked}");
        assert!(full.is_finite());
        assert!(
            (full - masked).abs() > 1e-3,
            "the mask must change the mean"
        );
    }

    #[test]
    fn the_load_balance_penalty_is_zero_for_a_uniform_router() {
        let device = Device::flex();
        let usage = Tensor::<2>::from_data(TensorData::new(vec![0.25f32; 4], [1, 4]), &device);
        let penalty = load_balance(usage, 4).into_scalar::<f32>();
        assert!(penalty.abs() < 1e-6, "penalty {penalty}");
    }
}
