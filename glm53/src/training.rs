//! A custom pretraining loop: next-byte cross-entropy over the synthetic corpus.
//!
//! This does not use Burn's `Learner`, because the reference's training loop is
//! a plain optimizer step over a byte-level language-model loss.

use burn::{
    data::dataloader::batcher::Batcher,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams},
    prelude::*,
};

use std::collections::BTreeMap;

use crate::{
    data::{ByteBatcher, Corpus, Document},
    model::GLM53Flash,
    receipt,
    tasks::Rng,
};

/// Training hyperparameters.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub corpus_size: usize,
    pub log_every: usize,
    /// Steps at which to write a checkpoint directory.
    pub checkpoints: Vec<usize>,
    /// Where checkpoints are written. `None` skips checkpointing.
    pub artifact_dir: Option<String>,
    /// Steps already taken before this call.
    ///
    /// The sampler is advanced by `start_step * batch_size` draws so a continued
    /// run sees the batches it would have seen without the interruption. The
    /// optimizer still restarts, which is the part a checkpoint cannot restore
    /// today (see the resume gap in the README).
    pub start_step: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 400,
            // The MoE evaluates every expert densely, which keeps
            // `[tokens, experts, dim]` activations live for the backward pass.
            // 16 x 128 tokens fits a 6 GiB laptop GPU; 32 x 128 does not.
            batch_size: 16,
            sequence_length: 128,
            learning_rate: 3e-4,
            seed: 42,
            corpus_size: 512,
            log_every: 25,
            checkpoints: Vec::new(),
            artifact_dir: None,
            start_step: 0,
        }
    }
}

/// What one training step produced.
#[derive(Clone, Debug)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    /// One row per layer: the share of routing slots each expert took.
    pub usage: Vec<f32>,
}

/// Train a model from its current weights.
pub fn train(
    model: GLM53Flash,
    config: &TrainConfig,
    device: &Device,
    mut on_step: impl FnMut(&StepMetrics),
) -> GLM53Flash {
    let mut model = model;
    let batcher = ByteBatcher::new(config.sequence_length);
    let corpus = Corpus::new(config.corpus_size, config.seed);

    let mut optimizer = AdamWConfig::new().init();
    let mut rng = Rng::new(config.seed ^ 0x5DEECE66D);
    // Replay the sampler so the data order continues rather than restarting.
    for _ in 0..config.start_step.saturating_mul(config.batch_size) {
        rng.below(corpus.len());
    }

    for step in (config.start_step + 1)..=(config.start_step + config.steps) {
        let documents: Vec<Document> = (0..config.batch_size)
            .map(|_| {
                corpus
                    .get(rng.below(corpus.len()))
                    .expect("corpus index")
                    .clone()
            })
            .collect();

        let batch = batcher.batch(documents, device);
        let tokens = batch.tokens;
        let targets = batch.targets;
        let [batch_size, width] = tokens.dims();

        let (logits, usage) = model.forward(tokens);

        let logits = logits.reshape::<2, _>([batch_size * width, model.config().vocab_size]);
        let targets = targets.reshape::<1, _>([batch_size * width]);

        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits, targets);

        let loss_value = loss.clone().into_scalar::<f32>();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        // `log_every == 0` disables periodic logging; without the guard the
        // remainder below is a division by zero.
        let periodic = config.log_every > 0 && step % config.log_every == 0;
        if periodic || step == 1 || step == config.steps {
            on_step(&StepMetrics {
                step,
                loss: loss_value,
                usage: usage.into_data().as_slice::<f32>().unwrap().to_vec(),
            });
        }

        if config.checkpoints.contains(&step) {
            save_checkpoint(&model, config.artifact_dir.as_deref(), step);
        }
    }

    model
}

/// Write `<dir>/checkpoint-<step>/` with weights, config and a receipt.
pub fn save_checkpoint(model: &GLM53Flash, directory: Option<&str>, step: usize) {
    save_checkpoint_staged(model, directory, step, "pretrain", BTreeMap::new());
}

/// Write a checkpoint tagged with the stage that produced it.
pub fn save_checkpoint_staged(
    model: &GLM53Flash,
    directory: Option<&str>,
    step: usize,
    stage: &str,
    notes: BTreeMap<String, String>,
) {
    let Some(directory) = directory else {
        return;
    };
    let path = receipt::checkpoint_path(directory, step);
    receipt::prepare(&path).expect("create the checkpoint directory");

    model
        .valid()
        .clone()
        .into_record()
        .save(path.join("model"))
        .expect("save the checkpoint");

    model
        .config()
        .clone()
        .save(path.join("config.json"))
        .expect("save the model config");

    receipt::write_receipt(&path, stage, step, notes).expect("write the checkpoint receipt");
}

/// List the checkpoint directories under `root`, oldest step first.
pub fn list_checkpoints(root: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(step) = name.strip_prefix("checkpoint-") else {
            continue;
        };
        let Ok(step) = step.parse::<usize>() else {
            continue;
        };
        if path.join("model.bpk").exists() {
            found.push((step, path.to_string_lossy().into_owned()));
        }
    }
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    fn tiny() -> ModelConfig {
        ModelConfig::new()
            .with_dim(32)
            .with_heads(2)
            .with_layers(1)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_max_sequence_length(16)
    }

    /// `log_every = 0` means "no periodic logging". It used to divide by zero.
    #[test]
    fn zero_log_every_disables_periodic_logging() {
        let device = Device::flex().autodiff();
        let config = TrainConfig {
            steps: 4,
            batch_size: 2,
            sequence_length: 8,
            learning_rate: 1e-3,
            seed: 3,
            corpus_size: 16,
            log_every: 0,
            checkpoints: Vec::new(),
            artifact_dir: None,
            start_step: 0,
        };

        let mut seen = Vec::new();
        let model = GLM53Flash::new(&tiny(), &device);
        let _ = train(model, &config, &device, |metrics| seen.push(metrics.step));

        assert_eq!(
            seen,
            vec![1, 4],
            "only the first and last step should report"
        );
    }

    /// Every step reports when `log_every` is 1.
    #[test]
    fn log_every_one_reports_every_step() {
        let device = Device::flex().autodiff();
        let config = TrainConfig {
            steps: 3,
            batch_size: 2,
            sequence_length: 8,
            learning_rate: 1e-3,
            seed: 3,
            corpus_size: 16,
            log_every: 1,
            checkpoints: Vec::new(),
            artifact_dir: None,
            start_step: 0,
        };

        let mut seen = Vec::new();
        let model = GLM53Flash::new(&tiny(), &device);
        let _ = train(model, &config, &device, |metrics| seen.push(metrics.step));

        assert_eq!(seen, vec![1, 2, 3]);
    }
}
