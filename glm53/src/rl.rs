//! RLOO post-training with executable unit-test rewards.
//!
//! Mirrors `scripts/train_rl.py`: sample a group of completions per prompt,
//! score each with the executable reward, turn the scores into leave-one-out
//! advantages, and take one policy-gradient step.

use std::time::Instant;

use burn::{
    optim::{AdamWConfig, GradientsParams, grad_clipping::GradientClippingConfig},
    prelude::*,
    tensor::activation,
};
use serde::Serialize;

use crate::{
    evaluator::{self, Evaluation, Request},
    generate::{Decoding, Generation, GroupConfig, generate_group},
    model::GLM53Flash,
    tasks::{CodingTask, Rng},
    training::save_checkpoint,
};

/// Which parameters receive gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainScope {
    /// The whole model.
    All,
    /// The last transformer block and the final norm.
    LastBlock,
    /// The last transformer block, the final norm and the tied embedding/head.
    LastBlockHead,
}

impl TrainScope {
    pub fn parse(value: &str) -> Self {
        match value {
            "all" => Self::All,
            "last-block" => Self::LastBlock,
            "last-block-head" => Self::LastBlockHead,
            other => panic!("unknown train scope `{other}`"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::LastBlock => "last-block",
            Self::LastBlockHead => "last-block-head",
        }
    }
}

/// How a rollout is scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardMode {
    /// 1 when every test passes, otherwise the invalid penalty or 0.
    Binary,
    /// The fraction of test cases passed, plus the bonuses.
    CaseFraction,
}

impl RewardMode {
    pub fn parse(value: &str) -> Self {
        match value {
            "binary" => Self::Binary,
            "case-fraction" => Self::CaseFraction,
            other => panic!("unknown reward mode `{other}`"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::CaseFraction => "case-fraction",
        }
    }
}

/// RL hyperparameters.
#[derive(Clone, Debug)]
pub struct RlConfig {
    pub groups: usize,
    pub group_size: usize,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub learning_rate: f64,
    pub train_scope: TrainScope,
    pub reward_mode: RewardMode,
    pub invalid_penalty: f64,
    pub exact_bonus: f64,
    pub tasks_per_family: usize,
    pub families: Vec<String>,
    pub seed: u64,
    pub checkpoints: Vec<usize>,
    pub artifact_dir: String,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            groups: 32,
            group_size: 4,
            max_new_tokens: 48,
            temperature: 0.8,
            learning_rate: 2e-5,
            train_scope: TrainScope::LastBlockHead,
            reward_mode: RewardMode::Binary,
            invalid_penalty: -0.1,
            exact_bonus: 0.0,
            tasks_per_family: 4,
            families: Vec::new(),
            seed: 31_415,
            checkpoints: Vec::new(),
            artifact_dir: "artifacts/glm53-rl".to_string(),
        }
    }
}

/// What one RLOO group produced.
#[derive(Clone, Debug, Serialize)]
pub struct GroupMetrics {
    pub group: usize,
    pub task_id: String,
    pub family: String,
    pub rewards: Vec<f64>,
    pub advantages: Vec<f64>,
    pub reward_spread: f64,
    pub updated: bool,
    pub policy_loss: Option<f64>,
    pub exact_rollouts: usize,
    pub valid_rollouts: usize,
    pub token_cap_hits: usize,
    pub seconds: f64,
}

/// Leave-one-out advantages: `r_i - mean(r_others)`.
pub fn leave_one_out(rewards: &[f64]) -> Vec<f64> {
    assert!(rewards.len() >= 2, "RLOO needs a group of at least two");
    let total: f64 = rewards.iter().sum();
    let count = rewards.len() as f64;
    rewards
        .iter()
        .map(|reward| reward - (total - reward) / (count - 1.0))
        .collect()
}

/// Score one rollout.
pub fn reward_for(evaluation: &Evaluation, config: &RlConfig) -> f64 {
    let invalid = evaluation.status == "invalid";
    match config.reward_mode {
        RewardMode::Binary => {
            if evaluation.passed {
                1.0
            } else if invalid {
                config.invalid_penalty
            } else {
                0.0
            }
        }
        RewardMode::CaseFraction => {
            let mut reward = evaluation.pass_fraction;
            if evaluation.passed {
                reward += config.exact_bonus;
            }
            if invalid {
                reward += config.invalid_penalty;
            }
            reward
        }
    }
}

/// Build the group schedule: shuffled passes over `tasks` until `groups` long.
pub fn schedule(tasks: &[CodingTask], groups: usize, seed: u64) -> Vec<CodingTask> {
    let mut result = Vec::with_capacity(groups);
    let mut epoch = 0u64;
    while result.len() < groups {
        let mut block = tasks.to_vec();
        let mut rng = Rng::new(seed + epoch * 1009);
        for index in (1..block.len()).rev() {
            let swap = rng.below(index + 1);
            block.swap(index, swap);
        }
        result.extend(block);
        epoch += 1;
    }
    result.truncate(groups);
    result
}

/// Freeze everything outside `scope`.
pub fn apply_scope(model: GLM53Flash, scope: TrainScope) -> GLM53Flash {
    if scope == TrainScope::All {
        return model;
    }

    let last_block = model.last_block_group();
    let final_norm = model.final_norm_group();
    let embedding = model.embedding_group();

    let mut model = model.set_require_grad(false);
    model = model.set_require_grad_group(last_block, true);
    model = model.set_require_grad_group(final_norm, true);
    if scope == TrainScope::LastBlockHead {
        model = model.set_require_grad_group(embedding, true);
    }
    model
}

/// Score a rollout group in one model call.
///
/// Returns the mean per-token log-probability of each completion, shaped `[group]`.
pub fn completion_log_probabilities(
    model: &GLM53Flash,
    prompt: &[u32],
    completions: &[Vec<u32>],
    device: &Device,
) -> Tensor<1> {
    let group = completions.len();
    assert!(group > 0, "empty rollout group");
    assert!(
        completions.iter().all(|completion| !completion.is_empty()),
        "a completion was empty"
    );

    let longest = completions.iter().map(Vec::len).max().expect("non-empty");
    let width = prompt.len() + longest;

    let mut flat = Vec::with_capacity(group * width);
    for completion in completions {
        flat.extend(prompt.iter().map(|token| *token as i64));
        flat.extend(completion.iter().map(|token| *token as i64));
        flat.extend(std::iter::repeat_n(0i64, longest - completion.len()));
    }
    let input = Tensor::<2, Int>::from_data(TensorData::new(flat, [group, width]), device);

    // Everything but the final token predicts the next one.
    let trimmed = input.slice([0..group, 0..width - 1]);
    let (logits, _) = model.forward(trimmed);

    let start = prompt.len() - 1;
    let vocab = model.config().vocab_size;
    let selected = logits.slice([0..group, start..start + longest, 0..vocab]);
    let log_probs = activation::log_softmax(selected, 2);

    let mut targets = Vec::with_capacity(group * longest);
    let mut mask = Vec::with_capacity(group * longest);
    for completion in completions {
        targets.extend(completion.iter().map(|token| *token as i64));
        targets.extend(std::iter::repeat_n(0i64, longest - completion.len()));
        for position in 0..longest {
            mask.push(if position < completion.len() {
                1.0f32
            } else {
                0.0
            });
        }
    }

    let targets = Tensor::<2, Int>::from_data(TensorData::new(targets, [group, longest]), device);
    let mask = Tensor::<2>::from_data(TensorData::new(mask, [group, longest]), device);

    let picked = log_probs
        .gather(2, targets.unsqueeze_dim::<3>(2))
        .squeeze_dim::<2>(2);

    let lengths: Vec<f32> = completions
        .iter()
        .map(|completion| completion.len() as f32)
        .collect();
    let lengths = Tensor::<1>::from_data(TensorData::new(lengths, [group]), device);

    picked.mul(mask).sum_dim(1).squeeze_dim::<1>(1).div(lengths)
}

/// Run RLOO.
pub fn run_rl(
    model: GLM53Flash,
    tasks: &[CodingTask],
    config: &RlConfig,
    device: &Device,
    mut on_group: impl FnMut(&GroupMetrics),
) -> GLM53Flash {
    assert!(
        config.group_size >= 2,
        "RLOO requires a group of at least two"
    );

    let schedule = schedule(tasks, config.groups, config.seed);
    let mut model = apply_scope(model, config.train_scope);

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let tokenizer = crate::tokenizer::ByteTokenizer::new();

    for (index, task) in schedule.iter().enumerate() {
        let group_index = index + 1;
        let started = Instant::now();

        // Rollouts must not record the autodiff graph: nothing backpropagates
        // through them, and on a GPU the retained graphs exhaust the device.
        // This is the equivalent of the reference's `model.eval()` under
        // `torch.no_grad()`.
        let inference = model.valid();
        let generations: Vec<Generation> = generate_group(
            &inference,
            task,
            &GroupConfig {
                group_size: config.group_size,
                max_new_tokens: config.max_new_tokens,
                decoding: Decoding::Sample {
                    temperature: config.temperature,
                },
                seed: config.seed + group_index as u64 * 17,
                stop_on_parseable: true,
            },
            device,
        );

        let requests: Vec<Request> = generations
            .iter()
            .map(|generation| Request {
                family: task.family.clone(),
                entry_point: task.entry_point.clone(),
                prompt: task.prompt.clone(),
                completion: generation.completion.clone(),
                run_tests: true,
            })
            .collect();
        let evaluations = evaluator::evaluate_batch(&requests);

        let rewards: Vec<f64> = evaluations
            .iter()
            .map(|evaluation| reward_for(evaluation, config))
            .collect();
        let advantages = leave_one_out(&rewards);

        let highest = rewards.iter().cloned().fold(f64::MIN, f64::max);
        let lowest = rewards.iter().cloned().fold(f64::MAX, f64::min);
        let spread = highest - lowest;

        let updated = spread > 1e-8
            && generations
                .iter()
                .all(|generation| !generation.token_ids.is_empty());

        let mut policy_loss = None;

        if updated {
            let prompt = tokenizer.encode(&task.prompt, true, false);
            let completions: Vec<Vec<u32>> = generations
                .iter()
                .map(|generation| generation.token_ids.clone())
                .collect();

            let log_probabilities =
                completion_log_probabilities(&model, &prompt, &completions, device);
            let advantage_values: Vec<f32> = advantages.iter().map(|value| *value as f32).collect();
            let advantage_tensor = Tensor::<1>::from_data(
                TensorData::new(advantage_values, [advantages.len()]),
                device,
            );

            let objective = advantage_tensor.mul(log_probabilities).mean().neg();
            let value = objective.clone().into_scalar::<f32>();
            assert!(
                value.is_finite(),
                "non-finite RL objective at group {group_index}"
            );

            let grads = objective.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);

            policy_loss = Some(value as f64);
        }

        let metrics = GroupMetrics {
            group: group_index,
            task_id: task.task_id.clone(),
            family: task.family.clone(),
            rewards: rewards.clone(),
            advantages,
            reward_spread: spread,
            updated,
            policy_loss,
            exact_rollouts: evaluations
                .iter()
                .filter(|evaluation| evaluation.passed)
                .count(),
            valid_rollouts: evaluations
                .iter()
                .filter(|evaluation| evaluation.status != "invalid")
                .count(),
            token_cap_hits: generations
                .iter()
                .filter(|generation| generation.hit_token_cap)
                .count(),
            seconds: started.elapsed().as_secs_f64(),
        };
        on_group(&metrics);

        if config.checkpoints.contains(&group_index) {
            save_checkpoint(&model, Some(&config.artifact_dir), group_index);
        }
    }

    model
}

/// A `[layers, experts]` usage tensor flattened for reporting.
pub fn usage_rows(usage: &Tensor<2>, layers: usize, experts: usize) -> Vec<Vec<f32>> {
    let data = usage.clone().into_data();
    let values = data.as_slice::<f32>().unwrap_or(&[]);
    (0..layers)
        .map(|layer| {
            let start = layer * experts;
            values.get(start..start + experts).unwrap_or(&[]).to_vec()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GLM53Flash, ModelConfig};

    #[test]
    fn leave_one_out_centres_the_rewards() {
        // The winner beats the mean of the other two by 1.0; each loser sits
        // 0.5 below the mean of the other two.
        let advantages = leave_one_out(&[1.0, 0.0, 0.0]);
        assert!((advantages[0] - 1.0).abs() < 1e-9);
        assert!((advantages[1] + 0.5).abs() < 1e-9);
        assert!((advantages[2] + 0.5).abs() < 1e-9);
        // Advantages always sum to zero.
        assert!(advantages.iter().sum::<f64>().abs() < 1e-9);
    }

    #[test]
    fn leave_one_out_is_zero_for_identical_rewards() {
        let advantages = leave_one_out(&[0.4, 0.4, 0.4, 0.4]);
        assert!(advantages.iter().all(|value| value.abs() < 1e-9));
    }

    #[test]
    fn binary_reward_maps_the_three_outcomes() {
        let config = RlConfig {
            reward_mode: RewardMode::Binary,
            invalid_penalty: -0.1,
            ..RlConfig::default()
        };

        let passed = Evaluation {
            status: "passed".into(),
            passed: true,
            tests_passed: 3,
            tests_total: 3,
            pass_fraction: 1.0,
            message: String::new(),
            parseable: true,
        };
        let failed = Evaluation {
            status: "failed".into(),
            passed: false,
            tests_passed: 1,
            tests_total: 3,
            pass_fraction: 1.0 / 3.0,
            message: String::new(),
            parseable: true,
        };
        let invalid = Evaluation {
            status: "invalid".into(),
            passed: false,
            tests_passed: 0,
            tests_total: 3,
            pass_fraction: 0.0,
            message: "unsafe call".into(),
            parseable: false,
        };

        assert_eq!(reward_for(&passed, &config), 1.0);
        assert_eq!(reward_for(&failed, &config), 0.0);
        assert_eq!(reward_for(&invalid, &config), -0.1);
    }

    #[test]
    fn case_fraction_reward_uses_the_pass_fraction() {
        let config = RlConfig {
            reward_mode: RewardMode::CaseFraction,
            invalid_penalty: -0.1,
            exact_bonus: 0.5,
            ..RlConfig::default()
        };

        let partial = Evaluation {
            status: "failed".into(),
            passed: false,
            tests_passed: 2,
            tests_total: 3,
            pass_fraction: 2.0 / 3.0,
            message: String::new(),
            parseable: true,
        };
        let invalid = Evaluation {
            status: "invalid".into(),
            passed: false,
            tests_passed: 0,
            tests_total: 3,
            pass_fraction: 0.0,
            message: String::new(),
            parseable: false,
        };

        assert!((reward_for(&partial, &config) - 2.0 / 3.0).abs() < 1e-9);
        assert!((reward_for(&invalid, &config) + 0.1).abs() < 1e-9);
    }

    #[test]
    fn the_schedule_covers_its_quota() {
        let tasks = crate::tasks::frozen_tasks("rl", 1);
        let schedule = schedule(&tasks, 20, 7);
        assert_eq!(schedule.len(), 20);
        // Every task appears at least once.
        for task in &tasks {
            assert!(schedule.iter().any(|row| row.task_id == task.task_id));
        }
    }

    #[test]
    fn scopes_freeze_everything_outside_them() {
        let device = Device::flex().autodiff();
        let config = ModelConfig::new()
            .with_dim(32)
            .with_heads(2)
            .with_layers(3)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_max_sequence_length(64);
        let model = GLM53Flash::new(&config, &device);

        let all = model.num_trainable_params();
        assert_eq!(all, model.num_params());

        let last_block = apply_scope(model.clone(), TrainScope::LastBlock);
        assert!(last_block.num_trainable_params() < all);

        let with_head = apply_scope(model, TrainScope::LastBlockHead);
        assert!(with_head.num_trainable_params() > last_block.num_trainable_params());
    }

    #[test]
    fn log_probabilities_are_negative_and_length_normalised() {
        let device = Device::flex();
        let config = ModelConfig::new()
            .with_dim(32)
            .with_heads(2)
            .with_layers(2)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_max_sequence_length(64);
        let model = GLM53Flash::new(&config, &device);

        let prompt: Vec<u32> = vec![1, 14, 40, 20];
        let completions = vec![vec![30u32, 40, 50], vec![30, 40, 50, 60]];
        let log_probs = completion_log_probabilities(&model, &prompt, &completions, &device);

        assert_eq!(log_probs.dims(), [2]);
        let data = log_probs.into_data();
        for value in data.as_slice::<f32>().unwrap() {
            assert!(value.is_finite());
            // A mean log-probability over a 256-way vocabulary is negative.
            assert!(*value < 0.0);
        }
    }
}
