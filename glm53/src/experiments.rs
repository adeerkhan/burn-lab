//! The two matched miniature pretraining studies.
//!
//! Mirrors `experiments/pretraining_data_diversity.py` and
//! `experiments/pretraining_curriculum_order.py`: identical model, initialisation
//! within a seed, optimiser, step count and batch size, with only the data
//! presentation changing.

use std::{collections::BTreeMap, time::Instant};

use burn::{
    optim::{AdamWConfig, GradientsParams, grad_clipping::GradientClippingConfig},
    prelude::*,
};
use serde::Serialize;

use crate::{
    arithmetic::{self, Batch, Score, Structure},
    model::GLM53Flash,
    tokenizer::ByteTokenizer,
    vision_train::masked_cross_entropy,
};

/// Settings shared by both studies.
#[derive(Clone, Debug)]
pub struct StudyConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub sequence_length: usize,
    pub seeds: Vec<u64>,
}

impl Default for StudyConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            batch_size: 24,
            learning_rate: 8e-4,
            sequence_length: 128,
            seeds: vec![11, 22, 33],
        }
    }
}

/// One training arm's outcome.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct ArmResult {
    pub arm: String,
    pub seed: u64,
    pub steps: usize,
    pub batch_size: usize,
    pub tokens_seen: usize,
    pub elapsed_seconds: f64,
    pub before: Score,
    pub after: Score,
    pub training_curve: Vec<CurvePoint>,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct CurvePoint {
    pub step: usize,
    pub training_loss: f64,
    pub tokens_seen: usize,
}

/// A study receipt.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct StudyReport {
    pub schema_version: String,
    pub status: String,
    pub research_question: String,
    pub independent_variable: String,
    pub controlled: Vec<String>,
    pub device: String,
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub sequence_length: usize,
    pub arms: Vec<String>,
    pub seeds: Vec<u64>,
    pub held_out_structures: Vec<String>,
    pub elapsed_seconds: f64,
    pub environment: serde_json::Value,
    pub runs: Vec<ArmResult>,
}

/// One planned training example.
type Item = (Structure, usize);

/// Run the data-diversity study: 8, 32 and 88 structures at fixed compute.
pub fn data_diversity(diversities: &[usize], config: &StudyConfig, device: &Device) -> StudyReport {
    let structures = arithmetic::all_structures();
    let train_pool: Vec<Structure> = structures[..88].to_vec();
    let eval_pool: Vec<Structure> = structures[88..120].to_vec();
    let total = config.steps * config.batch_size;

    let mut runs = Vec::new();
    for diversity in diversities {
        assert!(*diversity <= train_pool.len(), "diversity exceeds the pool");
        for seed in &config.seeds {
            let plan: Vec<Item> = (0..total)
                .map(|index| (train_pool[index % diversity], index))
                .collect();
            runs.push(train_arm(
                &format!("{diversity} structures"),
                *seed,
                &plan,
                config,
                &eval_pool,
                device,
            ));
        }
    }

    report(
        "At fixed compute, does more unique program structure improve held-out compositional code prediction?",
        "number of unique training expression structures",
        &diversities
            .iter()
            .map(|d| format!("{d} structures"))
            .collect::<Vec<_>>(),
        &eval_pool,
        config,
        runs,
    )
}

/// Run the curriculum and ordering study.
pub fn curriculum_order(
    conditions: &[String],
    config: &StudyConfig,
    device: &Device,
) -> StudyReport {
    let structures = arithmetic::all_structures();
    let train_pool: Vec<Structure> = structures[..88].to_vec();
    let eval_pool: Vec<Structure> = structures[88..120].to_vec();
    let total = config.steps * config.batch_size;

    let mut runs = Vec::new();
    for condition in conditions {
        let plan = plan_for(condition, &train_pool, total, config);
        for seed in &config.seeds {
            runs.push(train_arm(
                condition, *seed, &plan, config, &eval_pool, device,
            ));
        }
    }

    report(
        "At fixed compute and identical examples, does the order of presentation change held-out performance?",
        "example ordering and curriculum schedule",
        conditions,
        &eval_pool,
        config,
        runs,
    )
}

fn plan_for(
    condition: &str,
    train_pool: &[Structure],
    total: usize,
    config: &StudyConfig,
) -> Vec<Item> {
    let mut items: Vec<Item> = (0..total)
        .map(|index| (train_pool[index % train_pool.len()], index))
        .collect();

    match condition {
        "diverse-interleaved" => items,
        "diverse-blocked" => {
            let rank: BTreeMap<Structure, usize> = train_pool
                .iter()
                .enumerate()
                .map(|(index, structure)| (*structure, index))
                .collect();
            items.sort_by_key(|(structure, index)| (rank[structure], *index));
            items
        }
        "curriculum-8-to-88" => {
            let switch = (config.steps / 2) * config.batch_size;
            let repeated = &train_pool[..8];
            items
                .into_iter()
                .map(|(_, index)| {
                    let pool = if index < switch { repeated } else { train_pool };
                    (pool[index % pool.len()], index)
                })
                .collect()
        }
        other => panic!("unknown condition `{other}`"),
    }
}

fn train_arm(
    arm: &str,
    seed: u64,
    plan: &[Item],
    config: &StudyConfig,
    eval_pool: &[Structure],
    device: &Device,
) -> ArmResult {
    let started = Instant::now();
    let model_config = arithmetic::miniature_config(config.sequence_length);
    let mut model = GLM53Flash::new(&model_config, device);

    let mut optimizer = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let before = arithmetic::evaluate(&model.valid(), eval_pool, seed, device);

    let mut curve = Vec::new();
    let mut tokens_seen = 0usize;
    let mut final_loss = 0.0f64;

    model = model.train();
    for step in 0..config.steps {
        let batch: Batch = batch_from_items(
            &plan[step * config.batch_size..(step + 1) * config.batch_size],
            seed,
            config.sequence_length,
            device,
        );

        let (logits, usage) = model.forward(batch.inputs);
        let language_loss = masked_cross_entropy(logits, batch.labels);
        let balance = load_balance(usage, model_config.experts);
        let loss = language_loss.clone().add(balance.mul_scalar(0.01));

        final_loss = language_loss.into_scalar::<f32>() as f64;
        tokens_seen += batch.tokens;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if step == 0 || (step + 1) % 20 == 0 {
            curve.push(CurvePoint {
                step: step + 1,
                training_loss: final_loss,
                tokens_seen,
            });
        }
    }

    let after = arithmetic::evaluate(&model.valid(), eval_pool, seed, device);
    let _ = final_loss;

    ArmResult {
        arm: arm.to_string(),
        seed,
        steps: config.steps,
        batch_size: config.batch_size,
        tokens_seen,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        before,
        after,
        training_curve: curve,
    }
}

fn batch_from_items(items: &[Item], seed: u64, sequence_length: usize, device: &Device) -> Batch {
    let tokenizer = ByteTokenizer::new();
    let width = sequence_length + 1;
    let mut rows: Vec<i64> = Vec::with_capacity(items.len() * width);
    let mut valid = 0usize;

    for (structure, example_id) in items {
        let (text, _) = arithmetic::render_example(structure, *example_id, seed);
        let mut ids: Vec<i64> = tokenizer
            .encode(&text, true, true)
            .into_iter()
            .map(|token| token as i64)
            .collect();
        assert!(
            ids.len() <= width,
            "example is {} tokens, limit {width}",
            ids.len()
        );
        ids.resize(width, 0);
        valid += ids.iter().filter(|token| **token != 0).count();
        rows.extend(ids);
    }

    let batch = items.len();
    let mut inputs = Vec::with_capacity(batch * sequence_length);
    let mut labels = Vec::with_capacity(batch * sequence_length);
    for row in 0..batch {
        let base = row * width;
        inputs.extend_from_slice(&rows[base..base + sequence_length]);
        for column in 0..sequence_length {
            let token = rows[base + column + 1];
            labels.push(if token == 0 { -100 } else { token });
        }
    }

    Batch {
        inputs: Tensor::<2, Int>::from_data(
            TensorData::new(inputs, [batch, sequence_length]),
            device,
        ),
        labels: Tensor::<2, Int>::from_data(
            TensorData::new(labels, [batch, sequence_length]),
            device,
        ),
        tokens: valid,
    }
}

fn load_balance(usage: Tensor<2>, experts: usize) -> Tensor<1> {
    let [layers, _] = usage.dims();
    let mean = usage.reshape::<2, _>([layers, experts]).mean_dim(0);
    mean.sub_scalar(1.0 / experts as f32)
        .square()
        .mean()
        .mul_scalar(experts as f32)
}

fn report(
    question: &str,
    independent: &str,
    arms: &[String],
    eval_pool: &[Structure],
    config: &StudyConfig,
    runs: Vec<ArmResult>,
) -> StudyReport {
    StudyReport {
        schema_version: "1.0".to_string(),
        status: "complete".to_string(),
        research_question: question.to_string(),
        independent_variable: independent.to_string(),
        controlled: [
            "model",
            "initialisation within seed",
            "optimizer",
            "steps",
            "batch size",
            "tokenizer",
            "evaluation set",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect(),
        device: "flex".to_string(),
        steps: config.steps,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        sequence_length: config.sequence_length,
        arms: arms.to_vec(),
        seeds: config.seeds.clone(),
        held_out_structures: eval_pool
            .iter()
            .map(|structure| {
                structure
                    .iter()
                    .map(|operation| arithmetic::OPS[*operation])
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .collect(),
        elapsed_seconds: runs.iter().map(|run| run.elapsed_seconds).sum(),
        environment: crate::receipt::system_receipt(),
        runs,
    }
}

/// Write a study receipt, refusing to overwrite.
pub fn write_report(path: &str, report: &StudyReport) -> Result<(), String> {
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

    fn tiny() -> StudyConfig {
        StudyConfig {
            steps: 2,
            batch_size: 2,
            learning_rate: 8e-4,
            sequence_length: 128,
            seeds: vec![11],
        }
    }

    #[test]
    fn the_diversity_study_runs_every_arm() {
        let device = Device::flex().autodiff();
        let report = data_diversity(&[8], &tiny(), &device);
        assert_eq!(report.runs.len(), 1);
        assert_eq!(report.runs[0].arm, "8 structures");
        assert_eq!(report.runs[0].steps, 2);
        assert!(report.runs[0].after.target_byte_accuracy <= 1.0);
    }

    #[test]
    fn blocked_and_interleaved_use_the_same_examples() {
        let pool = arithmetic::all_structures()[..8].to_vec();
        let config = StudyConfig {
            steps: 4,
            batch_size: 3,
            ..tiny()
        };
        let interleaved = plan_for("diverse-interleaved", &pool, 12, &config);
        let blocked = plan_for("diverse-blocked", &pool, 12, &config);
        assert_eq!(interleaved.len(), blocked.len());

        let mut a: Vec<usize> = interleaved
            .iter()
            .map(|(structure, _)| structure[0])
            .collect();
        let mut b: Vec<usize> = blocked.iter().map(|(structure, _)| structure[0]).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "the two orderings must present the same examples");
    }

    #[test]
    fn the_curriculum_starts_narrow() {
        let pool = arithmetic::all_structures()[..88].to_vec();
        let config = StudyConfig {
            steps: 4,
            batch_size: 5,
            ..tiny()
        };
        let plan = plan_for("curriculum-8-to-88", &pool, 20, &config);
        let first_half: Vec<Structure> = plan[..10].iter().map(|(s, _)| *s).collect();
        let allowed: Vec<Structure> = pool[..8].to_vec();
        assert!(
            first_half
                .iter()
                .all(|structure| allowed.contains(structure)),
            "the first half must draw from the narrow pool"
        );
    }
}
