//! Executable evaluation over the frozen coding splits.
//!
//! Mirrors `scripts/evaluate.py` (greedy pass@1) and
//! `scripts/evaluate_passk.py` (sampled pass@k).

use std::time::Instant;

use burn::prelude::*;
use serde::Serialize;

use crate::{
    evaluator::{self, Evaluation, Request},
    generate::{Decoding, GroupConfig, generate_group},
    model::GLM53Flash,
    tasks::CodingTask,
};

/// One task's greedy episode.
#[derive(Clone, Debug, Serialize)]
pub struct Episode {
    pub task_id: String,
    pub family: String,
    pub generated_completion: String,
    pub generated_tokens: usize,
    pub hit_token_cap: bool,
    pub evaluation: Verdict,
}

/// A serialisable view of [`Evaluation`].
#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    pub status: String,
    pub passed: bool,
    pub tests_passed: usize,
    pub tests_total: usize,
    pub pass_fraction: f64,
    pub message: String,
}

impl From<Evaluation> for Verdict {
    fn from(evaluation: Evaluation) -> Self {
        Self {
            status: evaluation.status,
            passed: evaluation.passed,
            tests_passed: evaluation.tests_passed,
            tests_total: evaluation.tests_total,
            pass_fraction: evaluation.pass_fraction,
            message: evaluation.message,
        }
    }
}

/// Per-family tallies.
#[derive(Clone, Debug, Serialize)]
pub struct FamilyScore {
    pub family: String,
    pub tasks: usize,
    /// Tasks solved: pass@1, or solved by at least one sample for pass@k.
    pub solved: usize,
    /// Individual rollouts that passed every test.
    pub exact_rollouts: usize,
    /// Rollouts run.
    pub rollouts: usize,
    /// Rollouts that produced any valid program.
    pub valid_rollouts: usize,
}

/// Greedy pass@1 result.
#[derive(Clone, Debug, Serialize)]
pub struct PassAtOne {
    pub tasks_solved: usize,
    pub tasks_total: usize,
    pub pass_at_1: f64,
    pub tests_passed: usize,
    pub tests_total: usize,
    pub test_accuracy: f64,
    pub valid_programs: usize,
    pub token_cap_hits: usize,
    pub elapsed_seconds: f64,
    pub by_family: Vec<FamilyScore>,
    pub episodes: Vec<Episode>,
}

/// Sampled pass@k result.
#[derive(Clone, Debug, Serialize)]
pub struct PassAtK {
    pub tasks_solved_at_k: usize,
    pub tasks_total: usize,
    pub pass_at_k: f64,
    pub samples_per_task: usize,
    pub exact_rollouts: usize,
    pub rollouts_total: usize,
    pub exact_rollout_rate: f64,
    pub tests_passed: usize,
    pub tests_total: usize,
    pub test_accuracy: f64,
    pub valid_rollouts: usize,
    pub token_cap_hits: usize,
    pub elapsed_seconds: f64,
    pub by_family: Vec<FamilyScore>,
}

/// Greedy evaluation settings.
#[derive(Clone, Copy, Debug)]
pub struct GreedyConfig {
    pub max_new_tokens: usize,
    pub seed: u64,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            seed: 2026,
        }
    }
}

/// Sampled evaluation settings.
#[derive(Clone, Copy, Debug)]
pub struct SampledConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub samples: usize,
    pub seed: u64,
}

impl Default for SampledConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            temperature: 0.35,
            samples: 8,
            seed: 8_675_309,
        }
    }
}

/// Greedy pass@1 over `tasks`.
pub fn pass_at_one(
    model: &GLM53Flash,
    tasks: &[CodingTask],
    config: &GreedyConfig,
    device: &Device,
) -> PassAtOne {
    let started = Instant::now();

    let mut episodes = Vec::with_capacity(tasks.len());
    for (index, task) in tasks.iter().enumerate() {
        let generations = generate_group(
            model,
            task,
            &GroupConfig {
                group_size: 1,
                max_new_tokens: config.max_new_tokens,
                decoding: Decoding::Greedy,
                seed: config.seed + index as u64,
                stop_on_parseable: true,
            },
            device,
        );

        let generation = &generations[0];
        let evaluation = verify(task, &generation.completion);

        episodes.push(Episode {
            task_id: task.task_id.clone(),
            family: task.family.clone(),
            generated_completion: generation.completion.clone(),
            generated_tokens: generation.tokens,
            hit_token_cap: generation.hit_token_cap,
            evaluation: evaluation.into(),
        });
    }

    let solved = episodes
        .iter()
        .filter(|episode| episode.evaluation.passed)
        .count();
    let tests_passed: usize = episodes
        .iter()
        .map(|episode| episode.evaluation.tests_passed)
        .sum();
    let tests_total: usize = episodes
        .iter()
        .map(|episode| episode.evaluation.tests_total)
        .sum();

    let mut by_family = Vec::new();
    for family in families_of(tasks) {
        let rows: Vec<&Episode> = episodes.iter().filter(|row| row.family == family).collect();
        by_family.push(FamilyScore {
            family,
            tasks: rows.len(),
            solved: rows.iter().filter(|row| row.evaluation.passed).count(),
            exact_rollouts: rows.iter().filter(|row| row.evaluation.passed).count(),
            rollouts: rows.len(),
            valid_rollouts: rows
                .iter()
                .filter(|row| row.evaluation.status != "invalid")
                .count(),
        });
    }

    PassAtOne {
        tasks_solved: solved,
        tasks_total: episodes.len(),
        pass_at_1: ratio(solved, episodes.len()),
        tests_passed,
        tests_total,
        test_accuracy: ratio(tests_passed, tests_total),
        valid_programs: episodes
            .iter()
            .filter(|episode| episode.evaluation.status != "invalid")
            .count(),
        token_cap_hits: episodes
            .iter()
            .filter(|episode| episode.hit_token_cap)
            .count(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        by_family,
        episodes,
    }
}

/// Sampled pass@k over `tasks`.
pub fn pass_at_k(
    model: &GLM53Flash,
    tasks: &[CodingTask],
    config: &SampledConfig,
    device: &Device,
) -> PassAtK {
    let started = Instant::now();

    let mut solved_at_k = 0usize;
    let mut exact_rollouts = 0usize;
    let mut rollouts_total = 0usize;
    let mut valid_rollouts = 0usize;
    let mut token_cap_hits = 0usize;
    let mut tests_passed = 0usize;
    let mut tests_total = 0usize;
    let families = families_of(tasks);
    // (solved_at_k, exact_rollouts, valid_rollouts, rollouts)
    let mut family_rows: Vec<(usize, usize, usize, usize)> = vec![(0, 0, 0, 0); families.len()];

    for (task_index, task) in tasks.iter().enumerate() {
        let slot = families
            .iter()
            .position(|family| *family == task.family)
            .expect("family");

        let generations = generate_group(
            model,
            task,
            &GroupConfig {
                group_size: config.samples,
                max_new_tokens: config.max_new_tokens,
                decoding: Decoding::Sample {
                    temperature: config.temperature,
                },
                seed: config.seed + task_index as u64 * 104_729,
                stop_on_parseable: true,
            },
            device,
        );

        let verdicts: Vec<Evaluation> = generations
            .iter()
            .map(|generation| verify(task, &generation.completion))
            .collect();

        let task_solved = verdicts.iter().any(|verdict| verdict.passed);
        solved_at_k += usize::from(task_solved);

        for (generation, verdict) in generations.iter().zip(&verdicts) {
            rollouts_total += 1;
            exact_rollouts += usize::from(verdict.passed);
            valid_rollouts += usize::from(verdict.status != "invalid");
            token_cap_hits += usize::from(generation.hit_token_cap);
            tests_passed += verdict.tests_passed;
            tests_total += verdict.tests_total;
        }

        let row = &mut family_rows[slot];
        row.0 += usize::from(task_solved);
        row.1 += verdicts.iter().filter(|verdict| verdict.passed).count();
        row.2 += verdicts
            .iter()
            .filter(|verdict| verdict.status != "invalid")
            .count();
        row.3 += config.samples;
    }

    let by_family = families
        .into_iter()
        .zip(family_rows)
        .map(|(family, (solved, exact, valid, rollouts))| FamilyScore {
            tasks: tasks.iter().filter(|task| task.family == family).count(),
            family,
            solved,
            exact_rollouts: exact,
            rollouts,
            valid_rollouts: valid,
        })
        .collect();

    PassAtK {
        tasks_solved_at_k: solved_at_k,
        tasks_total: tasks.len(),
        pass_at_k: ratio(solved_at_k, tasks.len()),
        samples_per_task: config.samples,
        exact_rollouts,
        rollouts_total,
        exact_rollout_rate: ratio(exact_rollouts, rollouts_total),
        tests_passed,
        tests_total,
        test_accuracy: ratio(tests_passed, tests_total),
        valid_rollouts,
        token_cap_hits,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        by_family,
    }
}

/// Verify one completion against the task.
pub fn verify(task: &CodingTask, completion: &str) -> Evaluation {
    evaluator::evaluate(Request {
        family: task.family.clone(),
        entry_point: task.entry_point.clone(),
        prompt: task.prompt.clone(),
        completion: completion.to_string(),
        run_tests: true,
    })
}

fn families_of(tasks: &[CodingTask]) -> Vec<String> {
    let mut families: Vec<String> = tasks.iter().map(|task| task.family.clone()).collect();
    families.sort();
    families.dedup();
    families
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GLM53Flash, ModelConfig, tasks};

    fn tiny_config() -> ModelConfig {
        ModelConfig::new()
            .with_dim(32)
            .with_heads(2)
            .with_layers(2)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_max_sequence_length(192)
    }

    #[test]
    fn pass_at_one_reports_one_episode_per_task() {
        let device = Device::flex();
        let config = tiny_config();
        let model = GLM53Flash::new(&config, &device).valid();
        let all = tasks::frozen_tasks("dev", 1);

        let report = pass_at_one(
            &model,
            &all,
            &GreedyConfig {
                max_new_tokens: 4,
                seed: 1,
            },
            &device,
        );

        assert_eq!(report.tasks_total, all.len());
        assert_eq!(report.episodes.len(), all.len());
        assert_eq!(report.by_family.len(), 8);
        assert!(report.pass_at_1 <= 1.0);
        // A randomly initialised model cannot solve these.
        assert_eq!(report.tasks_solved, 0);
        assert_eq!(report.tests_total, all.len() * 3);
    }

    #[test]
    fn pass_at_k_runs_every_sample() {
        let device = Device::flex();
        let config = tiny_config();
        let model = GLM53Flash::new(&config, &device).valid();
        let all = tasks::frozen_tasks("dev", 1);

        let report = pass_at_k(
            &model,
            &all,
            &SampledConfig {
                max_new_tokens: 4,
                temperature: 1.0,
                samples: 3,
                seed: 5,
            },
            &device,
        );

        assert_eq!(report.tasks_total, all.len());
        assert_eq!(report.rollouts_total, all.len() * 3);
        assert_eq!(report.samples_per_task, 3);
    }
}
