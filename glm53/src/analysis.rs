//! Paired statistics for the miniature studies.
//!
//! Mirrors the analyzers in the reference: exact McNemar, a paired permutation
//! test, bootstrap confidence intervals and Holm correction.

use serde::Serialize;

use crate::experiments::StudyReport;

/// Two-sided exact McNemar test on discordant pair counts.
///
/// Returns `1.0` when there are no discordant pairs, which is the honest answer:
/// the data cannot distinguish the arms.
pub fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let tail = (0..=b.min(c)).map(|k| binomial(n, k)).sum::<f64>();
    (2.0 * tail).min(1.0)
}

fn binomial(n: usize, k: usize) -> f64 {
    // log C(n,k) - n log 2
    let mut log_term = 0.0f64;
    for index in 0..k {
        log_term += ((n - index) as f64).ln() - ((index + 1) as f64).ln();
    }
    (log_term - (n as f64) * std::f64::consts::LN_2).exp()
}

/// Two-sided paired permutation test on the mean difference.
///
/// Exhaustive for up to 12 pairs, otherwise a fixed number of random sign flips
/// seeded deterministically.
pub fn paired_permutation_p(differences: &[f64]) -> f64 {
    let observed: f64 = differences.iter().sum::<f64>().abs();
    if differences.is_empty() || observed == 0.0 {
        return 1.0;
    }

    let count = differences.len();
    if count > 12 {
        // Deterministic sampling above 12 pairs; exhaustive below.
        let mut rng = crate::tasks::Rng::new(0x5EED);
        let trials = 20_000;
        let mut extreme = 0usize;
        for _ in 0..trials {
            let sum: f64 = differences
                .iter()
                .map(|difference| {
                    if rng.below(2) == 0 {
                        *difference
                    } else {
                        -*difference
                    }
                })
                .sum();
            if sum.abs() >= observed - 1e-12 {
                extreme += 1;
            }
        }
        return (extreme as f64 / trials as f64).clamp(1.0 / trials as f64, 1.0);
    }

    let total = 1usize << count;
    let mut extreme = 0usize;
    for mask in 0..total {
        let sum: f64 = differences
            .iter()
            .enumerate()
            .map(|(index, difference)| {
                if mask & (1 << index) == 0 {
                    *difference
                } else {
                    -*difference
                }
            })
            .sum();
        if sum.abs() >= observed - 1e-12 {
            extreme += 1;
        }
    }
    extreme as f64 / total as f64
}

/// Percentile bootstrap confidence interval for the mean.
pub fn bootstrap_ci(values: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = crate::tasks::Rng::new(seed);
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.below(values.len())];
        }
        means.push(total / values.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let low = means[(0.025 * resamples as f64) as usize];
    let high = means[((0.975 * resamples as f64) as usize).min(resamples - 1)];
    (low, high)
}

/// Holm-Bonferroni step-down correction. Input order is preserved.
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    let count = p_values.len();
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|a, b| p_values[*a].partial_cmp(&p_values[*b]).unwrap());

    let mut adjusted = vec![0.0; count];
    let mut running = 0.0f64;
    for (rank, index) in order.iter().enumerate() {
        let scaled = (p_values[*index] * (count - rank) as f64).min(1.0);
        running = running.max(scaled);
        adjusted[*index] = running;
    }
    adjusted
}

/// One arm's summary.
#[derive(Clone, Debug, Serialize)]
pub struct ArmSummary {
    pub arm: String,
    pub seeds: usize,
    pub mean_byte_accuracy: f64,
    pub mean_exact_accuracy: f64,
    pub mean_loss: f64,
    pub std_byte_accuracy: f64,
}

/// One paired comparison against a reference arm.
#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub arm: String,
    pub against: String,
    pub mean_difference: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub permutation_p: f64,
    pub adjusted_p: f64,
}

/// The analyzed study.
#[derive(Clone, Debug, Serialize)]
pub struct Analysis {
    pub schema_version: String,
    pub research_question: String,
    pub independent_variable: String,
    pub arms: Vec<ArmSummary>,
    pub comparisons: Vec<Comparison>,
    pub reference_arm: String,
    pub note: String,
}

/// Analyze a study receipt.
pub fn analyze(report: &StudyReport) -> Analysis {
    let arms: Vec<String> = report.arms.clone();
    let reference = arms.first().cloned().unwrap_or_default();

    let summaries: Vec<ArmSummary> = arms
        .iter()
        .map(|arm| {
            let runs: Vec<&crate::experiments::ArmResult> =
                report.runs.iter().filter(|run| run.arm == *arm).collect();
            let accuracies: Vec<f64> = runs
                .iter()
                .map(|run| run.after.target_byte_accuracy)
                .collect();
            let exact: Vec<f64> = runs
                .iter()
                .map(|run| run.after.target_exact_accuracy)
                .collect();
            let losses: Vec<f64> = runs.iter().map(|run| run.after.target_loss).collect();
            ArmSummary {
                arm: arm.clone(),
                seeds: runs.len(),
                mean_byte_accuracy: mean(&accuracies),
                mean_exact_accuracy: mean(&exact),
                mean_loss: mean(&losses),
                std_byte_accuracy: std(&accuracies),
            }
        })
        .collect();

    let mut comparisons = Vec::new();
    let mut raw_p = Vec::new();

    for arm in arms.iter().filter(|arm| **arm != reference) {
        let pairs: Vec<(f64, f64)> = report
            .runs
            .iter()
            .filter(|run| run.arm == *arm)
            .filter_map(|run| {
                report
                    .runs
                    .iter()
                    .find(|other| other.arm == reference && other.seed == run.seed)
                    .map(|other| {
                        (
                            run.after.target_byte_accuracy,
                            other.after.target_byte_accuracy,
                        )
                    })
            })
            .collect();

        let differences: Vec<f64> = pairs.iter().map(|(a, b)| a - b).collect();
        let (low, high) = bootstrap_ci(&differences, 10_000, 0xB007);
        let p = paired_permutation_p(&differences);
        raw_p.push(p);

        comparisons.push(Comparison {
            arm: arm.clone(),
            against: reference.clone(),
            mean_difference: mean(&differences),
            ci_low: low,
            ci_high: high,
            permutation_p: p,
            adjusted_p: p,
        });
    }

    let adjusted = holm(&raw_p);
    for (comparison, value) in comparisons.iter_mut().zip(adjusted) {
        comparison.adjusted_p = value;
    }

    Analysis {
        schema_version: "1.0".to_string(),
        research_question: report.research_question.clone(),
        independent_variable: report.independent_variable.clone(),
        arms: summaries,
        comparisons,
        reference_arm: reference,
        note:
            "Paired by seed. Permutation p is two-sided; the adjusted p is Holm-corrected across \
               the comparisons against the reference arm."
                .to_string(),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcnemar_matches_the_textbook_example() {
        // b=1, c=9: the exact two-sided binomial p is 0.0215.
        let p = mcnemar_exact(1, 9);
        assert!((p - 0.021484).abs() < 1e-4, "p = {p}");
    }

    #[test]
    fn mcnemar_is_one_without_discordant_pairs() {
        assert_eq!(mcnemar_exact(0, 0), 1.0);
    }

    #[test]
    fn mcnemar_is_symmetric() {
        assert!((mcnemar_exact(1, 9) - mcnemar_exact(9, 1)).abs() < 1e-12);
    }

    #[test]
    fn a_consistent_gain_is_significant() {
        // Ten seeds, every one better by the same amount: exhaustive p = 2/1024.
        let differences = vec![0.05; 10];
        let p = paired_permutation_p(&differences);
        assert!(p < 0.01, "p = {p}");
    }

    #[test]
    fn a_null_difference_is_not_significant() {
        assert_eq!(paired_permutation_p(&[0.0, 0.0, 0.0]), 1.0);
    }

    #[test]
    fn holm_never_decreases_and_is_bounded() {
        let adjusted = holm(&[0.01, 0.04, 0.03]);
        assert!(adjusted.iter().all(|value| (0.0..=1.0).contains(value)));
        assert!(adjusted[0] <= adjusted[2], "ordered by rank");
        assert!((adjusted[0] - 0.03).abs() < 1e-12, "smallest becomes 3x");
    }

    #[test]
    fn the_bootstrap_interval_brackets_the_mean() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (low, high) = bootstrap_ci(&values, 5_000, 7);
        assert!(low <= 0.3 && 0.3 <= high, "interval {low}..{high}");
    }
}

/// One task's outcome, for paired tests across arms.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct TaskOutcome {
    pub task_id: String,
    pub passed: bool,
}

/// Bootstrap over tasks: resample tasks with replacement and report the central
/// 95% interval of the mean difference.
///
/// This is the reference's task-level interval: the unit of resampling is the
/// task, not the seed, because tasks are what the claim generalises over.
pub fn task_bootstrap(differences: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    bootstrap_ci(differences, resamples, seed)
}

/// A paired comparison between two arms over the same tasks.
#[derive(Clone, Debug, Serialize)]
pub struct PairedComparison {
    /// Tasks where only the treatment arm passed.
    pub treatment_only: usize,
    /// Tasks where only the reference arm passed.
    pub reference_only: usize,
    /// Tasks where both passed.
    pub both: usize,
    /// Tasks where neither passed.
    pub neither: usize,
    pub mean_difference: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub mcnemar_p: f64,
    pub adjusted_p: f64,
}

/// Compare every arm against a reference arm over paired task outcomes.
pub fn paired_variants(
    reference: &[TaskOutcome],
    arms: &[(String, Vec<TaskOutcome>)],
) -> Vec<(String, PairedComparison)> {
    let mut comparisons = Vec::new();
    let mut raw = Vec::new();

    for (name, outcomes) in arms {
        let mut treatment_only = 0usize;
        let mut reference_only = 0usize;
        let mut both = 0usize;
        let mut neither = 0usize;
        let mut differences = Vec::new();

        for outcome in reference {
            let Some(other) = outcomes.iter().find(|row| row.task_id == outcome.task_id) else {
                continue;
            };
            match (outcome.passed, other.passed) {
                (false, true) => treatment_only += 1,
                (true, false) => reference_only += 1,
                (true, true) => both += 1,
                (false, false) => neither += 1,
            }
            differences
                .push(f64::from(u8::from(other.passed)) - f64::from(u8::from(outcome.passed)));
        }

        let (low, high) = task_bootstrap(&differences, 10_000, 0xC0FFEE);
        let p = mcnemar_exact(treatment_only, reference_only);
        raw.push(p);
        comparisons.push((
            name.clone(),
            PairedComparison {
                treatment_only,
                reference_only,
                both,
                neither,
                mean_difference: mean(&differences),
                ci_low: low,
                ci_high: high,
                mcnemar_p: p,
                adjusted_p: p,
            },
        ));
    }

    let adjusted = holm(&raw);
    for ((_, comparison), value) in comparisons.iter_mut().zip(adjusted) {
        comparison.adjusted_p = value;
    }
    comparisons
}

/// The RL variant screen, analyzed.
#[derive(Clone, Debug, Serialize)]
pub struct VariantAnalysis {
    pub schema_version: String,
    pub reference_arm: String,
    pub arms: Vec<crate::rl::ScreenArmResult>,
    pub comparisons: Vec<(String, PairedComparison)>,
    pub note: String,
}

/// Analyze an RL variant screen against its first arm.
pub fn analyze_variants(arms: &[crate::rl::ScreenArmResult]) -> VariantAnalysis {
    let reference = arms.first().expect("at least one arm");
    let rest: Vec<(String, Vec<TaskOutcome>)> = arms
        .iter()
        .skip(1)
        .map(|arm| (arm.arm.clone(), arm.dev_outcomes.clone()))
        .collect();

    VariantAnalysis {
        schema_version: "1.0".to_string(),
        reference_arm: reference.arm.clone(),
        arms: arms.to_vec(),
        comparisons: paired_variants(&reference.dev_outcomes, &rest),
        note: "Exact McNemar over paired dev tasks; Holm-corrected across arms. The CI is a \
               task-level bootstrap."
            .to_string(),
    }
}

/// One group-size confirmation run.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct ConfirmationRun {
    pub group_size: usize,
    pub seed: u64,
    pub groups: usize,
    pub groups_with_signal: usize,
    pub exact_rollouts: usize,
    pub rollouts: usize,
    pub seconds: f64,
    pub dev_outcomes: Vec<TaskOutcome>,
}

/// The group-size confirmation, analyzed across seeds and tasks.
#[derive(Clone, Debug, Serialize)]
pub struct ConfirmationAnalysis {
    pub schema_version: String,
    pub per_seed: Vec<SeedComparison>,
    pub seed_mean_difference: f64,
    pub task_ci_low: f64,
    pub task_ci_high: f64,
    pub task_mean_difference: f64,
    pub note: String,
}

/// One seed's paired comparison between two group sizes.
#[derive(Clone, Debug, Serialize)]
pub struct SeedComparison {
    pub seed: u64,
    pub smaller: f64,
    pub larger: f64,
    pub difference: f64,
    pub mcnemar_p: f64,
}

/// Compare two group sizes across seeds, then across tasks.
///
/// `smaller` and `larger` are group sizes; every seed is compared on the same dev
/// tasks, and the seed-level differences are averaged before the task bootstrap.
pub fn analyze_group_confirmation(
    runs: &[ConfirmationRun],
    smaller: usize,
    larger: usize,
) -> ConfirmationAnalysis {
    let seeds: Vec<u64> = {
        let mut values: Vec<u64> = runs.iter().map(|run| run.seed).collect();
        values.sort_unstable();
        values.dedup();
        values
    };

    let mut per_seed = Vec::new();
    let mut task_differences = vec![0.0f64; 0];

    for seed in &seeds {
        let small = runs
            .iter()
            .find(|run| run.seed == *seed && run.group_size == smaller)
            .expect("the smaller group size ran on this seed");
        let large = runs
            .iter()
            .find(|run| run.seed == *seed && run.group_size == larger)
            .expect("the larger group size ran on this seed");

        let mut treatment_only = 0usize;
        let mut reference_only = 0usize;
        let mut small_pass = 0usize;
        let mut large_pass = 0usize;

        for outcome in &small.dev_outcomes {
            let Some(other) = large
                .dev_outcomes
                .iter()
                .find(|row| row.task_id == outcome.task_id)
            else {
                continue;
            };
            match (other.passed, outcome.passed) {
                (true, false) => treatment_only += 1,
                (false, true) => reference_only += 1,
                _ => {}
            }
            small_pass += usize::from(outcome.passed);
            large_pass += usize::from(other.passed);
        }

        if task_differences.len() != small.dev_outcomes.len() {
            task_differences = vec![0.0; small.dev_outcomes.len()];
        }
        for (index, outcome) in small.dev_outcomes.iter().enumerate() {
            let other = large
                .dev_outcomes
                .iter()
                .find(|row| row.task_id == outcome.task_id);
            if let Some(other) = other {
                task_differences[index] +=
                    f64::from(u8::from(other.passed)) - f64::from(u8::from(outcome.passed));
            }
        }

        let count = small.dev_outcomes.len() as f64;
        per_seed.push(SeedComparison {
            seed: *seed,
            smaller: small_pass as f64 / count,
            larger: large_pass as f64 / count,
            difference: (large_pass as f64 - small_pass as f64) / count,
            mcnemar_p: mcnemar_exact(treatment_only, reference_only),
        });
    }

    let seed_count = per_seed.len() as f64;
    let task_means: Vec<f64> = task_differences
        .iter()
        .map(|total| total / seed_count)
        .collect();
    let (low, high) = task_bootstrap(&task_means, 10_000, 0xBADF00D);

    ConfirmationAnalysis {
        schema_version: "1.0".to_string(),
        seed_mean_difference: mean(
            &per_seed
                .iter()
                .map(|row| row.difference)
                .collect::<Vec<_>>(),
        ),
        task_mean_difference: mean(&task_means),
        task_ci_low: low,
        task_ci_high: high,
        per_seed,
        note: format!(
            "group {larger} minus group {smaller}. Per-seed McNemar over paired dev tasks; the \
             interval resamples tasks after averaging within a task across seeds."
        ),
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;

    fn outcomes(bits: &[bool]) -> Vec<TaskOutcome> {
        bits.iter()
            .enumerate()
            .map(|(index, passed)| TaskOutcome {
                task_id: format!("task-{index}"),
                passed: *passed,
            })
            .collect()
    }

    #[test]
    fn paired_variants_counts_the_contingency_table() {
        let reference = outcomes(&[true, true, false, false, true]);
        let treatment = outcomes(&[true, false, true, false, true]);
        let comparisons = paired_variants(&reference, &[("arm".to_string(), treatment)]);

        let (_, comparison) = &comparisons[0];
        assert_eq!(comparison.both, 2);
        assert_eq!(comparison.neither, 1);
        assert_eq!(comparison.treatment_only, 1);
        assert_eq!(comparison.reference_only, 1);
        assert!((comparison.mean_difference - 0.0).abs() < 1e-12);
    }

    #[test]
    fn a_consistent_win_is_detected() {
        let reference = outcomes(&[false; 6]);
        let treatment = outcomes(&[true; 6]);
        let comparisons = paired_variants(&reference, &[("arm".to_string(), treatment)]);
        let (_, comparison) = &comparisons[0];
        assert_eq!(comparison.treatment_only, 6);
        assert!(comparison.mcnemar_p < 0.05, "p = {}", comparison.mcnemar_p);
        assert!((comparison.mean_difference - 1.0).abs() < 1e-12);
    }

    #[test]
    fn holm_applies_across_arms() {
        let reference = outcomes(&[false; 6]);
        let arms = vec![
            ("a".to_string(), outcomes(&[true; 6])),
            (
                "b".to_string(),
                outcomes(&[true, false, false, false, false, false]),
            ),
        ];
        let comparisons = paired_variants(&reference, &arms);
        assert!(comparisons[0].1.adjusted_p >= comparisons[0].1.mcnemar_p);
        assert!(comparisons[1].1.adjusted_p >= comparisons[1].1.mcnemar_p);
    }

    #[test]
    fn the_group_confirmation_averages_within_task() {
        let runs = vec![
            ConfirmationRun {
                group_size: 4,
                seed: 1,
                groups: 2,
                groups_with_signal: 1,
                exact_rollouts: 1,
                rollouts: 4,
                seconds: 0.0,
                dev_outcomes: outcomes(&[true, false, false, false]),
            },
            ConfirmationRun {
                group_size: 8,
                seed: 1,
                groups: 2,
                groups_with_signal: 2,
                exact_rollouts: 3,
                rollouts: 8,
                seconds: 0.0,
                dev_outcomes: outcomes(&[true, true, true, false]),
            },
            ConfirmationRun {
                group_size: 4,
                seed: 2,
                groups: 2,
                groups_with_signal: 1,
                exact_rollouts: 1,
                rollouts: 4,
                seconds: 0.0,
                dev_outcomes: outcomes(&[true, false, false, false]),
            },
            ConfirmationRun {
                group_size: 8,
                seed: 2,
                groups: 2,
                groups_with_signal: 2,
                exact_rollouts: 3,
                rollouts: 8,
                seconds: 0.0,
                dev_outcomes: outcomes(&[true, true, true, false]),
            },
        ];

        let analysis = analyze_group_confirmation(&runs, 4, 8);
        assert_eq!(analysis.per_seed.len(), 2);
        // Group 8 passes two more of four tasks in every seed.
        assert!((analysis.seed_mean_difference - 0.5).abs() < 1e-12);
        assert!((analysis.task_mean_difference - 0.5).abs() < 1e-12);
        assert!(analysis.task_ci_low <= analysis.task_ci_high);
    }
}

/// Turn receipts into one summary.
///
/// Mirrors `scripts/analyze_results.py`: read the immutable receipts written by
/// the runs and produce a single summary object. The charts stay in
/// `docs/make_plots.py`, which renders from this summary and the raw receipts.
pub fn analyze_results(directory: &str) -> serde_json::Value {
    use serde_json::json;

    let mut summary = serde_json::Map::new();
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    collect_json(std::path::Path::new(directory), &mut files);
    files.sort();

    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unnamed")
            .to_string();

        let headline = if let Some(pass) = value.get("pass_at_1") {
            json!({
                "kind": "pass@1",
                "pass_at_1": pass,
                "tasks_solved": value.get("tasks_solved"),
                "tasks_total": value.get("tasks_total"),
                "test_accuracy": value.get("test_accuracy"),
            })
        } else if let Some(pass) = value.get("pass_at_k") {
            json!({
                "kind": "pass@k",
                "pass_at_k": pass,
                "samples_per_task": value.get("samples_per_task"),
                "exact_rollout_rate": value.get("exact_rollout_rate"),
            })
        } else if value.get("after").is_some() && value.get("before").is_some() {
            json!({
                "kind": "before_after",
                "before": value["before"]["accuracy"],
                "after": value["after"]["accuracy"],
            })
        } else if value.get("research_question").is_some() {
            json!({
                "kind": "study",
                "research_question": value.get("research_question"),
                "arms": value.get("arms"),
                "runs": value.get("runs").and_then(|runs| runs.as_array()).map(|runs| runs.len()),
            })
        } else if value.get("results").is_some() {
            let rows: Vec<serde_json::Value> = value["results"]
                .as_array()
                .map(|results| {
                    results
                        .iter()
                        .map(|row| {
                            json!({
                                "architecture": row.get("architecture"),
                                "before": row.get("before").and_then(|b| b.get("accuracy")),
                                "after": row.get("after").and_then(|a| a.get("accuracy")),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            json!({ "kind": "vision", "architectures": rows })
        } else if value
            .as_array()
            .is_some_and(|rows| rows.first().and_then(|row| row.get("arm")).is_some())
        {
            let rows = value.as_array().expect("array");
            let arms: Vec<serde_json::Value> = rows
                .iter()
                .map(|row| {
                    json!({
                        "arm": row.get("arm"),
                        "groups_with_signal": row.get("groups_with_signal"),
                        "pass_at_one": row.get("pass_at_one"),
                    })
                })
                .collect();
            json!({ "kind": "rl_screen", "arms": arms })
        } else {
            continue;
        };

        summary.insert(name, headline);
    }

    json!({
        "schema_version": "1.0",
        "directory": directory,
        "receipts": summary.len(),
        "summary": summary,
    })
}

fn collect_json(directory: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json(&path, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            files.push(path);
        }
    }
}

#[cfg(test)]
mod results_tests {
    use super::*;

    #[test]
    fn the_summary_classifies_receipts() {
        let root = std::env::temp_dir().join("glm53-analyze-results");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        std::fs::write(
            root.join("pass1.json"),
            r#"{"pass_at_1":0.5,"tasks_solved":16,"tasks_total":32,"test_accuracy":0.6}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("study.json"),
            r#"{"research_question":"q","arms":["a","b"],"runs":[]}"#,
        )
        .unwrap();
        std::fs::write(root.join("ignore.txt"), "not json").unwrap();

        let summary = analyze_results(root.to_str().unwrap());
        assert_eq!(summary["receipts"], 2);
        assert_eq!(summary["summary"]["pass1"]["kind"], "pass@1");
        assert_eq!(summary["summary"]["study"]["kind"], "study");
    }
}

/// Golden values computed by hand, so a silent change in any statistic fails here
/// rather than in a published table.
#[cfg(test)]
mod golden_tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn mcnemar_matches_the_exact_binomial_tail() {
        // No discordant pairs cannot distinguish the arms.
        close(mcnemar_exact(0, 0), 1.0);
        // One discordant pair: p = 2 * C(1,0)/2 = 1.0.
        close(mcnemar_exact(1, 0), 1.0);
        // Six wins, no losses: p = 2 / 2^6 = 0.03125.
        close(mcnemar_exact(6, 0), 2.0_f64.powi(-5));
        // Exactly balanced: 2 * P(X <= 3) = 1.3125, clipped to 1.
        close(mcnemar_exact(3, 3), 1.0);
        // Ten against two: 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12 = 158/4096.
        close(mcnemar_exact(10, 2), 158.0 / 4096.0);
        // Symmetry in the argument order.
        close(mcnemar_exact(10, 2), mcnemar_exact(2, 10));
    }

    #[test]
    fn the_permutation_test_is_exhaustive_below_thirteen_pairs() {
        // A single pair: both signs are equally extreme, p = 2/2 = 1.
        close(paired_permutation_p(&[1.0]), 1.0);
        // Three identical positive differences: only +,+,+ and -,-,- reach the
        // observed magnitude, so p = 2/8.
        close(paired_permutation_p(&[1.0, 1.0, 1.0]), 0.25);
        // Four identical differences: p = 2/16.
        close(paired_permutation_p(&[2.0; 4]), 0.125);
        // No difference at all is not evidence.
        close(paired_permutation_p(&[0.0, 0.0, 0.0]), 1.0);
        // Sign symmetry: flipping every sign must not change the p-value.
        let differences = [1.0, -2.0, 3.0, 0.5, -1.0];
        let flipped: Vec<f64> = differences.iter().map(|value| -value).collect();
        close(
            paired_permutation_p(&differences),
            paired_permutation_p(&flipped),
        );
    }

    #[test]
    fn holm_is_step_down_and_order_preserving() {
        // Sorted: 0.01, 0.03, 0.04 -> 0.03, 0.06, 0.04, then enforce monotonicity
        // so the third becomes 0.06.
        let adjusted = holm(&[0.01, 0.04, 0.03]);
        assert!((adjusted[0] - 0.03).abs() < 1e-12, "{adjusted:?}");
        assert!((adjusted[1] - 0.06).abs() < 1e-12, "{adjusted:?}");
        assert!((adjusted[2] - 0.06).abs() < 1e-12, "{adjusted:?}");

        // Never below the raw value, never above one, and monotone in rank.
        let raw = [0.001, 0.02, 0.2, 0.5];
        let corrected = holm(&raw);
        for (raw, corrected) in raw.iter().zip(&corrected) {
            assert!(*corrected >= *raw - 1e-12);
            assert!(*corrected <= 1.0);
        }
    }

    #[test]
    fn the_bootstrap_is_seeded_and_contains_a_constant() {
        let values = [2.0; 8];
        let (low, high) = bootstrap_ci(&values, 2_000, 7);
        close(low, 2.0);
        close(high, 2.0);

        // Same seed, same interval.
        let a = bootstrap_ci(&[1.0, 2.0, 3.0, 4.0], 2_000, 11);
        let b = bootstrap_ci(&[1.0, 2.0, 3.0, 4.0], 2_000, 11);
        assert_eq!(a, b);

        // The interval is ordered and brackets the sample mean.
        let sample = [1.0, 2.0, 3.0, 4.0, 10.0];
        let (low, high) = bootstrap_ci(&sample, 4_000, 3);
        let mean = sample.iter().sum::<f64>() / sample.len() as f64;
        assert!(low <= mean && mean <= high, "{low}..{high} vs {mean}");
    }

    #[test]
    fn task_bootstrap_resamples_tasks_not_seeds() {
        // One value per task: the interval must cover the zero-mean case.
        let differences = [0.0; 32];
        let (low, high) = task_bootstrap(&differences, 1_000, 5);
        close(low, 0.0);
        close(high, 0.0);
    }
}
