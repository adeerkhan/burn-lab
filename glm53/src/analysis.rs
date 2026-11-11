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
