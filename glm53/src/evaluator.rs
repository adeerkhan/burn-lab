//! A fail-closed executable verifier for the synthetic coding tasks.
//!
//! Mirrors `glm53_flash/evaluator.py`. The reference validates the candidate
//! with a Python AST allowlist and then *executes* it against unit tests, so
//! the port shells out to `python3` with [`HARNESS`] rather than reimplementing
//! a Python parser and interpreter.
//!
//! Requests are batched: one interpreter start serves many candidates, which
//! matters because evaluation sits inside the RL loop.

use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

use serde::{Deserialize, Serialize};

/// The harness, compiled into the binary.
const HARNESS: &str = include_str!("../python/evaluator_harness.py");

/// One candidate to verify.
#[derive(Debug, Clone, Serialize)]
pub struct Request {
    /// Operation family, used to look up the test cases.
    pub family: String,
    /// The function name the candidate must define.
    pub entry_point: String,
    /// The task prompt.
    pub prompt: String,
    /// The model's completion, appended to the prompt.
    pub completion: String,
    /// Whether to execute the tests. Generation only needs `parseable`.
    pub run_tests: bool,
}

/// The verifier's verdict.
#[derive(Debug, Clone, Deserialize)]
pub struct Evaluation {
    /// `passed`, `failed`, `invalid` or `skipped`.
    pub status: String,
    /// True only when every test case passed.
    pub passed: bool,
    pub tests_passed: usize,
    pub tests_total: usize,
    pub pass_fraction: f64,
    /// Why the candidate was rejected, empty on success.
    pub message: String,
    /// Whether `prompt + completion` parses and ends with a newline.
    pub parseable: bool,
}

#[derive(Deserialize)]
struct Response {
    results: Vec<Evaluation>,
}

impl Evaluation {
    /// A verdict for a candidate that never reached the verifier.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: "invalid".to_string(),
            passed: false,
            tests_passed: 0,
            tests_total: 0,
            pass_fraction: 0.0,
            message: reason.into(),
            parseable: false,
        }
    }
}

/// Where the harness is written, once per process.
///
/// A fixed, shared path is not safe: every call used to truncate and rewrite
/// `harness.py`, so a second thread (the test suite runs in parallel) could hand
/// `python3` a half-written file and fail with a syntax error. The path is now
/// per-process and written once through `OnceLock`, and it is published with a
/// rename so a reader never observes a partial file.
fn harness_path() -> PathBuf {
    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let directory =
            std::env::temp_dir().join(format!("glm53-evaluator-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create the evaluator scratch directory");
        let path = directory.join("harness.py");
        let staged = directory.join("harness.py.staged");
        fs::write(&staged, HARNESS).expect("write the evaluator harness");
        fs::rename(&staged, &path).expect("publish the evaluator harness");
        path
    })
    .clone()
}

/// Verify a batch of candidates.
///
/// # Panics
///
/// Panics if `python3` is missing or the harness fails, because every caller
/// needs a verdict and silently returning "invalid" would poison the reward.
pub fn evaluate_batch(requests: &[Request]) -> Vec<Evaluation> {
    if requests.is_empty() {
        return Vec::new();
    }

    let harness = harness_path();
    let payload = serde_json::json!({ "requests": requests });

    let mut child = Command::new("python3")
        .arg(&harness)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| {
            panic!("the evaluator needs `python3` on PATH, but it could not be started: {error}")
        });

    child
        .stdin
        .as_mut()
        .expect("stdin is piped")
        .write_all(payload.to_string().as_bytes())
        .expect("write the evaluator request");

    let output = child.wait_with_output().expect("run the evaluator harness");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("the evaluator harness failed: {stderr}");
    }

    let response: Response =
        serde_json::from_slice(&output.stdout).expect("parse the evaluator response");

    assert_eq!(
        response.results.len(),
        requests.len(),
        "the evaluator returned {} verdicts for {} requests",
        response.results.len(),
        requests.len()
    );

    response.results
}

/// Verify one candidate.
pub fn evaluate(request: Request) -> Evaluation {
    evaluate_batch(std::slice::from_ref(&request))
        .pop()
        .expect("one verdict")
}

/// Whether `prompt + completion` would stop generation.
pub fn is_parseable(family: &str, entry_point: &str, prompt: &str, completion: &str) -> bool {
    evaluate(Request {
        family: family.to_string(),
        entry_point: entry_point.to_string(),
        prompt: prompt.to_string(),
        completion: completion.to_string(),
        run_tests: false,
    })
    .parseable
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt() -> &'static str {
        "# Complete this Python function.\n# Return x plus one.\ndef increment_abc(x):"
    }

    fn check(completion: &str) -> Evaluation {
        evaluate(Request {
            family: "increment".to_string(),
            entry_point: "increment_abc".to_string(),
            prompt: prompt().to_string(),
            completion: completion.to_string(),
            run_tests: true,
        })
    }

    /// The harness used to be rewritten at a fixed shared path on every call, so
    /// parallel callers could hand `python3` a half-written file. Eight threads
    /// now evaluate at once and every verdict must still be a pass.
    #[test]
    fn concurrent_evaluations_do_not_share_a_half_written_harness() {
        let handles: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(|| check("\n    return x + 1\n")))
            .collect();

        for handle in handles {
            let evaluation = handle.join().expect("the evaluator thread panicked");
            assert_eq!(evaluation.status, "passed", "{}", evaluation.message);
            assert_eq!(evaluation.tests_passed, 3);
        }
    }

    #[test]
    fn a_correct_completion_passes_every_case() {
        let evaluation = check("\n    return x + 1\n");
        assert_eq!(evaluation.status, "passed");
        assert!(evaluation.passed);
        assert_eq!(evaluation.tests_passed, 3);
        assert_eq!(evaluation.tests_total, 3);
        assert!((evaluation.pass_fraction - 1.0).abs() < 1e-9);
        assert!(evaluation.parseable);
    }

    #[test]
    fn a_wrong_but_valid_completion_fails() {
        // `x - 1` misses all three cases: -4 vs -2, -1 vs 1, 6 vs 8.
        let evaluation = check("\n    return x - 1\n");
        assert_eq!(evaluation.status, "failed");
        assert!(!evaluation.passed);
        assert_eq!(evaluation.tests_passed, 0);
        assert_eq!(evaluation.pass_fraction, 0.0);
    }

    #[test]
    fn an_import_is_rejected() {
        let evaluation = check("\n    import os\n    return x + 1\n");
        assert_eq!(evaluation.status, "invalid");
        assert!(evaluation.message.contains("disallowed syntax"));
    }

    #[test]
    fn attribute_access_is_rejected() {
        let evaluation = check("\n    return x.__class__\n");
        assert_eq!(evaluation.status, "invalid");
        assert!(evaluation.message.contains("dunder") || evaluation.message.contains("attribute"));
    }

    #[test]
    fn a_dunder_name_is_rejected() {
        let evaluation = check("\n    return __import__\n");
        assert_eq!(evaluation.status, "invalid");
        assert!(evaluation.message.contains("dunder"));
    }

    #[test]
    fn recursion_is_rejected() {
        // The reference checks the SAFE_CALLS allowlist before the recursion
        // branch, so a self-call reports "unsafe call". The recursion branch is
        // unreachable for any entry point outside SAFE_CALLS; the port keeps
        // the same order so the messages match.
        let evaluation = check("\n    return increment_abc(x)\n");
        assert_eq!(evaluation.status, "invalid");
        assert!(evaluation.message.contains("unsafe call"));
    }

    #[test]
    fn a_loop_is_rejected() {
        let evaluation = check("\n    while x:\n        x = x - 1\n    return x + 1\n");
        assert_eq!(evaluation.status, "invalid");
    }

    #[test]
    fn a_wrong_function_name_is_rejected() {
        let evaluation = check("\n    return x + 1\n\n\ndef other(x):\n    return x\n");
        assert_eq!(evaluation.status, "invalid");
        assert!(evaluation.message.contains("required function"));
    }

    #[test]
    fn oversized_source_is_rejected() {
        let padding = "# ".to_string() + &"x".repeat(2100) + "\n";
        let evaluation = check(&format!("\n    return x + 1\n{padding}"));
        assert_eq!(evaluation.status, "invalid");
        assert!(evaluation.message.contains("too large"));
    }

    #[test]
    fn strict_type_equality_is_enforced() {
        // `True == 1` holds in Python, but `type(True) is not type(1)`.
        let evaluation = evaluate(Request {
            family: "increment".to_string(),
            entry_point: "increment_abc".to_string(),
            prompt: prompt().to_string(),
            completion: "\n    return True\n".to_string(),
            run_tests: true,
        });
        assert_eq!(evaluation.status, "failed");
        assert_eq!(evaluation.tests_passed, 0);
    }

    #[test]
    fn parseable_needs_a_trailing_newline() {
        assert!(is_parseable(
            "increment",
            "increment_abc",
            prompt(),
            "\n    return x + 1\n"
        ));
        // Valid Python but no trailing newline.
        assert!(!is_parseable(
            "increment",
            "increment_abc",
            prompt(),
            "\n    return x + 1"
        ));
        // Incomplete statement.
        assert!(!is_parseable(
            "increment",
            "increment_abc",
            prompt(),
            "\n    return x +\n"
        ));
    }

    #[test]
    fn batching_returns_one_verdict_per_request() {
        let requests: Vec<Request> = ["increment", "double", "square"]
            .iter()
            .map(|family| Request {
                family: family.to_string(),
                entry_point: "f".to_string(),
                prompt: "def f(x):".to_string(),
                completion: "\n    return x\n".to_string(),
                run_tests: true,
            })
            .collect();
        let verdicts = evaluate_batch(&requests);
        assert_eq!(verdicts.len(), 3);
        // `return x` is wrong for increment/double/square but still valid Python.
        assert!(verdicts.iter().all(|verdict| verdict.status == "failed"));
        assert!(verdicts.iter().all(|verdict| verdict.tests_total == 3));
    }
}
