//! Compare greedy generations before and after RL on the same prompts.
//!
//! Prints, for each task, both completions and both verdicts, and marks the
//! cases where the reward actually changed the outcome. Used to pick the example
//! in the README figure.
//!
//! ```sh
//! cargo run -p glm53 --release --example before_after -- \
//!     artifacts/glm53-run/checkpoint-0200 artifacts/glm53-rl3 confirm 8
//! ```

use burn::{prelude::*, tensor::Device};
use glm53::{
    GLM53Flash, ModelConfig,
    evaluation::verify,
    generate::{Decoding, GroupConfig, generate_group},
    tasks,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let before_path = args
        .next()
        .unwrap_or_else(|| "artifacts/glm53-run/checkpoint-0200".to_string());
    let after_path = args
        .next()
        .unwrap_or_else(|| "artifacts/glm53-rl3".to_string());
    let split = args.next().unwrap_or_else(|| "confirm".to_string());
    let per_family: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);

    let device = Device::flex();
    let config = ModelConfig::new();

    let load = |path: &str| {
        let record = burn::store::ModuleRecord::load(format!("{path}/model"))
            .unwrap_or_else(|error| panic!("could not load {path}: {error}"));
        GLM53Flash::new(&config, &device)
            .load_record(record)
            .valid()
    };
    let before = load(&before_path);
    let after = load(&after_path);

    let mut gained = 0usize;
    let mut lost = 0usize;
    let mut same_pass = 0usize;
    let mut same_fail = 0usize;

    println!(
        "{:<24} {:<12} {:>5} {:>5}  changed",
        "task", "family", "before", "after"
    );

    for task in tasks::frozen_tasks(&split, per_family) {
        let run = |model: &GLM53Flash| {
            let generations = generate_group(
                model,
                &task,
                &GroupConfig {
                    group_size: 1,
                    max_new_tokens: 48,
                    decoding: Decoding::Greedy,
                    seed: 2026,
                    stop_on_parseable: true,
                },
                &device,
            );
            let completion = generations[0].completion.clone();
            let verdict = verify(&task, &completion);
            (completion, verdict)
        };

        let (before_completion, before_verdict) = run(&before);
        let (after_completion, after_verdict) = run(&after);

        match (before_verdict.passed, after_verdict.passed) {
            (false, true) => gained += 1,
            (true, false) => lost += 1,
            (true, true) => same_pass += 1,
            (false, false) => same_fail += 1,
        }

        let changed = if before_completion != after_completion {
            "output"
        } else {
            "-"
        };
        println!(
            "{:<24} {:<12} {:>5} {:>5}  {}",
            task.task_id,
            task.family,
            if before_verdict.passed {
                "PASS"
            } else {
                "FAIL"
            },
            if after_verdict.passed { "PASS" } else { "FAIL" },
            changed
        );

        if before_completion != after_completion {
            println!("    before: {}", before_completion.replace('\n', "\\n"));
            println!("    after : {}", after_completion.replace('\n', "\\n"));
        }
    }

    println!("\ngained {gained}  lost {lost}  both pass {same_pass}  both fail {same_fail}");
}
