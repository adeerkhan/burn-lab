//! CLI for the Burn port of GLM-5.3-Flash-from-scratch.

use std::time::Instant;

use burn::{prelude::*, store::ModuleRecord, tensor::Device};
use glm53::{
    GLM53Flash, ModelConfig, analysis,
    evaluation::{GreedyConfig, SampledConfig, pass_at_k, pass_at_one},
    experiments::{self, StudyConfig, write_report as write_study},
    generate::{Decoding, GroupConfig, generate_raw},
    metrics::{PeakMemory, describe},
    receipt,
    rl::{RewardMode, RlConfig, TrainScope, run_rl},
    tasks,
    training::{TrainConfig, list_checkpoints, train},
    vision_pilot::{PilotConfig, run_pilot, write_report as write_pilot_report},
    vision_train::{
        Architecture, VisionTrainConfig, train_vision_with, vision_config, write_report,
    },
};

const USAGE: &str = "\
glm53 — a Burn port of vukrosic/glm-5.3-flash-from-scratch

USAGE:
    glm53 <COMMAND> [OPTIONS]

COMMANDS:
    params          Print the parameter counts and the layer mix
    train           Pretrain from random initialization
    select          Pick the best pretrain checkpoint on dev pass@1
    evaluate        Greedy pass@1 over a frozen split
    evaluate-passk  Sampled pass@k over a frozen split
    rl              RLOO post-training with executable reward
    generate        Sample a completion for a prompt
    tasks           Print tasks from a split
    vision          Train and compare the two image paths on RGB digits
    vision-pilot    Attach the vision tower to the full 25.7M language model
    experiment      Run a matched miniature pretraining study
    rl-screen       Run the eight-arm RLOO variant screen from one checkpoint
    rl-confirm      Compare group sizes across seeds from one checkpoint
    analyze-rl      Analyze a screen or confirmation receipt
    analyze-results Summarize every receipt in a directory
    analyze         Apply paired statistics to a study receipt
    system-receipt  Capture the runtime and hardware that produced a result

OPTIONS:
    --backend <NAME>        wgpu | flex                        (default: wgpu)
    --steps <N>             Pretraining steps                  (default: 400)
    --batch-size <N>        Batch size                         (default: 16)
    --seq-len <N>           Sequence length                    (default: 128)
    --learning-rate <F>     Learning rate                      (default: 3e-4 train, 2e-5 rl)
    --corpus <N>            Training documents                 (default: 512)
    --log-every <N>         Log interval                       (default: 25)
    --checkpoints <LIST>    Steps to checkpoint, e.g. 100,200,400
    --out <PATH>            Output directory                   (default: artifacts/glm53)
    --ckpt <PATH>           Checkpoint directory to load
    --split <NAME>          dev | final | confirm | rl         (default: dev)
    --per-family <N>        Tasks per family                   (default: 4)
    --families <LIST>       Restrict to families, comma separated
    --json <PATH>           Write the full report as JSON
    --architecture <A>      vision: faithful | baseline | both (default: both)
    --smoke-test            Cap a vision run at 40 steps, batch 20, 100 examples
    --study <S>             experiment: diversity | curriculum  (default: diversity)
    --diversities <LIST>    Diversity arms                       (default: 8,32,88)
    --conditions <LIST>     Curriculum conditions                (default: all three)
    --seeds <LIST>          Paired seeds                         (default: 11,22,33)
    --study-json <PATH>     analyze: a study receipt to read
    --samples <N>           Samples per task for pass@k        (default: 8)
    --temperature <F>       Sampling temperature, 0 = greedy   (default: 0)
    --max-new <N>           Max new bytes                      (default: 48)
    --prompt <TEXT>         Prompt for generate
    --groups <N>            RL groups                          (default: 32)
    --group-size <N>        RL rollouts per group              (default: 4)
    --tasks-per-family <N>  RL tasks per family                (default: 4)
    --train-scope <S>       all | last-block | last-block-head (default: last-block-head)
    --reward-mode <M>       binary | case-fraction             (default: binary)
    --invalid-penalty <F>   Reward for invalid code            (default: -0.1)
    --exact-bonus <F>       Extra reward when all tests pass   (default: 0)
    -h, --help              Show this message
";

#[derive(Debug, Clone)]
struct Options {
    backend: String,
    steps: usize,
    batch_size: usize,
    seq_len: usize,
    learning_rate: Option<f64>,
    corpus: usize,
    log_every: usize,
    checkpoints: Vec<usize>,
    out: String,
    ckpt: String,
    split: String,
    per_family: usize,
    families: Vec<String>,
    json: String,
    architecture: String,
    smoke: bool,
    study: String,
    kind: String,
    diversities: String,
    conditions: String,
    study_json: String,
    seeds: String,
    samples: usize,
    temperature: f64,
    max_new: usize,
    prompt: String,
    groups: usize,
    group_size: usize,
    tasks_per_family: usize,
    vision_width: usize,
    vision_depth: usize,
    seed: u64,
    train_scope: String,
    reward_mode: String,
    invalid_penalty: f64,
    exact_bonus: f64,
}

impl Default for Options {
    fn default() -> Self {
        let train = TrainConfig::default();
        let rl = RlConfig::default();
        Self {
            backend: "wgpu".to_string(),
            steps: train.steps,
            batch_size: train.batch_size,
            seq_len: train.sequence_length,
            learning_rate: None,
            corpus: train.corpus_size,
            log_every: train.log_every,
            checkpoints: Vec::new(),
            out: "artifacts/glm53".to_string(),
            ckpt: String::new(),
            split: "dev".to_string(),
            per_family: 4,
            families: Vec::new(),
            json: String::new(),
            architecture: "both".to_string(),
            smoke: false,
            study: "diversity".to_string(),
            kind: "variants".to_string(),
            diversities: "8,32,88".to_string(),
            conditions: "curriculum-8-to-88,diverse-interleaved,diverse-blocked".to_string(),
            study_json: String::new(),
            seeds: "11,22,33".to_string(),
            samples: 8,
            temperature: 0.0,
            max_new: 48,
            prompt: String::new(),
            groups: rl.groups,
            group_size: rl.group_size,
            tasks_per_family: rl.tasks_per_family,
            vision_width: 24,
            vision_depth: 2,
            seed: 42,
            train_scope: rl.train_scope.label().to_string(),
            reward_mode: rl.reward_mode.label().to_string(),
            invalid_penalty: rl.invalid_penalty,
            exact_bonus: rl.exact_bonus,
        }
    }
}

impl Options {
    fn parse(args: &[String]) -> Self {
        let mut options = Options::default();
        let mut index = 0;

        while index < args.len() {
            let key = args[index].as_str();
            let value = |index: &mut usize| -> String {
                *index += 1;
                args.get(*index)
                    .cloned()
                    .unwrap_or_else(|| panic!("missing value for `{key}`"))
            };

            match key {
                "--backend" => options.backend = value(&mut index),
                "--steps" => options.steps = value(&mut index).parse().expect("invalid --steps"),
                "--vision-width" => {
                    options.vision_width =
                        value(&mut index).parse().expect("invalid --vision-width")
                }
                "--vision-depth" => {
                    options.vision_depth =
                        value(&mut index).parse().expect("invalid --vision-depth")
                }
                "--seed" => options.seed = value(&mut index).parse().expect("invalid --seed"),
                "--batch-size" => {
                    options.batch_size = value(&mut index).parse().expect("invalid --batch-size")
                }
                "--seq-len" => {
                    options.seq_len = value(&mut index).parse().expect("invalid --seq-len")
                }
                "--learning-rate" => {
                    options.learning_rate =
                        Some(value(&mut index).parse().expect("invalid --learning-rate"))
                }
                "--corpus" => options.corpus = value(&mut index).parse().expect("invalid --corpus"),
                "--log-every" => {
                    options.log_every = value(&mut index).parse().expect("invalid --log-every")
                }
                "--checkpoints" => {
                    options.checkpoints = value(&mut index)
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| part.trim().parse().expect("invalid --checkpoints"))
                        .collect()
                }
                "--out" => options.out = value(&mut index),
                "--ckpt" => options.ckpt = value(&mut index),
                "--split" => options.split = value(&mut index),
                "--per-family" => {
                    options.per_family = value(&mut index).parse().expect("invalid --per-family")
                }
                "--json" => options.json = value(&mut index),
                "--architecture" => options.architecture = value(&mut index),
                "--study" => options.study = value(&mut index),
                "--kind" => options.kind = value(&mut index),
                "--diversities" => options.diversities = value(&mut index),
                "--conditions" => options.conditions = value(&mut index),
                "--seeds" => options.seeds = value(&mut index),
                "--study-json" => options.study_json = value(&mut index),
                "--smoke-test" => options.smoke = true,
                "--families" => {
                    options.families = value(&mut index)
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| part.trim().to_string())
                        .collect()
                }
                "--samples" => {
                    options.samples = value(&mut index).parse().expect("invalid --samples")
                }
                "--temperature" => {
                    options.temperature = value(&mut index).parse().expect("invalid --temperature")
                }
                "--max-new" => {
                    options.max_new = value(&mut index).parse().expect("invalid --max-new")
                }
                "--prompt" => options.prompt = value(&mut index),
                "--groups" => options.groups = value(&mut index).parse().expect("invalid --groups"),
                "--group-size" => {
                    options.group_size = value(&mut index).parse().expect("invalid --group-size")
                }
                "--tasks-per-family" => {
                    options.tasks_per_family = value(&mut index)
                        .parse()
                        .expect("invalid --tasks-per-family")
                }
                "--train-scope" => options.train_scope = value(&mut index),
                "--reward-mode" => options.reward_mode = value(&mut index),
                "--invalid-penalty" => {
                    options.invalid_penalty = value(&mut index)
                        .parse()
                        .expect("invalid --invalid-penalty")
                }
                "--exact-bonus" => {
                    options.exact_bonus = value(&mut index).parse().expect("invalid --exact-bonus")
                }
                other => panic!("unknown option `{other}`\n\n{USAGE}"),
            }

            index += 1;
        }

        options
    }

    fn device(&self) -> Device {
        match self.backend.as_str() {
            "wgpu" => Device::wgpu(Default::default()),
            "flex" => Device::flex(),
            other => panic!("unknown backend `{other}`, expected wgpu or flex"),
        }
    }

    fn tasks(&self, split: &str) -> Vec<tasks::CodingTask> {
        let mut selected = tasks::frozen_tasks(split, self.per_family);
        if !self.families.is_empty() {
            selected.retain(|task| self.families.contains(&task.family));
            let covered: Vec<&String> = selected.iter().map(|task| &task.family).collect();
            for family in &self.families {
                assert!(
                    covered.contains(&family),
                    "unknown or missing requested family `{family}`"
                );
            }
        }
        selected
    }

    fn checkpoint_dir(&self) -> String {
        if self.ckpt.is_empty() {
            self.out.clone()
        } else {
            self.ckpt.clone()
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "params".to_string());
    let rest: Vec<String> = args.collect();

    if command == "-h" || command == "--help" || command == "help" {
        println!("{USAGE}");
        return;
    }

    let options = Options::parse(&rest);
    let config = ModelConfig::new();

    match command.as_str() {
        "params" => {
            let device = options.device();
            let model = GLM53Flash::new(&config, &device);
            print_model(&model);
        }
        "train" => {
            let started = Instant::now();
            let memory = PeakMemory::start();
            let device = options.device().autodiff();
            println!("Backend: {}  Device: {device:?}", options.backend);
            // Seed before construction so a run with the same configuration and
            // seed starts from the same weights.
            device.seed(options.seed);
            let model = GLM53Flash::new(&config, &device);
            print_model(&model);

            let train_config = TrainConfig {
                seed: options.seed,
                steps: options.steps,
                batch_size: options.batch_size,
                sequence_length: options.seq_len,
                learning_rate: options.learning_rate.unwrap_or(3e-4),
                corpus_size: options.corpus,
                log_every: options.log_every,
                checkpoints: options.checkpoints.clone(),
                artifact_dir: Some(options.out.clone()),
                ..TrainConfig::default()
            };

            println!(
                "\nTraining {} steps, batch {}, sequence {}, lr {}",
                train_config.steps,
                train_config.batch_size,
                train_config.sequence_length,
                train_config.learning_rate
            );

            let model = train(model, &train_config, &device, |metrics| {
                let experts = config.experts;
                let layers = metrics.usage.len() / experts;
                let last: Vec<String> = metrics.usage[(layers - 1) * experts..layers * experts]
                    .iter()
                    .map(|share| format!("{share:.2}"))
                    .collect();
                println!(
                    "step {:>4}  loss {:.4}  last-layer usage [{}]",
                    metrics.step,
                    metrics.loss,
                    last.join(" ")
                );
            });

            std::fs::create_dir_all(&options.out).ok();
            model
                .valid()
                .clone()
                .into_record()
                .save(format!("{}/model", options.out))
                .expect("save the final model");
            println!(
                "\nSaved to {}  ({:.1}s, {})",
                options.out,
                started.elapsed().as_secs_f64(),
                describe(memory.stop())
            );
        }
        "select" => {
            let device = options.device();
            let tasks = options.tasks("dev");
            let checkpoints = list_checkpoints(&options.out);
            assert!(
                !checkpoints.is_empty(),
                "no checkpoints under {}; run `train --checkpoints ...` first",
                options.out
            );

            println!(
                "Selecting a checkpoint on dev pass@1 ({} tasks)",
                tasks.len()
            );
            let mut best: Option<(f64, usize, String)> = None;
            for (step, path) in &checkpoints {
                let model = load_model(&config, &device, path);
                let report = pass_at_one(
                    &model,
                    &tasks,
                    &GreedyConfig {
                        max_new_tokens: options.max_new,
                        seed: 2026,
                    },
                    &device,
                );
                println!(
                    "  checkpoint {:>4}  pass@1 {:.4}  ({}/{})  {:.1}s",
                    step,
                    report.pass_at_1,
                    report.tasks_solved,
                    report.tasks_total,
                    report.elapsed_seconds
                );
                if best
                    .as_ref()
                    .is_none_or(|(score, _, _)| report.pass_at_1 > *score)
                {
                    best = Some((report.pass_at_1, *step, path.clone()));
                }
            }
            let (score, step, path) = best.expect("at least one checkpoint");
            println!("\nBest: checkpoint {step} with dev pass@1 {score:.4}\n  {path}");
        }
        "evaluate" => {
            let device = options.device();
            let model = load_model(&config, &device, &options.checkpoint_dir());
            let tasks = options.tasks(&options.split);

            println!(
                "Greedy pass@1 on split `{}` ({} tasks)",
                options.split,
                tasks.len()
            );
            let report = pass_at_one(
                &model,
                &tasks,
                &GreedyConfig {
                    max_new_tokens: options.max_new,
                    seed: 2026,
                },
                &device,
            );
            print_pass_at_one(&report);
            write_json(&options.json, &report);
        }
        "evaluate-passk" => {
            let device = options.device();
            let model = load_model(&config, &device, &options.checkpoint_dir());
            let tasks = options.tasks(&options.split);

            println!(
                "Sampled pass@{} on split `{}` ({} tasks, T={})",
                options.samples,
                options.split,
                tasks.len(),
                options.temperature.max(0.35)
            );
            let report = pass_at_k(
                &model,
                &tasks,
                &SampledConfig {
                    max_new_tokens: options.max_new,
                    temperature: if options.temperature > 0.0 {
                        options.temperature
                    } else {
                        0.35
                    },
                    samples: options.samples,
                    seed: 8_675_309,
                },
                &device,
            );
            print_pass_at_k(&report);
            write_json(&options.json, &report);
        }
        "rl" => {
            let started = Instant::now();
            let memory = PeakMemory::start();
            let device = options.device().autodiff();
            let initial = options.checkpoint_dir();
            let model = load_model(&config, &device, &initial);
            // Apply the freeze before reporting, so the count is the real one.
            let scope = TrainScope::parse(&options.train_scope);
            let model = glm53::rl::apply_scope(model, scope);
            let trainable = model.num_trainable_params();
            let config_params = model.num_params();

            let mut rl_config = RlConfig {
                groups: options.groups,
                group_size: options.group_size,
                max_new_tokens: options.max_new,
                temperature: if options.temperature > 0.0 {
                    options.temperature
                } else {
                    0.8
                },
                learning_rate: options.learning_rate.unwrap_or(2e-5),
                train_scope: scope,
                reward_mode: RewardMode::parse(&options.reward_mode),
                invalid_penalty: options.invalid_penalty,
                exact_bonus: options.exact_bonus,
                tasks_per_family: options.tasks_per_family,
                families: options.families.clone(),
                checkpoints: options.checkpoints.clone(),
                artifact_dir: options.out.clone(),
                ..RlConfig::default()
            };
            // A final checkpoint must exist so evaluation has something to load.
            if rl_config.checkpoints.is_empty() {
                rl_config.checkpoints = vec![rl_config.groups];
            }

            let mut rl_tasks = tasks::frozen_tasks("rl", rl_config.tasks_per_family);
            if !rl_config.families.is_empty() {
                rl_tasks.retain(|task| rl_config.families.contains(&task.family));
            }

            println!(
                "RLOO: {} groups x {} rollouts, scope {}, reward {}, lr {}, T={}",
                rl_config.groups,
                rl_config.group_size,
                rl_config.train_scope.label(),
                rl_config.reward_mode.label(),
                rl_config.learning_rate,
                rl_config.temperature
            );
            println!(
                "Seed {}  tasks {}  trainable parameters {} of {}  device {device:?}",
                rl_config.seed,
                rl_tasks.len(),
                trainable,
                config_params
            );

            std::fs::create_dir_all(&options.out).ok();
            let model = run_rl(model, &rl_tasks, &rl_config, &device, |metrics| {
                println!(
                    "group {:>3}  {:<16} rewards [{}]  loss {}  exact {}  {:.2}s",
                    metrics.group,
                    metrics.family,
                    metrics
                        .rewards
                        .iter()
                        .map(|reward| format!("{reward:.2}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    metrics
                        .policy_loss
                        .map(|loss| format!("{loss:.4}"))
                        .unwrap_or_else(|| "-".to_string()),
                    metrics.exact_rollouts,
                    metrics.seconds
                );
            });

            model
                .valid()
                .clone()
                .into_record()
                .save(format!("{}/model", options.out))
                .expect("save the RL model");
            println!(
                "\nSaved to {}  ({:.1}s, {})",
                options.out,
                started.elapsed().as_secs_f64(),
                describe(memory.stop())
            );
        }
        "generate" => {
            let device = options.device();
            let model = load_model(&config, &device, &options.checkpoint_dir());

            let prompt = if options.prompt.is_empty() {
                let task = &options.tasks(&options.split)[0];
                println!("No --prompt given, using a {} task.", options.split);
                task.prompt.clone()
            } else {
                options.prompt.clone()
            };

            let decoding = if options.temperature > 0.0 {
                Decoding::Sample {
                    temperature: options.temperature,
                }
            } else {
                Decoding::Greedy
            };

            println!("--- prompt ---\n{prompt}");
            let completion = generate_raw(
                &model,
                &prompt,
                &GroupConfig {
                    group_size: 1,
                    max_new_tokens: options.max_new,
                    decoding,
                    seed: 2026,
                    stop_on_parseable: false,
                },
                &device,
            );
            println!("--- completion ---\n{completion}");
        }
        "vision" => {
            let mut config = VisionTrainConfig {
                steps: options.steps,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate.unwrap_or(2e-3),
                eval_examples: options.per_family * 50,
                seed: options.seed,
                noise: 0.06,
            };
            if options.smoke {
                config.steps = config.steps.min(40);
                config.batch_size = config.batch_size.min(20);
                config.eval_examples = config.eval_examples.min(100);
            }

            let architectures = match options.architecture.as_str() {
                "faithful" => vec![Architecture::Faithful],
                "baseline" | "direct_patch_baseline" => vec![Architecture::DirectPatch],
                "both" => vec![Architecture::Faithful, Architecture::DirectPatch],
                other => panic!("unknown architecture `{other}`"),
            };

            let device = options.device().autodiff();
            println!(
                "Vision comparison: {} architecture(s), {} steps, batch {}, lr {}, {} held-out examples",
                architectures.len(),
                config.steps,
                config.batch_size,
                config.learning_rate,
                config.eval_examples
            );

            let vision = vision_config()
                .with_hidden_size(options.vision_width)
                .with_intermediate_size(options.vision_width * 2)
                .with_depth(options.vision_depth)
                .with_heads(4);
            let report = train_vision_with(
                &architectures,
                &config,
                &vision,
                &device,
                &mut |name, step, loss| {
                    println!("  {name:<22} step {step:>3}  loss {loss:.4}");
                },
            );

            println!(
                "\n{:<24} {:>8} {:>9} {:>9} {:>7}",
                "architecture", "before", "after", "params", "tokens"
            );
            for result in &report.results {
                println!(
                    "{:<24} {:>7.1}% {:>8.1}% {:>9} {:>7}",
                    result.architecture,
                    result.before.accuracy * 100.0,
                    result.after.accuracy * 100.0,
                    result.parameter_count,
                    result.visual_token_count
                );
            }
            println!("\ntotal {:.1}s", report.total_wall_time_seconds);

            if !options.json.is_empty() {
                write_report(&options.json, &report);
                println!("wrote {}", options.json);
            }
        }
        "rl-screen" => {
            let base = options.checkpoint_dir();
            let dev_tasks = options.tasks("dev");
            let rl_tasks = options.tasks("rl");

            // The reference's eight-arm screen: one design choice changed at a time.
            let preset: Vec<(&str, RewardMode, f64, f64, usize)> = vec![
                ("baseline", RewardMode::Binary, -0.1, 0.35, 8),
                ("partial-reward", RewardMode::CaseFraction, -0.1, 0.35, 8),
                ("no-penalty", RewardMode::Binary, 0.0, 0.35, 8),
                ("harsh-penalty", RewardMode::Binary, -1.0, 0.35, 8),
                ("low-temp", RewardMode::Binary, -0.1, 0.20, 8),
                ("high-temp", RewardMode::Binary, -0.1, 0.80, 8),
                ("group-4", RewardMode::Binary, -0.1, 0.35, 4),
                ("group-16", RewardMode::Binary, -0.1, 0.35, 16),
            ];

            let arms: Vec<(String, RlConfig)> = preset
                .into_iter()
                .map(|(name, mode, penalty, temperature, group_size)| {
                    (
                        name.to_string(),
                        RlConfig {
                            groups: options.groups,
                            group_size,
                            temperature,
                            reward_mode: mode,
                            invalid_penalty: penalty,
                            tasks_per_family: options.tasks_per_family,
                            learning_rate: options.learning_rate.unwrap_or(5e-5),
                            artifact_dir: format!("{}/{name}", options.out),
                            checkpoints: Vec::new(),
                            ..RlConfig::default()
                        },
                    )
                })
                .collect();

            let device = options.device().autodiff();
            let memory = PeakMemory::start();
            println!(
                "RL variant screen: {} arms x {} groups x {} rollouts from {base}",
                arms.len(),
                options.groups,
                8
            );

            let started = Instant::now();
            let results = glm53::rl::screen(
                &base,
                &arms,
                &rl_tasks,
                &dev_tasks,
                &device,
                |result| {
                    println!(
                        "  {:<15} signal {:>2}/{}  exact {:>3}/{}  T={:.2}  penalty {:+.1}  {}  dev {:.3}",
                        result.arm,
                        result.groups_with_signal,
                        result.groups,
                        result.exact_rollouts,
                        result.rollouts,
                        result.temperature,
                        result.invalid_penalty,
                        result.reward_mode,
                        result.pass_at_one.unwrap_or(0.0)
                    );
                },
            );

            println!(
                "\ntotal {:.1}s, {}",
                started.elapsed().as_secs_f64(),
                describe(memory.stop())
            );
            if !options.json.is_empty() {
                if let Some(parent) = std::path::Path::new(&options.json).parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::write(
                    &options.json,
                    serde_json::to_string_pretty(&results).expect("serialise") + "\n",
                )
                .expect("write the screen");
                println!("wrote {}", options.json);
            }
        }
        "analyze-results" => {
            let directory = if options.out.is_empty() {
                "artifacts".to_string()
            } else {
                options.out.clone()
            };
            let summary = analysis::analyze_results(&directory);
            println!(
                "summarised {} receipts from {directory}",
                summary["receipts"]
            );
            let json = if options.json.is_empty() {
                "artifacts/results-summary.json".to_string()
            } else {
                options.json.clone()
            };
            if let Some(parent) = std::path::Path::new(&json).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(
                &json,
                serde_json::to_string_pretty(&summary).expect("serialise") + "\n",
            )
            .expect("write the summary");
            println!("wrote {json}");
        }
        "rl-confirm" => {
            let base = options.checkpoint_dir();
            let dev_tasks = options.tasks("dev");
            let rl_tasks = options.tasks("rl");
            let group_sizes: Vec<usize> = options
                .diversities
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .map(|part| part.trim().parse().expect("invalid group sizes"))
                .collect();
            let seeds: Vec<u64> = options
                .seeds
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .map(|part| part.trim().parse().expect("invalid --seeds"))
                .take(3)
                .collect();

            let config = RlConfig {
                groups: options.groups,
                temperature: options.temperature.max(0.35),
                reward_mode: RewardMode::parse(&options.reward_mode),
                invalid_penalty: options.invalid_penalty,
                tasks_per_family: options.tasks_per_family,
                learning_rate: options.learning_rate.unwrap_or(5e-5),
                artifact_dir: format!("{}/confirm", options.out),
                ..RlConfig::default()
            };

            let device = options.device().autodiff();
            let memory = PeakMemory::start();
            println!(
                "Group-size confirmation: sizes {:?} x {} seeds from {base}, {} groups each",
                group_sizes,
                seeds.len(),
                config.groups
            );

            let runs = glm53::rl::group_confirmation(
                &base,
                &group_sizes,
                &seeds,
                &rl_tasks,
                &dev_tasks,
                &config,
                &device,
                |run| {
                    println!(
                        "  group {:>2} seed {:>3}  signal {:>2}/{}  exact {:>3}/{}  {:.1}s",
                        run.group_size,
                        run.seed,
                        run.groups_with_signal,
                        run.groups,
                        run.exact_rollouts,
                        run.rollouts,
                        run.seconds
                    );
                },
            );

            println!("\n{}", describe(memory.stop()));
            if !options.json.is_empty() {
                if let Some(parent) = std::path::Path::new(&options.json).parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::write(
                    &options.json,
                    serde_json::to_string_pretty(&runs).expect("serialise") + "\n",
                )
                .expect("write the confirmation");
                println!("wrote {}", options.json);
            }
        }
        "analyze-rl" => {
            assert!(
                !options.study_json.is_empty(),
                "--study-json is required by `analyze-rl`"
            );
            let text = std::fs::read_to_string(&options.study_json)
                .unwrap_or_else(|error| panic!("could not read {}: {error}", options.study_json));

            let json = if options.json.is_empty() {
                "artifacts/analysis-rl.json".to_string()
            } else {
                options.json.clone()
            };

            if options.kind == "confirmation" {
                let runs: Vec<glm53::analysis::ConfirmationRun> =
                    serde_json::from_str(&text).expect("parse the confirmation receipt");
                let sizes: Vec<usize> = {
                    let mut values: Vec<usize> = runs.iter().map(|run| run.group_size).collect();
                    values.sort_unstable();
                    values.dedup();
                    values
                };
                let smaller = *sizes.first().expect("at least one group size");
                let larger = *sizes.last().expect("at least one group size");
                let analysis = glm53::analysis::analyze_group_confirmation(&runs, smaller, larger);

                println!(
                    "group {larger} minus group {smaller}, {} seeds",
                    analysis.per_seed.len()
                );
                for row in &analysis.per_seed {
                    println!(
                        "  seed {:>3}  group {smaller} {:.3}  group {larger} {:.3}  diff {:+.3}  p {:.3}",
                        row.seed, row.smaller, row.larger, row.difference, row.mcnemar_p
                    );
                }
                println!(
                    "\nseed mean {:+.3}   task-level {:+.3} (95% CI {:+.3} to {:+.3})",
                    analysis.seed_mean_difference,
                    analysis.task_mean_difference,
                    analysis.task_ci_low,
                    analysis.task_ci_high
                );
                std::fs::write(
                    &json,
                    serde_json::to_string_pretty(&analysis).expect("serialise") + "\n",
                )
                .expect("write the analysis");
                println!("wrote {json}");
            } else {
                let arms: Vec<glm53::rl::ScreenArmResult> =
                    serde_json::from_str(&text).expect("parse the screen receipt");
                let analysis = glm53::analysis::analyze_variants(&arms);

                println!("reference arm: {}\n", analysis.reference_arm);
                println!(
                    "{:<15} {:>10} {:>10} {:>12} {:>10}",
                    "arm", "only arm", "only ref", "mean diff", "Holm p"
                );
                for (name, comparison) in &analysis.comparisons {
                    println!(
                        "{:<15} {:>10} {:>10} {:>+12.3} {:>10.3}",
                        name,
                        comparison.treatment_only,
                        comparison.reference_only,
                        comparison.mean_difference,
                        comparison.adjusted_p
                    );
                }
                std::fs::write(
                    &json,
                    serde_json::to_string_pretty(&analysis).expect("serialise") + "\n",
                )
                .expect("write the analysis");
                println!("\nwrote {json}");
            }
        }
        "experiment" => {
            let config = StudyConfig {
                steps: options.steps,
                batch_size: options.batch_size,
                learning_rate: options.learning_rate.unwrap_or(8e-4),
                sequence_length: options.seq_len,
                seeds: options
                    .seeds
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| part.trim().parse().expect("invalid --seeds"))
                    .collect(),
            };
            let device = options.device();

            let report = match options.study.as_str() {
                "diversity" => {
                    let arms: Vec<usize> = options
                        .diversities
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| part.trim().parse().expect("invalid --diversities"))
                        .collect();
                    println!(
                        "Data diversity: {:?} structures x {} seeds, {} steps x batch {}",
                        arms,
                        config.seeds.len(),
                        config.steps,
                        config.batch_size
                    );
                    let report = experiments::data_diversity(&arms, &config, &device);
                    for run in &report.runs {
                        println!(
                            "  {:<14} seed {:>3}  byte acc {:.3}  exact {:.3}  {:.1}s",
                            run.arm,
                            run.seed,
                            run.after.target_byte_accuracy,
                            run.after.target_exact_accuracy,
                            run.elapsed_seconds
                        );
                    }
                    report
                }
                "curriculum" => {
                    let conditions: Vec<String> = options
                        .conditions
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| part.trim().to_string())
                        .collect();
                    println!(
                        "Curriculum and ordering: {} conditions x {} seeds, {} steps x batch {}",
                        conditions.len(),
                        config.seeds.len(),
                        config.steps,
                        config.batch_size
                    );
                    let report = experiments::curriculum_order(&conditions, &config, &device);
                    for run in &report.runs {
                        println!(
                            "  {:<22} seed {:>3}  byte acc {:.3}  exact {:.3}  {:.1}s",
                            run.arm,
                            run.seed,
                            run.after.target_byte_accuracy,
                            run.after.target_exact_accuracy,
                            run.elapsed_seconds
                        );
                    }
                    report
                }
                other => panic!("unknown study `{other}`"),
            };

            println!("total {:.1}s", report.elapsed_seconds);
            if !options.json.is_empty() {
                write_study(&options.json, &report).expect("write the study receipt");
                println!("wrote {}", options.json);
            }
        }
        "analyze" => {
            assert!(
                !options.study_json.is_empty(),
                "--study-json is required by `analyze`"
            );
            let text = std::fs::read_to_string(&options.study_json)
                .unwrap_or_else(|error| panic!("could not read {}: {error}", options.study_json));
            let report: experiments::StudyReport =
                serde_json::from_str(&text).expect("parse the study receipt");
            let analysis = analysis::analyze(&report);

            println!("{}", analysis.research_question);
            println!(
                "\n{:<22} {:>8} {:>8} {:>8}",
                "arm", "byte acc", "exact", "sd"
            );
            for arm in &analysis.arms {
                println!(
                    "{:<22} {:>8.3} {:>8.3} {:>8.3}",
                    arm.arm, arm.mean_byte_accuracy, arm.mean_exact_accuracy, arm.std_byte_accuracy
                );
            }
            if !analysis.comparisons.is_empty() {
                println!(
                    "\n{:<22} {:>10} {:>18} {:>10} {:>10}",
                    "vs reference", "mean diff", "95% CI", "p", "Holm p"
                );
                for comparison in &analysis.comparisons {
                    println!(
                        "{:<22} {:>+10.4} {:>8.4}..{:<8.4} {:>10.4} {:>10.4}",
                        comparison.arm,
                        comparison.mean_difference,
                        comparison.ci_low,
                        comparison.ci_high,
                        comparison.permutation_p,
                        comparison.adjusted_p
                    );
                }
            }

            let json = if options.json.is_empty() {
                "artifacts/analysis.json".to_string()
            } else {
                options.json.clone()
            };
            if let Some(parent) = std::path::Path::new(&json).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(
                &json,
                serde_json::to_string_pretty(&analysis).expect("serialise") + "\n",
            )
            .expect("write the analysis");
            println!("\nwrote {json}");
        }
        "vision-pilot" => {
            let checkpoint = options.checkpoint_dir();
            let config = PilotConfig {
                steps: options.steps,
                batch_size: options.batch_size.min(10),
                learning_rate: options.learning_rate.unwrap_or(1e-3),
                eval_examples: 100,
                seed: 73,
                max_seconds: Some(540.0),
            };
            let device = options.device().autodiff();

            println!(
                "Vision pilot on {checkpoint}: {} steps, batch {}, lr {}",
                config.steps, config.batch_size, config.learning_rate
            );
            let report = run_pilot(&checkpoint, &config, &device, &mut |step, loss| {
                println!("  step {step:>3}  loss {loss:.4}");
            })
            .unwrap_or_else(|error| panic!("the pilot failed: {error}"));

            println!(
                "\nheld-out accuracy {:.1}% -> {:.1}%   loss {:.3} -> {:.3}",
                report.before.accuracy * 100.0,
                report.after.accuracy * 100.0,
                report.before.loss,
                report.after.loss
            );
            println!(
                "trainable {} of {} (vision {}), {} steps, {:.1}s, weights saved: {}",
                report.trainable_parameters,
                report.language_parameters,
                report.visual_parameters,
                report.steps_completed,
                report.wall_time_seconds,
                report.weights_saved
            );

            if !options.json.is_empty() {
                write_pilot_report(&options.json, &report).expect("write the pilot receipt");
                println!("wrote {}", options.json);
            }
        }
        "system-receipt" => {
            let receipt = receipt::system_receipt();
            let text = serde_json::to_string_pretty(&receipt).expect("serialise");
            match options.json.is_empty() {
                true => println!("{text}"),
                false => {
                    if let Some(parent) = std::path::Path::new(&options.json).parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    std::fs::write(&options.json, text + "\n").expect("write the system receipt");
                    println!("wrote {}", options.json);
                }
            }
        }
        "tasks" => {
            for task in options.tasks(&options.split) {
                println!("=== {} ===", task.task_id);
                println!("{}", task.reference_source());
            }
        }
        other => {
            eprintln!("unknown command `{other}`\n\n{USAGE}");
            std::process::exit(1);
        }
    }
}

/// Write a report when `--json` was given.
fn write_json<T: serde::Serialize>(path: &str, report: &T) {
    if path.is_empty() {
        return;
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let text = serde_json::to_string_pretty(report).expect("serialise the report");
    std::fs::write(path, text + "\n").expect("write the report");
    println!("wrote {path}");
}

fn load_model(config: &ModelConfig, device: &Device, directory: &str) -> GLM53Flash {
    // A checkpoint that carries a receipt must match it: a changed weight file is
    // a hard error. Checkpoints written before receipts existed load with a note.
    let path = std::path::Path::new(directory);
    match receipt::verify(path) {
        Ok(receipt) => {
            println!(
                "  verified {directory} ({} at step {})",
                receipt.stage, receipt.step
            );
        }
        Err(receipt::ReceiptError::Missing(_)) => {
            eprintln!("  warning: {directory} has no receipt; loading unverified");
        }
        Err(error) => panic!("refusing to load `{directory}`: {error}"),
    }

    let record = ModuleRecord::load(format!("{directory}/model")).unwrap_or_else(|error| {
        panic!("could not load a model from `{directory}`: {error}; run `train` first")
    });
    GLM53Flash::new(config, device).load_record(record)
}

fn print_model(model: &GLM53Flash) {
    let config = model.config();
    let sparse = (0..config.layers)
        .filter(|index| config.is_sparse(*index))
        .count();
    println!("GLM-5.3-Flash from scratch, ported to Burn");
    println!("  total parameters : {}", model.num_params());
    println!("  active per token : {}", model.num_active_params());
    println!(
        "  layers           : {} ({} linear, {} sparse)",
        config.layers,
        config.layers - sparse,
        sparse
    );
    println!(
        "  experts          : {} top-{}, plus 1 shared",
        config.experts, config.top_k
    );
    println!("  streams          : {}", config.streams);
    println!("  vocab            : {}", config.vocab_size);
}

fn print_pass_at_one(report: &glm53::evaluation::PassAtOne) {
    println!(
        "\npass@1 {:.4}  ({}/{})   test accuracy {:.4}   valid {}/{}   cap hits {}",
        report.pass_at_1,
        report.tasks_solved,
        report.tasks_total,
        report.test_accuracy,
        report.valid_programs,
        report.tasks_total,
        report.token_cap_hits
    );
    println!(
        "{:<14} {:>6} {:>7} {:>7}",
        "family", "solved", "tasks", "exact"
    );
    for family in &report.by_family {
        println!(
            "{:<14} {:>6} {:>7} {:>7}",
            family.family, family.solved, family.tasks, family.exact_rollouts
        );
    }
    println!("{:.2}s", report.elapsed_seconds);
}

fn print_pass_at_k(report: &glm53::evaluation::PassAtK) {
    println!(
        "\npass@{} {:.4}  ({}/{})   exact rollouts {:.4}   test accuracy {:.4}   {:.2}s",
        report.samples_per_task,
        report.pass_at_k,
        report.tasks_solved_at_k,
        report.tasks_total,
        report.exact_rollout_rate,
        report.test_accuracy,
        report.elapsed_seconds
    );
    println!(
        "{:<14} {:>6} {:>7} {:>8} {:>7}",
        "family", "solved", "tasks", "rollouts", "valid"
    );
    for family in &report.by_family {
        println!(
            "{:<14} {:>6} {:>7} {:>8} {:>7}",
            family.family, family.solved, family.tasks, family.rollouts, family.valid_rollouts
        );
    }
}
