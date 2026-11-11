//! CLI for the Burn port of GLM-5.3-Flash-from-scratch.

use std::time::Instant;

use burn::{prelude::*, store::ModuleRecord, tensor::Device};
use glm53::{
    GLM53Flash, ModelConfig,
    evaluation::{GreedyConfig, SampledConfig, pass_at_k, pass_at_one},
    generate::{Decoding, GroupConfig, generate_raw},
    metrics::{PeakMemory, describe},
    rl::{RewardMode, RlConfig, TrainScope, run_rl},
    tasks,
    training::{TrainConfig, list_checkpoints, train},
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
    samples: usize,
    temperature: f64,
    max_new: usize,
    prompt: String,
    groups: usize,
    group_size: usize,
    tasks_per_family: usize,
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
            samples: 8,
            temperature: 0.0,
            max_new: 48,
            prompt: String::new(),
            groups: rl.groups,
            group_size: rl.group_size,
            tasks_per_family: rl.tasks_per_family,
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
            let model = GLM53Flash::new(&config, &device);
            print_model(&model);

            let train_config = TrainConfig {
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

fn load_model(config: &ModelConfig, device: &Device, directory: &str) -> GLM53Flash {
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
