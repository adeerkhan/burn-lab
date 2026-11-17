//! Train and sample a DDPM/DDIM or a rectified flow on synthetic shapes or
//! packed photographs.

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::store::ModuleRecord;

use diffusion::classify_data::LabeledImages;
use diffusion::diffusion::{BetaSchedule, GaussianDiffusion, Objective};
use diffusion::ema::Ema;
use diffusion::flow::{
    DeltaSchedule, FlowConfig, FlowLoss, FlowSolver, NoiseSchedule, Predict, RectifiedFlow,
};
use diffusion::unet::{Unet, UnetConfig};
use diffusion::vit::{AttentionKind, Vit, VitConfig};
use diffusion::vit_classifier::{ClassifierConfig, VitClassifier};
use diffusion::world_model::{TOKENS, WorldConfig, WorldModel};
use diffusion::world_vq::{VqConfig, VqVae, edge_loss};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CheckpointConfig {
    dim: usize,
    image_size: usize,
    channels: usize,
    timesteps: usize,
    sampling_timesteps: usize,
    dim_mults: Vec<usize>,
    #[serde(default)]
    data: String,
    /// Train the velocity field of a rectified flow instead of a DDPM.
    #[serde(default)]
    flow: bool,
    /// `cnn` (the reference UNet) or `vit`.
    #[serde(default = "default_backbone")]
    backbone: String,
    #[serde(default = "default_patch")]
    patch_size: usize,
    #[serde(default = "default_depth")]
    depth: usize,
    #[serde(default = "default_heads")]
    heads: usize,
    /// Per-block attention for the ViT backbone: `full` or `linear`.
    #[serde(default)]
    attention: Vec<String>,
    /// Flow options, recorded so sampling replays the same objective.
    #[serde(default = "default_predict")]
    predict: String,
    #[serde(default = "default_flow_loss")]
    loss: String,
    #[serde(default = "default_schedule")]
    schedule: String,
    #[serde(default)]
    immiscible: bool,
    #[serde(default)]
    consistency: bool,
    #[serde(default = "default_delta")]
    consistency_delta: f32,
    #[serde(default = "default_delta_schedule")]
    delta_schedule: String,
}

fn default_delta() -> f32 {
    1e-3
}
fn default_delta_schedule() -> String {
    "constant".to_string()
}

fn default_predict() -> String {
    "flow".to_string()
}
fn default_flow_loss() -> String {
    "mse".to_string()
}
fn default_schedule() -> String {
    "identity".to_string()
}

/// The flow options stored in a checkpoint, as a typed config.
fn flow_config(config: &CheckpointConfig) -> FlowConfig {
    FlowConfig {
        predict: match config.predict.as_str() {
            "noise" => Predict::Noise,
            "clean" => Predict::Clean,
            _ => Predict::Flow,
        },
        loss: match config.loss.as_str() {
            "pseudo-huber" => FlowLoss::PseudoHuber,
            _ => FlowLoss::Mse,
        },
        schedule: match config.schedule.as_str() {
            "cosmap" => NoiseSchedule::Cosmap,
            _ => NoiseSchedule::Identity,
        },
        immiscible: config.immiscible,
        consistency: config.consistency,
        consistency_delta_time: config.consistency_delta,
        delta_schedule: match config.delta_schedule.as_str() {
            "linear" => DeltaSchedule::Linear,
            "cosine" => DeltaSchedule::Cosine,
            _ => DeltaSchedule::Constant,
        },
        ..FlowConfig::default()
    }
}

fn default_backbone() -> String {
    "cnn".to_string()
}
fn default_patch() -> usize {
    8
}
fn default_depth() -> usize {
    8
}
fn default_heads() -> usize {
    8
}

#[derive(Clone)]
struct Options {
    backend: String,
    steps: usize,
    batch: usize,
    image_size: usize,
    dim: usize,
    timesteps: usize,
    sampling_timesteps: usize,
    out: String,
    checkpoint: String,
    data: String,
    dump: String,
    eval_data: String,
    raw_out: String,
    ema_decay: f64,
    checkpoint_every: usize,
    resume: bool,
    seed: u64,
    samples: usize,
    flow: bool,
    vit: bool,
    patch_size: usize,
    depth: usize,
    heads: usize,
    attention: Vec<String>,
    predict: String,
    loss: String,
    schedule: String,
    immiscible: bool,
    consistency: bool,
    pairs: String,
    count: usize,
    solver: String,
    consistency_delta: f32,
    delta_schedule: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backend: "flex".to_string(),
            steps: 400,
            batch: 32,
            image_size: 32,
            dim: 32,
            timesteps: 1000,
            sampling_timesteps: 0, // 0 = use the train default, or the checkpoint's at sample time
            out: "artifacts/ddpm".to_string(),
            checkpoint: String::new(),
            data: String::new(),
            dump: String::new(),
            eval_data: String::new(),
            raw_out: String::new(),
            ema_decay: 0.999,
            checkpoint_every: 1000,
            resume: false,
            seed: 11,
            samples: 16,
            flow: false,
            vit: false,
            patch_size: 8,
            depth: 8,
            heads: 8,
            attention: Vec::new(),
            predict: "flow".to_string(),
            loss: "mse".to_string(),
            schedule: "identity".to_string(),
            immiscible: false,
            consistency: false,
            pairs: String::new(),
            count: 2048,
            solver: "midpoint".to_string(),
            consistency_delta: 1e-3,
            delta_schedule: "constant".to_string(),
        }
    }
}

impl Options {
    fn parse() -> (String, Self) {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let command = args.first().cloned().unwrap_or_else(|| "smoke".to_string());
        let mut options = Options::default();
        let mut index = 1;
        while index < args.len() {
            let key = args[index].clone();
            let mut value = || {
                index += 1;
                args.get(index)
                    .cloned()
                    .unwrap_or_else(|| panic!("missing value for {key}"))
            };
            match key.as_str() {
                "--backend" => options.backend = value(),
                "--steps" => options.steps = value().parse().expect("steps"),
                "--batch" => options.batch = value().parse().expect("batch"),
                "--image-size" => options.image_size = value().parse().expect("image size"),
                "--dim" => options.dim = value().parse().expect("dim"),
                "--timesteps" => options.timesteps = value().parse().expect("timesteps"),
                "--sampling-steps" => {
                    options.sampling_timesteps = value().parse().expect("sampling steps")
                }
                "--out" => options.out = value(),
                "--ckpt" => options.checkpoint = value(),
                "--data" => options.data = value(),
                "--dump" => options.dump = value(),
                "--eval-data" => options.eval_data = value(),
                "--raw-out" => options.raw_out = value(),
                "--ema-decay" => options.ema_decay = value().parse().expect("ema decay"),
                "--checkpoint-every" => {
                    options.checkpoint_every = value().parse().expect("checkpoint interval")
                }
                "--seed" => options.seed = value().parse().expect("seed"),
                "--resume" => options.resume = true,
                "--samples" => options.samples = value().parse().expect("samples"),
                "--flow" => options.flow = true,
                "--vit" => options.vit = true,
                "--patch-size" => options.patch_size = value().parse().expect("patch size"),
                "--depth" => options.depth = value().parse().expect("depth"),
                "--heads" => options.heads = value().parse().expect("heads"),
                "--predict" => options.predict = value(),
                "--loss" => options.loss = value(),
                "--schedule" => options.schedule = value(),
                "--immiscible" => options.immiscible = true,
                "--consistency" => options.consistency = true,
                "--pairs" => options.pairs = value(),
                "--count" => options.count = value().parse().expect("count"),
                "--solver" => options.solver = value(),
                "--consistency-delta" => {
                    options.consistency_delta = value().parse().expect("consistency delta")
                }
                "--delta-schedule" => options.delta_schedule = value(),
                "--attention" => {
                    options.attention = value()
                        .split(',')
                        .map(|kind| kind.trim().to_string())
                        .collect()
                }
                other => panic!("unknown flag `{other}`"),
            }
            index += 1;
        }
        (command, options)
    }

    fn device(&self) -> Device {
        match self.backend.as_str() {
            "wgpu" => Device::wgpu(Default::default()),
            "flex" => Device::flex(),
            other => panic!("unknown backend `{other}`"),
        }
    }
}

fn checkpoint_config(options: &Options) -> CheckpointConfig {
    CheckpointConfig {
        dim: options.dim,
        image_size: options.image_size,
        channels: 3,
        timesteps: options.timesteps,
        sampling_timesteps: options.sampling_timesteps,
        dim_mults: vec![1, 2, 4, 8],
        data: options.data.clone(),
        flow: options.flow,
        backbone: if options.vit {
            "vit".to_string()
        } else {
            "cnn".to_string()
        },
        patch_size: options.patch_size,
        depth: options.depth,
        heads: options.heads,
        attention: options.attention.clone(),
        predict: options.predict.clone(),
        loss: options.loss.clone(),
        schedule: options.schedule.clone(),
        immiscible: options.immiscible,
        consistency: options.consistency,
        consistency_delta: options.consistency_delta,
        delta_schedule: options.delta_schedule.clone(),
    }
}

fn unet_config(config: &CheckpointConfig) -> UnetConfig {
    UnetConfig {
        dim: config.dim,
        dim_mults: config.dim_mults.clone(),
        channels: config.channels,
        ..UnetConfig::default()
    }
}

/// Exit without unwinding the backend. All artefacts are already on disk.
fn leave_after_writing() -> ! {
    use std::io::Write;
    std::io::stdout().flush().ok();
    std::process::exit(0);
}

fn save_samples(samples: Tensor<4>, path: &str) {
    let [batch, _, height, width] = samples.dims();
    let flat = samples
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    let columns = (batch as f64).sqrt().ceil() as usize;
    let rows = batch.div_ceil(columns);
    let mut image = image::RgbImage::new((columns * width) as u32, (rows * height) as u32);

    let plane = height * width;
    for index in 0..batch {
        let offset = index * 3 * plane;
        let cell_x = (index % columns) * width;
        let cell_y = (index / columns) * height;
        for y in 0..height {
            for x in 0..width {
                // The tensor is `[channels, height, width]`: read one value
                // from each plane, do not assume the buffer is interleaved.
                let at = offset + y * width + x;
                let to_byte = |value: f32| ((value.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
                image.put_pixel(
                    (cell_x + x) as u32,
                    (cell_y + y) as u32,
                    image::Rgb([
                        to_byte(flat[at]),
                        to_byte(flat[at + plane]),
                        to_byte(flat[at + 2 * plane]),
                    ]),
                );
            }
        }
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image.save(path).expect("save the sample grid");
    println!("wrote {path}");
}

fn main() {
    let (command, options) = Options::parse();
    match command.as_str() {
        "train" => train(&options),
        "sample" => sample(&options),
        "smoke" => smoke(&options),
        "search-attention" => search_attention(&options),
        "reflow" => train_reflow(&options),
        "migrate" => migrate_record(&options),
        "classify" => classify(&options),
        "world-vq" => train_world_vq(&options),
        "world-train" => train_world_model(&options),
        "world-encode" => encode_world_tokens(&options),
        "world-dream" => dream_rollout(&options),
        "make-pairs" => make_pairs(&options),
        other => panic!("unknown command `{other}`; see README for the command list"),
    }
}

/// The backbone named by the checkpoint: the reference UNet, or the ViT port.
fn vit_config(config: &CheckpointConfig) -> VitConfig {
    VitConfig {
        image_size: config.image_size,
        patch_size: config.patch_size,
        channels: config.channels,
        dim: config.dim,
        depth: config.depth,
        heads: config.heads,
        mlp_ratio: 4,
        attention: if config.attention.is_empty() {
            vec![AttentionKind::Full; config.depth]
        } else {
            config
                .attention
                .iter()
                .map(|kind| match kind.as_str() {
                    "linear" => AttentionKind::Linear,
                    _ => AttentionKind::Full,
                })
                .collect()
        },
    }
}

fn build_model(config: &CheckpointConfig, device: &Device) -> Unet {
    if config.backbone == "vit" {
        Unet::Vit(Vit::new(&vit_config(config), device))
    } else {
        Unet::new(&unet_config(config), device)
    }
}

/// Train a small ViT classifier on a streaming u8 dataset, then report held-out
/// accuracy. No full dataset is ever held in RAM.
fn classify(options: &Options) {
    let device = options.device().autodiff();
    let options = &options;
    let prefix = if options.data.is_empty() {
        "/home/adeerkhan/tmp/flowers/cls".to_string()
    } else {
        options.data.clone()
    };
    let held_prefix = if options.eval_data.is_empty() {
        format!("{prefix}-held")
    } else {
        options.eval_data.clone()
    };
    let train_prefix = if prefix.ends_with("-train") {
        prefix.clone()
    } else {
        format!("{prefix}-train")
    };
    let classes = std::fs::read_to_string(format!("{prefix}-classes.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| value["classes"].as_array().map(|a| a.len()))
        .unwrap_or(5);
    let config = ClassifierConfig {
        image_size: options.image_size,
        classes,
        dim: options.dim.min(128),
        depth: options.depth.min(6),
        heads: options.heads.min(8),
        ..ClassifierConfig::default()
    };

    let mut train = LabeledImages::load(
        &format!("{train_prefix}-images.bin"),
        &format!("{train_prefix}-labels.bin"),
        config.image_size,
    );
    let mut held = LabeledImages::load(
        &format!("{held_prefix}-images.bin"),
        &format!("{held_prefix}-labels.bin"),
        config.image_size,
    );
    println!(
        "classifier dim={} depth={} heads={} classes={} train={} held={}",
        config.dim,
        config.depth,
        config.heads,
        classes,
        train.len(),
        held.len()
    );

    let mut model = VitClassifier::new(&config, &device);
    let mut optimizer = AdamWConfig::new().init();
    let started = std::time::Instant::now();
    let steps = options.steps.max(1);

    for step in 1..=steps {
        let seed = options.seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (images, labels) = train.batch(options.batch, seed, &device);
        let logits = model.forward(images);
        let loss = cross_entropy(logits, &labels, &device);
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        // Warm up then cosine decay; a constant 1e-3 oscillates around this loss.
        let progress = step as f64 / steps as f64;
        let lr = 1e-3 * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()));
        model = optimizer.step(lr, model, grads);

        if step == 1 || step % 100 == 0 {
            println!("step {step:>5} loss {value:.4}");
        }
    }

    // Exact held-out accuracy: walk every image once, in order, in batches.
    // Nothing is sampled with replacement and the full split is never in RAM.
    let total = held.len();
    let mut correct = 0usize;
    let mut done = 0usize;
    while done < total {
        let count = options.batch.min(total - done);
        let (images, labels) = held.sequential_batch(done, count, &device);
        let predicted = model.forward(images).argmax(1).reshape([count, 1]);
        let actual = labels.reshape([count, 1]);
        let hits = predicted
            .equal(actual)
            .to_data()
            .convert::<bool>()
            .try_to_vec::<bool>()
            .unwrap();
        correct += hits.iter().filter(|hit| **hit).count();
        done += count;
    }
    let accuracy = correct as f32 / total as f32;
    println!(
        "held-out accuracy {:.3} ({} / {}), trained in {:.1}s",
        accuracy,
        correct,
        total,
        started.elapsed().as_secs_f64()
    );

    std::fs::create_dir_all(&options.out).ok();
    model
        .clone()
        .valid()
        .into_record()
        .save(format!("{}/model", options.out))
        .expect("save the classifier");
    std::fs::write(
        format!("{}/config.json", options.out),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write config");
    leave_after_writing();
}

fn cross_entropy(logits: Tensor<2>, labels: &Tensor<2, Int>, _device: &Device) -> Tensor<1> {
    // log-softmax then gather the log-probability of the correct class.
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    log_probs.gather(1, labels.clone()).mul_scalar(-1.0).mean()
}

/// Stream one frame by episode and time index; no dataset is held in memory.
fn world_frame_at(path: &str, episode: usize, time: usize, device: &Device) -> Tensor<4> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).expect("open the frame file");
    let index = episode * 129 + time;
    let mut raw = vec![0u8; 3 * 64 * 64];
    file.seek(SeekFrom::Start((index * 3 * 64 * 64) as u64))
        .expect("seek frames");
    file.read_exact(&mut raw).expect("read frame");
    let mut pixels = Vec::with_capacity(raw.len());
    let plane = 64 * 64;
    for channel in 0..3 {
        for pixel in 0..plane {
            pixels.push(raw[3 * pixel + channel] as f32 / 127.5 - 1.0);
        }
    }
    Tensor::<4>::from_data(TensorData::new(pixels, [1, 3, 64, 64]), device)
}

/// Tokenizer training: L1+MSE reconstruction and an edge loss, with an EMA
/// codebook. No GAN — the reference found it unstable.
fn train_world_vq(options: &Options) {
    let device = options.device().autodiff();
    let options = &options;
    let prefix = if options.data.is_empty() {
        "/home/adeerkhan/tmp/world/data".to_string()
    } else {
        options.data.clone()
    };
    let episodes = 400;
    let steps_per_episode = 129;
    let frames = episodes * steps_per_episode;
    let config = VqConfig {
        dim: options.dim.min(64),
        codes: if options.timesteps > 0 {
            options.timesteps
        } else {
            64
        },
        ..VqConfig::default()
    };
    let mut model = VqVae::new(&config, &device);
    let mut optimizer = AdamWConfig::new().init();
    println!(
        "VQ-VAE dim={} codes={} frames={}",
        config.dim, config.codes, frames
    );

    let started = std::time::Instant::now();
    for step in 1..=options.steps {
        let mut rng = diffusion::data::Rng::new(options.seed + step as u64);
        let episode = rng.below(380);
        let time = rng.below(128);
        let x = world_frame_at(&format!("{prefix}/frames.bin"), episode, time, &device);
        let (recon, _, commit) = model.forward(x.clone());
        let error = x.clone().sub(recon.clone());
        let rec = error
            .clone()
            .abs()
            .mean()
            .add(error.powf_scalar(2.0).mean());
        let loss = rec.add(edge_loss(&recon, &x)).add(commit.mul_scalar(0.25));
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let progress = step as f64 / options.steps as f64;
        let lr = 1e-4 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        model = optimizer.step(lr, model, grads);
        model.update_codebook(&model.encode(x.clone()).detach());

        if step == 1 || step % 100 == 0 {
            println!("step {step:>5} rec {value:.4}");
        }
    }
    println!("trained in {:.1}s", started.elapsed().as_secs_f64());
    std::fs::create_dir_all(&options.out).ok();
    model
        .clone()
        .valid()
        .into_record()
        .save(format!("{}/model", options.out))
        .expect("save the tokenizer");
    std::fs::write(
        format!("{}/config.json", options.out),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write config");
    leave_after_writing();
}

/// Encode all recorded frames into a flat i64 token file the world model reads.
fn encode_world_tokens(options: &Options) {
    use std::io::Write;
    let device = options.device();
    let prefix = if options.data.is_empty() {
        "/home/adeerkhan/tmp/world/data".to_string()
    } else {
        options.data.clone()
    };
    let checkpoints = if options.checkpoint.is_empty() {
        "/home/adeerkhan/tmp/world/vq".to_string()
    } else {
        options.checkpoint.clone()
    };
    let config: VqConfig = serde_json::from_str(
        &std::fs::read_to_string(format!("{checkpoints}/config.json"))
            .expect("read tokenizer config"),
    )
    .expect("parse tokenizer config");
    let mut model = VqVae::new(&config, &device);
    let record = ModuleRecord::load(format!("{checkpoints}/model")).expect("load tokenizer");
    model = model.load_record(record);

    let episodes = 400;
    let steps = 129;
    let mut file =
        std::fs::File::create(format!("{prefix}/tokens.bin")).expect("create token file");
    let started = std::time::Instant::now();
    for episode in 0..episodes {
        for time in 0..steps {
            let x = world_frame_at(&format!("{prefix}/frames.bin"), episode, time, &device);
            let ids = model.ids(model.encode(x));
            let values = ids
                .into_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap();
            for value in values {
                file.write_all(&value.to_le_bytes()).expect("write token");
            }
        }
        if episode % 100 == 0 {
            println!("encoded episode {episode}");
        }
    }
    println!(
        "encoded {episodes} episodes in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    leave_after_writing();
}

/// World-model training on tokens encoded by the VQ-VAE. The dataset is packed
/// once into a flat i64 token file so training stays memory-bounded.
fn train_world_model(options: &Options) {
    let device = options.device().autodiff();
    let options = &options;
    let prefix = if options.data.is_empty() {
        "/home/adeerkhan/tmp/world/data".to_string()
    } else {
        options.data.clone()
    };
    let token_path = format!("{prefix}/tokens.bin");
    if !std::path::Path::new(&token_path).exists() {
        panic!("no {token_path}; run scripts/world_tokenize.py first");
    }
    let tokenizer_config: VqConfig = serde_json::from_str(
        &std::fs::read_to_string(format!("{prefix}/vq-config.json")).unwrap_or_else(|_| {
            std::fs::read_to_string("/home/adeerkhan/tmp/world/vq2/config.json")
                .expect("read tokenizer config")
        }),
    )
    .expect("parse tokenizer config");
    let config = WorldConfig {
        codes: tokenizer_config.codes,
        dim: options.dim.min(128),
        layers: options.depth.min(4),
        heads: options.heads.min(8),
        context: 4,
    };
    let mut model = WorldModel::new(&config, &device);
    let mut optimizer = AdamWConfig::new().init();
    let total = std::fs::metadata(&token_path).unwrap().len() as usize / (8 * TOKENS);
    println!(
        "world model dim={} layers={} heads={} tokens={}",
        config.dim, config.layers, config.heads, total
    );

    let started = std::time::Instant::now();
    for step in 1..=options.steps {
        let (seq, actions, targets) = world_batch(
            WorldBatchArgs {
                mask: config.codes as i64,
                tokens_path: &token_path,
                actions_path: &format!("{prefix}/actions.bin"),
                total,
                batch: options.batch,
                context: config.context,
                seed: options.seed + step as u64,
            },
            &device,
        );
        let logits = model.forward_seq(seq, actions);
        let [batch, frames, tokens, codes] = logits.dims();
        let flat = logits.reshape([batch * frames * tokens, codes]);
        let target = targets.reshape([batch * frames * tokens]);
        let loss = burn::tensor::activation::log_softmax(flat, 1)
            .gather(1, target.clone().reshape([-1, 1]))
            .mul_scalar(-1.0)
            .mean();
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, grads);

        if step == 1 || step % 100 == 0 {
            println!("step {step:>5} loss {value:.4}");
        }
    }
    println!("trained in {:.1}s", started.elapsed().as_secs_f64());
    std::fs::create_dir_all(&options.out).ok();
    model
        .clone()
        .valid()
        .into_record()
        .save(format!("{}/model", options.out))
        .expect("save the world model");
    std::fs::write(
        format!("{}/config.json", options.out),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write config");
    leave_after_writing();
}

/// Roll out a dream: warm-start from real frames, then feed the model its own
/// predictions and compare them with the real room under the same actions.
fn dream_rollout(options: &Options) {
    let device = options.device();
    let prefix = if options.data.is_empty() {
        "/home/adeerkhan/tmp/world/data".to_string()
    } else {
        options.data.clone()
    };
    let checkpoints = if options.checkpoint.is_empty() {
        "/home/adeerkhan/tmp/world/wm".to_string()
    } else {
        options.checkpoint.clone()
    };
    let vq_dir = if options.eval_data.is_empty() {
        "/home/adeerkhan/tmp/world/vq".to_string()
    } else {
        options.eval_data.clone()
    };
    let wm_config: WorldConfig = serde_json::from_str(
        &std::fs::read_to_string(format!("{checkpoints}/config.json")).expect("read world config"),
    )
    .expect("parse world config");
    let vq_config: VqConfig = serde_json::from_str(
        &std::fs::read_to_string(format!("{vq_dir}/config.json")).expect("read vq config"),
    )
    .expect("parse vq config");

    let mut vq = VqVae::new(&vq_config, &device);
    vq = vq.load_record(ModuleRecord::load(format!("{vq_dir}/model")).expect("load tokenizer"));
    let mut wm = WorldModel::new(&wm_config, &device);
    wm = wm
        .load_record(ModuleRecord::load(format!("{checkpoints}/model")).expect("load world model"));

    let context = wm_config.context;
    let steps = options.steps.max(1);
    let mut history: Vec<Tensor<1, Int>> = Vec::new();
    let mut actions: Vec<i64> = Vec::new();
    for time in 0..context {
        let frame = world_frame_at(&format!("{prefix}/frames.bin"), 0, time, &device);
        let ids = vq.ids(vq.encode(frame)).reshape([TOKENS]);
        history.push(ids);
        if time > 0 {
            actions.push(diffusion::data::Rng::new(options.seed + time as u64).below(5) as i64);
        }
    }
    let mut errors = Vec::new();
    for step in 0..steps {
        let action = diffusion::data::Rng::new(options.seed + 1000 + step as u64).below(5) as i64;
        actions.push(action);
        let seq = Tensor::<3, Int>::from_data(
            TensorData::new(
                history
                    .iter()
                    .flat_map(|ids| {
                        ids.clone()
                            .into_data()
                            .convert::<i64>()
                            .try_to_vec::<i64>()
                            .unwrap()
                    })
                    .collect::<Vec<_>>(),
                [1, context, TOKENS],
            ),
            &device,
        );
        let acts = Tensor::<2, Int>::from_data(
            TensorData::new(actions.clone(), [1, actions.len()]),
            &device,
        );
        let predicted = wm.predict(seq, acts);
        let dream = vq.decode_ids(predicted.reshape([TOKENS, 1]));

        let real_frame =
            world_frame_at(&format!("{prefix}/frames.bin"), 0, context + step, &device);
        let error = dream
            .clone()
            .sub(real_frame.clone())
            .abs()
            .mean()
            .into_scalar::<f32>();
        errors.push(error * 127.5);

        // [real | dream] strip for one image per 10 steps.
        if step % 10 == 0 {
            let strip = Tensor::cat(vec![real_frame, dream.clone()], 3);
            save_samples(strip, &format!("{}/step-{step:04}.png", options.out));
        }
        let next = vq.ids(vq.encode(dream.clone())).reshape([TOKENS]);
        history.remove(0);
        history.push(next);
        actions.remove(0);
    }
    let mean = |range: std::ops::Range<usize>| -> f32 {
        if range.start >= errors.len() {
            return f32::NAN;
        }
        let end = range.end.min(errors.len());
        errors[range.start..end].iter().sum::<f32>() / (end - range.start) as f32
    };
    let (early, mid, late) = (mean(0..10), mean(10..30), mean(30..errors.len()));
    println!("dream MAE (0-255): steps 1-10 {early:.2}, 11-30 {mid:.2}, 31+ {late:.2}");
    leave_after_writing();
}

/// One training window: context frames + masked target frame, with the action
/// that produced each frame. Read straight from the flat token file.
struct WorldBatchArgs<'a> {
    mask: i64,
    tokens_path: &'a str,
    actions_path: &'a str,
    total: usize,
    batch: usize,
    context: usize,
    seed: u64,
}

fn world_batch(
    args: WorldBatchArgs<'_>,
    device: &Device,
) -> (Tensor<3, Int>, Tensor<2, Int>, Tensor<3, Int>) {
    use std::io::{Read, Seek, SeekFrom};
    let mut tokens_file = std::fs::File::open(args.tokens_path).expect("open tokens");
    let mut actions_file = std::fs::File::open(args.actions_path).expect("open actions");
    let mut rng = diffusion::data::Rng::new(args.seed);
    let frames = args.context + 1;
    let mut seq = Vec::with_capacity(args.batch * frames * TOKENS);
    let mut acts = Vec::with_capacity(args.batch * frames);
    let mut targets = Vec::with_capacity(args.batch * frames * TOKENS);

    for _ in 0..args.batch {
        let episode = rng.below(args.total / 129);
        let time = rng.below(128 - frames) + frames;
        for slot in 0..frames {
            let index = episode * 129 + time - frames + slot;
            let mut raw = vec![0u8; TOKENS * 8];
            tokens_file
                .seek(SeekFrom::Start((index * TOKENS * 8) as u64))
                .expect("seek tokens");
            tokens_file.read_exact(&mut raw).expect("read tokens");
            let values: Vec<i64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| i64::from_le_bytes(*bytes))
                .collect();
            if slot == frames - 1 {
                targets.extend_from_slice(&values);
                seq.extend(std::iter::repeat_n(args.mask, TOKENS));
            } else {
                seq.extend_from_slice(&values);
                targets.extend(std::iter::repeat_n(args.mask, TOKENS));
            }
            // Actions are per episode, one per transition frame i -> i+1.
            // Frame `index` is global (episode*129 + local), so recover the
            // local frame and read the action that produced it, if any.
            let local_frame = index - episode * 129;
            let mut raw_action = [0u8; 1];
            if local_frame >= 1 {
                actions_file
                    .seek(SeekFrom::Start((episode * 128 + local_frame - 1) as u64))
                    .expect("seek actions");
                actions_file
                    .read_exact(&mut raw_action)
                    .expect("read action");
            }
            acts.push(raw_action[0] as i64);
        }
    }
    (
        Tensor::<3, Int>::from_data(TensorData::new(seq, [args.batch, frames, TOKENS]), device),
        Tensor::<2, Int>::from_data(TensorData::new(acts, [args.batch, frames]), device),
        Tensor::<3, Int>::from_data(
            TensorData::new(targets, [args.batch, frames, TOKENS]),
            device,
        ),
    )
}

/// One-off: rewrite a record saved by the enum wrapper into the inner
/// backbone's own layout. Plain `sample` works afterwards.
fn migrate_record(options: &Options) {
    let directory = if options.checkpoint.is_empty() {
        panic!("migrate needs --ckpt")
    } else {
        options.checkpoint.clone()
    };
    let text = std::fs::read_to_string(format!("{directory}/config.json")).expect("read config");
    let config: CheckpointConfig = serde_json::from_str(&text).expect("parse config");
    let device = options.device();
    let record = ModuleRecord::load(format!("{directory}/model")).expect("load the record");
    let model = if config.backbone == "vit" {
        Unet::Vit(Vit::new(&vit_config(&config), &device))
    } else {
        Unet::new(&unet_config(&config), &device)
    }
    .load_record(record);
    model.save_record(&format!("{directory}/model"));
    println!("migrated {directory}");
    leave_after_writing();
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Write a reflow pair cache: `count` fresh noises and the images a frozen flow
/// model produces from them. Training then reads the cache instead of running
/// the ODE every step, which is what makes reflow affordable here.
fn make_pairs(options: &Options) {
    let device = options.device();
    let directory = if options.checkpoint.is_empty() {
        panic!("make-pairs needs --ckpt, the base flow model")
    } else {
        options.checkpoint.clone()
    };
    let text = std::fs::read_to_string(format!("{directory}/config.json")).expect("read config");
    let config: CheckpointConfig = serde_json::from_str(&text).expect("parse config");
    let record = ModuleRecord::load(format!("{directory}/model")).expect("load the base model");
    let model = build_model(&config, &device).load_record_from(record);
    let flow = RectifiedFlow::new(model, config.image_size).with_config(flow_config(&config));

    let steps = if options.sampling_timesteps > 0 {
        options.sampling_timesteps
    } else {
        20
    };
    let elements = 3 * config.image_size * config.image_size;
    let mut noises: Vec<f32> = Vec::with_capacity(options.count * elements);
    let mut samples: Vec<f32> = Vec::with_capacity(options.count * elements);
    let started = std::time::Instant::now();
    let mut done = 0;
    while done < options.count {
        let batch = options.batch.min(options.count - done);
        let noise = Tensor::<4>::random(
            [batch, 3, config.image_size, config.image_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let pair = flow.sample_from(noise.clone(), steps, FlowSolver::Midpoint, &device);
        noises.extend(
            noise
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap(),
        );
        samples.extend(
            pair.into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap(),
        );
        done += batch;
    }
    let prefix = if options.pairs.is_empty() {
        format!("{directory}/pairs-{steps}")
    } else {
        options.pairs.clone()
    };
    let bytes = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    std::fs::write(format!("{prefix}-noise.bin"), bytes(&noises)).expect("write noise pairs");
    std::fs::write(format!("{prefix}-sample.bin"), bytes(&samples)).expect("write sample pairs");
    std::fs::write(
        format!("{prefix}.json"),
        serde_json::json!({"count": options.count, "steps": steps, "shape": [3, config.image_size, config.image_size]})
            .to_string(),
    )
    .expect("write pair metadata");
    println!(
        "wrote {} pairs ({steps} ODE steps) to {prefix}-noise.bin / {prefix}-sample.bin in {:.1}s",
        options.count,
        started.elapsed().as_secs_f64()
    );
    leave_after_writing();
}

/// Reflow, the second rectification pass from the rectified-flow paper.
///
/// The live model is trained on pairs `(noise, frozen.sample(noise))`: the
/// frozen copy of the base model generates the endpoint, and the live model
/// learns the straight path to it. The pair is coupled by the *same* noise, so
/// the trajectory becomes straighter and one or two ODE steps become usable.
fn train_reflow(options: &Options) {
    // The frozen sampler runs on the plain device: Burn has no `no_grad`, and
    // an autodiff device would record every step of the ODE it draws.
    let base_device = options.device();
    let device = base_device.clone().autodiff();
    let mut options = options.clone();
    if options.sampling_timesteps == 0 {
        options.sampling_timesteps = 50;
    }
    let options = &options;
    let directory = if options.checkpoint.is_empty() {
        panic!("reflow needs --ckpt, the base flow model");
    } else {
        options.checkpoint.clone()
    };
    let text = std::fs::read_to_string(format!("{directory}/config.json")).expect("read config");
    let config: CheckpointConfig = serde_json::from_str(&text).expect("parse config");
    assert!(config.flow, "reflow needs a flow checkpoint");

    let record = ModuleRecord::load(format!("{directory}/model")).expect("load the base model");
    let frozen = build_model(&config, &base_device).load_record_from(record.clone());
    let mut live = build_model(&config, &device).load_record_from(record);
    let frozen_flow =
        RectifiedFlow::new(frozen, config.image_size).with_config(flow_config(&config));
    println!(
        "reflow from {directory}: {} params, {} pair steps, {} training steps",
        diffusion::test_support::param_count(&live),
        options.sampling_timesteps,
        options.steps
    );

    std::fs::create_dir_all(&options.out).ok();
    let progress_path = format!("{}/progress.json", options.out);
    let mut start = 1usize;
    if options.resume && std::path::Path::new(&progress_path).exists() {
        let text = std::fs::read_to_string(&progress_path).expect("read progress");
        let progress: serde_json::Value = serde_json::from_str(&text).expect("parse progress");
        start = progress["step"].as_u64().unwrap_or(0) as usize + 1;
        let record = ModuleRecord::load(format!("{}/model-live", options.out))
            .expect("load the resume model");
        live = live.load_record_from(record);
        println!("resumed at step {}", start - 1);
    }

    let elements = 3 * config.image_size * config.image_size;
    let (cached_noise, cached_sample) = if options.pairs.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        (
            read_f32(&format!("{}-noise.bin", options.pairs)),
            read_f32(&format!("{}-sample.bin", options.pairs)),
        )
    };
    let cached_count = cached_noise.len() / elements;
    if cached_count > 0 {
        println!("pair cache {cached_count} pairs from {}", options.pairs);
    }

    let mut ema = Ema::new(&live, options.ema_decay);
    let mut optimizer = AdamWConfig::new().init();
    let started = std::time::Instant::now();
    let mut last = (start.saturating_sub(1), std::time::Instant::now());

    for step in start..=options.steps {
        let mut rng = diffusion::data::Rng::new(
            options.seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        // The coupling the paper rectifies: this noise and the image it produces.
        // With a cache the pairs are replayed; otherwise the frozen model draws
        // one on the plain device, so the ODE leaves no graph behind.
        let (noise, pair) = if cached_count > 0 {
            let mut noise_values = Vec::with_capacity(options.batch * elements);
            let mut sample_values = Vec::with_capacity(options.batch * elements);
            for _ in 0..options.batch {
                let index = rng.below(cached_count);
                noise_values
                    .extend_from_slice(&cached_noise[index * elements..(index + 1) * elements]);
                sample_values
                    .extend_from_slice(&cached_sample[index * elements..(index + 1) * elements]);
            }
            (
                Tensor::<4>::from_data(
                    TensorData::new(
                        noise_values,
                        [options.batch, 3, config.image_size, config.image_size],
                    ),
                    &device,
                ),
                Tensor::<4>::from_data(
                    TensorData::new(
                        sample_values,
                        [options.batch, 3, config.image_size, config.image_size],
                    ),
                    &device,
                ),
            )
        } else {
            let noise = Tensor::<4>::random(
                [options.batch, 3, config.image_size, config.image_size],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            let pair = frozen_flow
                .sample_from(
                    noise.clone().to_device(&base_device),
                    options.sampling_timesteps,
                    FlowSolver::Midpoint,
                    &base_device,
                )
                .to_device(&device);
            (noise, pair)
        };
        let times: Vec<f32> = (0..options.batch).map(|_| rng.unit()).collect();

        let flow =
            RectifiedFlow::new(live.clone(), config.image_size).with_config(flow_config(&config));
        let loss = flow.p_losses(pair, &times, noise, &device, None);
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &live);
        live = optimizer.step(2e-4, live, grads);
        ema.update(&live);

        if step == 1 || step % 100 == 0 {
            let now = std::time::Instant::now();
            let interval = (step - last.0).max(1) as f64;
            let rate = now.duration_since(last.1).as_secs_f64() / interval;
            last = (step, now);
            println!("step {step:>6}  loss {value:.5}  {rate:.3} s/step");
        }

        if step % options.checkpoint_every == 0 && step < options.steps {
            live.save_record(&format!("{}/model-live", options.out));
            ema.model.save_record(&format!("{}/model", options.out));
            let sampler = RectifiedFlow::new(ema.model.valid(), config.image_size)
                .with_config(flow_config(&config));
            let grid = sampler.sample(16, 4, &device);
            save_samples(grid, &format!("{}/grid-{step:06}.png", options.out));
            std::fs::write(
                format!("{}/config.json", options.out),
                serde_json::to_string_pretty(&config).expect("serialise"),
            )
            .expect("write the config");
            let progress = serde_json::json!({
                "step": step,
                "config": config,
                "ema_decay": options.ema_decay,
                "reflow": true,
            });
            std::fs::write(
                &progress_path,
                serde_json::to_string_pretty(&progress).expect("serialise"),
            )
            .expect("write progress");
            println!("checkpoint at step {step}");
        }
    }
    println!("trained in {:.1}s", started.elapsed().as_secs_f64());

    let progress = serde_json::json!({
        "step": options.steps,
        "config": config,
        "ema_decay": options.ema_decay,
        "reflow": true,
    });
    std::fs::write(
        &progress_path,
        serde_json::to_string_pretty(&progress).expect("serialise"),
    )
    .expect("write progress");
    live.save_record(&format!("{}/model-live", options.out));
    ema.model.save_record(&format!("{}/model", options.out));
    std::fs::write(
        format!("{}/config.json", options.out),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write the config");

    let started = std::time::Instant::now();
    let sampler =
        RectifiedFlow::new(ema.model.valid(), config.image_size).with_config(flow_config(&config));
    let images = sampler.sample(options.samples, options.sampling_timesteps, &device);
    println!(
        "sampled {} images in {:.2}s with {} steps",
        options.samples,
        started.elapsed().as_secs_f64(),
        options.sampling_timesteps
    );
    save_samples(images, &format!("{}/samples-ema.png", options.out));
    leave_after_writing();
}

/// JetViT's post-training attention search, in its parameter-compatible form.
///
/// Starting from a trained full-attention ViT, each block is switched to ReLU
/// linear attention in turn — same `to_qkv`/`to_out` weights, so nothing is
/// retrained — and the model is kept if the flow loss on a fixed set of
/// *training* batches does not rise by more than `tolerance`. The chosen
/// schedule is written to `--out`, which then samples exactly like any other
/// checkpoint.
fn search_attention(options: &Options) {
    let device = options.device();
    let directory = if options.checkpoint.is_empty() {
        panic!("search-attention needs --ckpt")
    } else {
        options.checkpoint.clone()
    };
    let text = std::fs::read_to_string(format!("{directory}/config.json")).expect("read config");
    let mut config: CheckpointConfig = serde_json::from_str(&text).expect("parse config");
    if config.backbone != "vit" {
        panic!("search-attention only applies to the ViT backbone");
    }
    let output = if options.out.is_empty() || options.out == "artifacts/ddpm" {
        format!("{directory}-search")
    } else {
        options.out.clone()
    };

    // Start from all-full attention and the checkpoint's EMA weights.
    config.attention = vec!["full".to_string(); config.depth];
    let base = build_model(&config, &device).load_record_from(
        ModuleRecord::load(format!("{directory}/model")).expect("load the model"),
    );

    // Fixed evaluation batches from the training split: the held-out split is
    // never read here.
    let images = diffusion::data::RawImages::load(&config.data, config.image_size);
    let mut rng = diffusion::data::Rng::new(options.seed);
    let batches: Vec<(Tensor<4>, Vec<f32>, Tensor<4>)> = (0..8)
        .map(|_| {
            let data = images.batch(options.batch, rng.next_u64(), &device);
            let times: Vec<f32> = (0..options.batch).map(|_| rng.unit()).collect();
            let noise = Tensor::<4>::random(
                [options.batch, 3, config.image_size, config.image_size],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            (data, times, noise)
        })
        .collect();

    let evaluate = |model: &Unet| -> f32 {
        let flow = RectifiedFlow::new(model.clone(), config.image_size);
        let total: f32 = batches
            .iter()
            .map(|(data, times, noise)| {
                flow.p_losses(data.clone(), times, noise.clone(), &device, None)
                    .into_scalar::<f32>()
            })
            .sum();
        total / batches.len() as f32
    };

    let baseline = evaluate(&base);
    println!("full attention: loss {baseline:.5}");

    let mut kinds = vec!["full".to_string(); config.depth];
    let tolerance = 1.02;
    let mut current = base;
    for block in 0..config.depth {
        let mut candidate = kinds.clone();
        candidate[block] = "linear".to_string();
        config.attention = candidate.clone();
        let model = build_model(&config, &device).load_record_from(
            ModuleRecord::load(format!("{directory}/model")).expect("load weights"),
        );
        let loss = evaluate(&model);
        let accepted = loss <= baseline * tolerance;
        println!(
            "block {block}: linear loss {loss:.5} ({:+.2}%) -> {}",
            (loss / baseline - 1.0) * 100.0,
            if accepted { "keep linear" } else { "keep full" }
        );
        if accepted {
            kinds = candidate;
            current = model;
        }
    }

    config.attention = kinds.clone();
    std::fs::create_dir_all(&output).ok();
    current.save_record(&format!("{output}/model"));
    std::fs::write(
        format!("{output}/config.json"),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write the config");
    let source_progress = std::fs::read_to_string(format!("{directory}/progress.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let mut progress = source_progress;
    progress["attention"] = serde_json::json!(kinds);
    progress["linear_blocks"] =
        serde_json::json!(kinds.iter().filter(|kind| *kind == "linear").count());
    progress["source"] = serde_json::json!(directory);
    std::fs::write(
        format!("{output}/progress.json"),
        serde_json::to_string_pretty(&progress).expect("serialise"),
    )
    .expect("write progress");
    println!("wrote {output} with kinds {kinds:?}");
    leave_after_writing();
}

fn build(config: &CheckpointConfig, device: &Device) -> GaussianDiffusion {
    let model = build_model(config, device);
    GaussianDiffusion::new(
        model,
        config.image_size,
        config.timesteps,
        Some(config.sampling_timesteps),
        Objective::PredV,
        BetaSchedule::Sigmoid,
        false,
        5.0,
    )
}

/// One training loop for both objectives. The DDPM regresses a discrete-time
/// target (noise, x0 or v); the rectified flow regresses a continuous velocity.
#[derive(Clone)]
enum Trainer {
    /// Both boxed: the UNet and schedule buffers make either variant large,
    /// and the enum is stored by value in a few places.
    Diffusion(Box<GaussianDiffusion>),
    Flow(Box<RectifiedFlow>),
}

impl Trainer {
    fn new(config: &CheckpointConfig, device: &Device) -> Self {
        if config.flow {
            Self::Flow(Box::new(
                RectifiedFlow::new(build_model(config, device), config.image_size)
                    .with_config(flow_config(config)),
            ))
        } else {
            Self::Diffusion(Box::new(build(config, device)))
        }
    }

    fn model(&self) -> &Unet {
        match self {
            Self::Diffusion(diffusion) => &diffusion.model,
            Self::Flow(flow) => &flow.model,
        }
    }

    fn set_model(&mut self, model: Unet) {
        match self {
            Self::Diffusion(diffusion) => diffusion.model = model,
            Self::Flow(flow) => flow.model = model,
        }
    }

    /// The same objective with another set of weights, for EMA sampling.
    fn with_model(&self, model: Unet) -> Self {
        match self {
            Self::Diffusion(diffusion) => {
                let mut copy = diffusion.clone();
                copy.model = model;
                Self::Diffusion(copy)
            }
            Self::Flow(flow) => Self::Flow(Box::new(RectifiedFlow::new(model, flow.image_size))),
        }
    }

    fn p_losses(
        &self,
        x0: Tensor<4>,
        rng: &mut diffusion::data::Rng,
        timesteps: usize,
        noise: Tensor<4>,
        device: &Device,
        ema: Option<&Unet>,
    ) -> Tensor<1> {
        match self {
            Self::Diffusion(diffusion) => {
                let times: Vec<i64> = (0..x0.dims()[0])
                    .map(|_| rng.below(timesteps) as i64)
                    .collect();
                diffusion.p_losses(x0, &times, noise, device)
            }
            Self::Flow(flow) => {
                let times: Vec<f32> = (0..x0.dims()[0]).map(|_| rng.unit()).collect();
                flow.p_losses(x0, &times, noise, device, ema)
            }
        }
    }

    fn sample(&self, batch: usize, steps: usize, device: &Device) -> Tensor<4> {
        match self {
            Self::Diffusion(diffusion) => diffusion.sample(batch, device),
            Self::Flow(flow) => flow.sample(batch, steps, device),
        }
    }
}

fn train(options: &Options) {
    let device = options.device().autodiff();
    let mut options = options.clone();
    if options.sampling_timesteps == 0 {
        options.sampling_timesteps = if options.flow {
            50
        } else {
            50.min(options.timesteps)
        };
    }
    if !options.flow {
        options.sampling_timesteps = options.sampling_timesteps.min(options.timesteps);
    }
    let options = &options;
    let config = checkpoint_config(options);
    let mut trainer = Trainer::new(&config, &device);
    println!(
        "{} dim={} params={} ema_decay={}",
        if config.flow { "RectifiedFlow" } else { "Unet" },
        config.dim,
        diffusion::test_support::param_count(trainer.model()),
        options.ema_decay
    );

    std::fs::create_dir_all(&options.out).ok();
    let progress_path = format!("{}/progress.json", options.out);
    let mut start = 1usize;
    if options.resume && std::path::Path::new(&progress_path).exists() {
        let text = std::fs::read_to_string(&progress_path).expect("read progress");
        let progress: serde_json::Value = serde_json::from_str(&text).expect("parse progress");
        start = progress["step"].as_u64().unwrap_or(0) as usize + 1;
        let record = ModuleRecord::load(format!("{}/model-live", options.out))
            .expect("load the resume model");
        trainer.set_model(trainer.model().clone().load_record_from(record));
        println!("resumed at step {}", start - 1);
    }

    let mut ema = Ema::new(trainer.model(), options.ema_decay);
    let mut optimizer = AdamWConfig::new().init();
    let raw = if options.data.is_empty() {
        None
    } else {
        Some(diffusion::data::RawImages::load(
            &options.data,
            options.image_size,
        ))
    };
    if let Some(images) = &raw {
        println!("dataset {} ({} images)", options.data, images.len());
    }

    let started = std::time::Instant::now();
    let mut last = (start.saturating_sub(1), std::time::Instant::now());
    for step in start..=options.steps {
        // Seed per step, so a resumed run sees exactly the same batches.
        let mut rng = diffusion::data::Rng::new(
            options.seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let x0 = match &raw {
            Some(images) => images.batch(options.batch, rng.next_u64(), &device),
            None => {
                diffusion::data::batch(options.batch, options.image_size, rng.next_u64(), &device)
            }
        };
        let noise = Tensor::<4>::random(
            [options.batch, 3, options.image_size, options.image_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let loss = trainer.p_losses(
            x0,
            &mut rng,
            options.timesteps,
            noise,
            &device,
            Some(&ema.model),
        );
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), trainer.model());

        if !options.dump.is_empty() && step == options.steps {
            let dump = diffusion::inspect::dump(trainer.model(), Some(&grads));
            if let Some(parent) = std::path::Path::new(&options.dump).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(
                &options.dump,
                serde_json::to_string(&dump).expect("serialise the dump"),
            )
            .expect("write the dump");
            println!(
                "wrote {} ({} weight layers, {} gradient layers)",
                options.dump,
                dump.weights.len(),
                dump.gradients.len()
            );
        }

        let updated = optimizer.step(2e-4, trainer.model().clone(), grads);
        trainer.set_model(updated);
        ema.update(trainer.model());

        if step == 1 || step % 100 == 0 {
            let now = std::time::Instant::now();
            let interval = (step - last.0).max(1) as f64;
            let rate = now.duration_since(last.1).as_secs_f64() / interval;
            last = (step, now);
            println!("step {step:>6}  loss {value:.5}  {rate:.3} s/step");
        }

        // Periodic checkpoint: live weights for resume, EMA weights and a grid
        // for watching progress.
        if step % options.checkpoint_every == 0 && step < options.steps {
            trainer
                .model()
                .save_record(&format!("{}/model-live", options.out));
            ema.model.save_record(&format!("{}/model", options.out));
            let sampler = trainer.with_model(ema.model.valid());
            let grid = sampler.sample(16, options.sampling_timesteps, &device);
            save_samples(grid, &format!("{}/grid-{step:06}.png", options.out));
            std::fs::write(
                format!("{}/config.json", options.out),
                serde_json::to_string_pretty(&config).expect("serialise"),
            )
            .expect("write the config");
            let progress = serde_json::json!({
                "step": step,
                "config": config,
                "ema_decay": options.ema_decay,
            });
            std::fs::write(
                &progress_path,
                serde_json::to_string_pretty(&progress).expect("serialise"),
            )
            .expect("write progress");
            println!("checkpoint at step {step}");
        }
    }
    println!("trained in {:.1}s", started.elapsed().as_secs_f64());

    // Record the step actually reached, so a snapshot of this directory names
    // the weights it contains instead of the last periodic checkpoint.
    let progress = serde_json::json!({
        "step": options.steps,
        "config": config,
        "ema_decay": options.ema_decay,
    });
    std::fs::write(
        &progress_path,
        serde_json::to_string_pretty(&progress).expect("serialise"),
    )
    .expect("write progress");

    // Final: EMA weights for sampling, live weights kept for reference.
    trainer
        .model()
        .save_record(&format!("{}/model-live", options.out));
    ema.model.save_record(&format!("{}/model", options.out));
    std::fs::write(
        format!("{}/config.json", options.out),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write the config");

    let started = std::time::Instant::now();
    let sampled = trainer.with_model(ema.model.valid());
    let images = sampled.sample(options.samples, options.sampling_timesteps, &device);
    let eval = if options.eval_data.is_empty() {
        &options.data
    } else {
        &options.eval_data
    };
    let distance = if eval.is_empty() {
        diffusion::data::nearest_template_distance(images.clone(), options.image_size)
    } else {
        diffusion::data::RawImages::load(eval, options.image_size).nearest_training_distance(
            images.clone(),
            256,
            5,
        )
    };
    println!(
        "sampled {} images from EMA in {:.2}s with {} steps; mean L1 to the held-out set {distance:.3}",
        options.samples,
        started.elapsed().as_secs_f64(),
        options.sampling_timesteps
    );
    save_samples(images, &format!("{}/samples-ema.png", options.out));
    let real = match &raw {
        Some(source) => source.batch(options.samples, 7, &device),
        None => diffusion::data::batch(options.samples, options.image_size, 7, &device),
    };
    save_samples(real, &format!("{}/real.png", options.out));
    println!(
        "wrote {}/samples-ema.png and {}/real.png",
        options.out, options.out
    );
    // wgpu teardown can segfault, abort or hang *after* every artefact is
    // written (GLM-023/024/034). Leave without running the destructors.
    leave_after_writing();
}

fn sample(options: &Options) {
    let device = options.device();
    let directory = if options.checkpoint.is_empty() {
        options.out.clone()
    } else {
        options.checkpoint.clone()
    };
    let text = std::fs::read_to_string(format!("{directory}/config.json")).expect("read config");
    let config: CheckpointConfig = serde_json::from_str(&text).expect("parse config");

    let model = build_model(&config, &device);
    let record = ModuleRecord::load(format!("{directory}/model")).expect("load the model");
    let model = model.load_record_from(record);
    let requested = if options.sampling_timesteps > 0 {
        options.sampling_timesteps
    } else {
        config.sampling_timesteps
    };
    // The DDPM chain length is fixed by the model; the flow's ODE can use any
    // number of steps, so a request below the trained timesteps is meaningful
    // there and is not clamped.
    let sampling = if config.flow {
        // One Euler step is meaningful for a reflowed model.
        requested.max(1)
    } else {
        requested.min(config.timesteps)
    };
    if !config.flow && requested > sampling {
        println!(
            "note: model has {} timesteps; running the full {sampling}-step DDPM chain",
            config.timesteps
        );
    }
    let trainer = if config.flow {
        Trainer::Flow(Box::new(
            RectifiedFlow::new(model, config.image_size).with_config(flow_config(&config)),
        ))
    } else {
        Trainer::Diffusion(Box::new(GaussianDiffusion::new(
            model,
            config.image_size,
            config.timesteps,
            Some(sampling),
            Objective::PredV,
            BetaSchedule::Sigmoid,
            false,
            5.0,
        )))
    };

    let started = std::time::Instant::now();
    let solver = match options.solver.as_str() {
        "euler" => FlowSolver::Euler,
        "heun" => FlowSolver::Heun,
        _ => FlowSolver::Midpoint,
    };
    let images = match &trainer {
        Trainer::Flow(flow) => flow.sample_with_solver(options.samples, sampling, solver, &device),
        _ => trainer.sample(options.samples, sampling, &device),
    };
    // A published checkpoint records the training file it was fitted on, which
    // will not exist on another machine. Sampling must still work; only the
    // nearest-training distance needs the data.
    let raw = if config.data.is_empty() || !std::path::Path::new(&config.data).exists() {
        None
    } else {
        Some(diffusion::data::RawImages::load(
            &config.data,
            config.image_size,
        ))
    };
    let distance = match &raw {
        Some(source) => Some(source.nearest_training_distance(images.clone(), 256, 5)),
        None if config.data.is_empty() => Some(diffusion::data::nearest_template_distance(
            images.clone(),
            config.image_size,
        )),
        None => None,
    };
    match distance {
        Some(distance) => println!(
            "sampled {} images in {:.2}s with {sampling} steps; mean L1 to the {} {distance:.3}",
            options.samples,
            started.elapsed().as_secs_f64(),
            if raw.is_some() {
                "nearest training image"
            } else {
                "closest template"
            }
        ),
        None => println!(
            "sampled {} images in {:.2}s with {sampling} steps; dataset {} not found, distance skipped",
            options.samples,
            started.elapsed().as_secs_f64(),
            config.data
        ),
    }
    let kind = if sampling < config.timesteps {
        "ddim"
    } else {
        "ddpm"
    };
    save_samples(
        images.clone(),
        &format!("{directory}/sample-{kind}-{sampling}.png"),
    );

    if !options.raw_out.is_empty() {
        let [batch, channels, height, width] = images.dims();
        let data = images
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in &data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&options.raw_out, &bytes).expect("write the raw samples");
        std::fs::write(
            format!("{}.json", options.raw_out),
            serde_json::json!({
                "shape": [batch, channels, height, width],
                "sampling_steps": sampling,
                "sampler": kind,
            })
            .to_string(),
        )
        .expect("write the shape");
        println!("wrote {} ({} samples)", options.raw_out, batch);
    }
    leave_after_writing();
}

/// A short run that has to show the loss falling, then samples.
fn smoke(options: &Options) {
    let mut options = options.clone();
    options.steps = options.steps.min(120);
    options.batch = options.batch.min(16);
    options.image_size = options.image_size.min(16);
    options.dim = options.dim.min(16);
    options.timesteps = options.timesteps.min(200);
    options.sampling_timesteps = options.sampling_timesteps.min(20);
    options.samples = options.samples.min(4);
    options.out = format!("{}/smoke", options.out);
    train(&options);
}
