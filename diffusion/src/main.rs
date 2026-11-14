//! Train and sample a DDPM/DDIM or a rectified flow on synthetic shapes or
//! packed photographs.

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::store::ModuleRecord;

use diffusion::diffusion::{BetaSchedule, GaussianDiffusion, Objective};
use diffusion::ema::Ema;
use diffusion::flow::{FlowConfig, FlowLoss, NoiseSchedule, Predict, RectifiedFlow};
use diffusion::unet::{Unet, UnetConfig};
use diffusion::vit::{AttentionKind, Vit, VitConfig};

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
        "make-pairs" => make_pairs(&options),
        other => panic!("unknown command `{other}`, expected train, sample or smoke"),
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
        let pair = flow.sample_from(noise.clone(), steps, &device);
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
        requested.max(2)
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
    let images = trainer.sample(options.samples, sampling, &device);
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
