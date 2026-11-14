//! Train and sample a DDPM/DDIM or a rectified flow on synthetic shapes or
//! packed photographs.

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::store::ModuleRecord;

use diffusion::diffusion::{BetaSchedule, GaussianDiffusion, Objective};
use diffusion::ema::Ema;
use diffusion::flow::RectifiedFlow;
use diffusion::unet::{Unet, UnetConfig};

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
        other => panic!("unknown command `{other}`, expected train, sample or smoke"),
    }
}

fn build(config: &CheckpointConfig, device: &Device) -> GaussianDiffusion {
    let model = Unet::new(&unet_config(config), device);
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
            Self::Flow(Box::new(RectifiedFlow::new(
                Unet::new(&unet_config(config), device),
                config.image_size,
            )))
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
                flow.p_losses(x0, &times, noise, device)
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
        trainer.set_model(trainer.model().clone().load_record(record));
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

        let loss = trainer.p_losses(x0, &mut rng, options.timesteps, noise, &device);
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
                .clone()
                .valid()
                .into_record()
                .save(format!("{}/model-live", options.out))
                .expect("save live");
            ema.model
                .clone()
                .valid()
                .into_record()
                .save(format!("{}/model", options.out))
                .expect("save ema");
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
        .clone()
        .valid()
        .into_record()
        .save(format!("{}/model-live", options.out))
        .expect("save live");
    ema.model
        .clone()
        .valid()
        .into_record()
        .save(format!("{}/model", options.out))
        .expect("save ema");
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

    let model = Unet::new(&unet_config(&config), &device);
    let record = ModuleRecord::load(format!("{directory}/model")).expect("load the model");
    let model = model.load_record(record);
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
        Trainer::Flow(Box::new(RectifiedFlow::new(model, config.image_size)))
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
