//! Train and sample a DDPM/DDIM on synthetic shape images.

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::store::ModuleRecord;

use diffusion::diffusion::{BetaSchedule, GaussianDiffusion, Objective};
use diffusion::ema::Ema;
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

fn train(options: &Options) {
    let device = options.device().autodiff();
    let mut options = options.clone();
    if options.sampling_timesteps == 0 {
        options.sampling_timesteps = 50.min(options.timesteps);
    }
    options.sampling_timesteps = options.sampling_timesteps.min(options.timesteps);
    let options = &options;
    let config = checkpoint_config(options);
    let mut diffusion = build(&config, &device);
    println!(
        "Unet dim={} params={} ema_decay={}",
        config.dim,
        diffusion::test_support::param_count(&diffusion.model),
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
        diffusion.model = diffusion.model.load_record(record);
        println!("resumed at step {}", start - 1);
    }

    let mut ema = Ema::new(&diffusion.model, options.ema_decay);
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
        let times: Vec<i64> = (0..options.batch)
            .map(|_| rng.below(options.timesteps) as i64)
            .collect();
        let noise = Tensor::<4>::random(
            [options.batch, 3, options.image_size, options.image_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let loss = diffusion.p_losses(x0, &times, noise, &device);
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &diffusion.model);

        if !options.dump.is_empty() && step == options.steps {
            let dump = diffusion::inspect::dump(&diffusion.model, Some(&grads));
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

        diffusion.model = optimizer.step(2e-4, diffusion.model, grads);
        ema.update(&diffusion.model);

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
            diffusion
                .model
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
            let mut sampler = diffusion.clone();
            sampler.model = ema.model.valid();
            let grid = sampler.sample(16, &device);
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

    // Final: EMA weights for sampling, live weights kept for reference.
    diffusion
        .model
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
    let mut sampled = diffusion.clone();
    sampled.model = ema.model.valid();
    let images = sampled.sample(options.samples, &device);
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
    // The model defines the chain: a DDPM request longer than the trained
    // timesteps cannot be honoured, and reporting the request would be a lie.
    let sampling = requested.min(config.timesteps);
    if requested > sampling {
        println!(
            "note: model has {} timesteps; running the full {sampling}-step DDPM chain",
            config.timesteps
        );
    }
    let diffusion = GaussianDiffusion::new(
        model,
        config.image_size,
        config.timesteps,
        Some(sampling),
        Objective::PredV,
        BetaSchedule::Sigmoid,
        false,
        5.0,
    );

    let started = std::time::Instant::now();
    let images = diffusion.sample(options.samples, &device);
    let raw = if config.data.is_empty() {
        None
    } else {
        Some(diffusion::data::RawImages::load(
            &config.data,
            config.image_size,
        ))
    };
    let distance = match &raw {
        Some(source) => source.nearest_training_distance(images.clone(), 256, 5),
        None => diffusion::data::nearest_template_distance(images.clone(), config.image_size),
    };
    println!(
        "sampled {} images in {:.2}s with {sampling} steps; mean L1 to the {} {distance:.3}",
        options.samples,
        started.elapsed().as_secs_f64(),
        if raw.is_some() {
            "nearest training image"
        } else {
            "closest template"
        }
    );
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
