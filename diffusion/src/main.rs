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

    for index in 0..batch {
        let offset = index * 3 * height * width;
        let cell_x = (index % columns) * width;
        let cell_y = (index / columns) * height;
        for y in 0..height {
            for x in 0..width {
                let at = offset + (y * width + x) * 3;
                let to_byte = |value: f32| ((value.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
                image.put_pixel(
                    (cell_x + x) as u32,
                    (cell_y + y) as u32,
                    image::Rgb([
                        to_byte(flat[at]),
                        to_byte(flat[at + 1]),
                        to_byte(flat[at + 2]),
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
    let options = &options;
    let config = checkpoint_config(options);
    let mut diffusion = build(&config, &device);
    println!(
        "Unet dim={} params={}",
        config.dim,
        diffusion::test_support::param_count(&diffusion.model)
    );
    let mut ema = Ema::new(&diffusion.model, 0.995);
    let mut optimizer = AdamWConfig::new().init();
    let mut rng = diffusion::data::Rng::new(11);
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
    for step in 1..=options.steps {
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

        // Capture weights and gradients on the last step, before the update.
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
        if step > options.steps / 10 {
            ema.update(&diffusion.model);
        }

        if step == 1 || step % (options.steps / 10).max(1) == 0 {
            println!("step {step:>5}  loss {value:.5}");
        }
    }
    println!("trained in {:.1}s", started.elapsed().as_secs_f64());

    // Sample from the EMA weights and report how close they land to a template.
    let started = std::time::Instant::now();
    let mut sampled = diffusion.clone();
    sampled.model = ema.model.valid();
    let images = sampled.sample(options.samples, &device);
    let distance = match &raw {
        Some(source) => source.nearest_training_distance(images.clone(), 256, 5),
        None => diffusion::data::nearest_template_distance(images.clone(), options.image_size),
    };
    println!(
        "sampled {} images in {:.2}s ({} steps); mean L1 distance to the {} {:.3}",
        options.samples,
        started.elapsed().as_secs_f64(),
        options.sampling_timesteps,
        if raw.is_some() {
            "nearest training image"
        } else {
            "closest template"
        },
        distance
    );
    save_samples(images, &format!("{}/samples.png", options.out));
    let real = match &raw {
        Some(source) => source.batch(options.samples, 7, &device),
        None => diffusion::data::batch(options.samples, options.image_size, 7, &device),
    };
    save_samples(real, &format!("{}/real.png", options.out));

    std::fs::create_dir_all(&options.out).ok();
    diffusion
        .model
        .clone()
        .valid()
        .into_record()
        .save(format!("{}/model", options.out))
        .expect("save the model");
    std::fs::write(
        format!("{}/config.json", options.out),
        serde_json::to_string_pretty(&config).expect("serialise"),
    )
    .expect("write the config");
    println!(
        "wrote {}/model and {}/config.json",
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
    let sampling = if options.sampling_timesteps > 0 {
        options.sampling_timesteps
    } else {
        config.sampling_timesteps
    };
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
    save_samples(images, &format!("{directory}/sample-{kind}.png"));
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
