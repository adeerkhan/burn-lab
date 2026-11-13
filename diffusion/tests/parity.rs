//! Numeric parity with `denoising_diffusion_pytorch`.
//!
//! `tests/data/schedules.json` is produced by extracting the three schedule
//! functions from the reference source and evaluating them there, so these
//! numbers come from the reference code, not from a retyped formula.

use burn::prelude::*;
use diffusion::data;
use diffusion::diffusion::{BetaSchedule, GaussianDiffusion, Objective};
use diffusion::ema::Ema;
use diffusion::unet::{Unet, UnetConfig};

const REFERENCE: &str = include_str!("data/schedules.json");

fn reference() -> serde_json::Value {
    serde_json::from_str(REFERENCE).expect("parse the reference dump")
}

fn diffusion_with(schedule: BetaSchedule) -> GaussianDiffusion {
    let device = Device::flex();
    let config = UnetConfig {
        dim: 8,
        dim_mults: vec![1, 2],
        channels: 3,
        ..UnetConfig::default()
    };
    let model = Unet::new(&config, &device);
    GaussianDiffusion::new(
        model,
        16,  // image size
        200, // timesteps, matching the dump
        Some(20),
        Objective::PredV,
        schedule,
        false,
        5.0,
    )
}

#[test]
fn the_three_schedules_match_the_reference() {
    let reference = reference();
    for (name, schedule) in [
        ("linear", BetaSchedule::Linear),
        ("cosine", BetaSchedule::Cosine),
        ("sigmoid", BetaSchedule::Sigmoid),
    ] {
        let ours = diffusion_with(schedule);
        let expected = &reference["schedules"][name];

        let betas: Vec<f64> = serde_json::from_value(expected["betas"].clone()).unwrap();
        let alphas: Vec<f64> = serde_json::from_value(expected["alphas_cumprod"].clone()).unwrap();

        assert_eq!(ours.betas.len(), betas.len(), "{name} length");
        for (index, (a, b)) in ours.betas.iter().zip(&betas).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "{name} beta[{index}]: ours {a} reference {b}"
            );
        }
        for (index, (a, b)) in ours.alphas_cumprod.iter().zip(&alphas).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "{name} alphas_cumprod[{index}]: ours {a} reference {b}"
            );
        }
    }
}

#[test]
fn q_sample_matches_the_reference() {
    let reference = reference();
    let device = Device::flex();
    let diffusion = diffusion_with(BetaSchedule::Sigmoid);

    let x0: Vec<f32> = serde_json::from_value(reference["q_sample_inputs"]["x0"].clone()).unwrap();
    let noise: Vec<f32> =
        serde_json::from_value(reference["q_sample_inputs"]["noise"].clone()).unwrap();
    let x0 = Tensor::<4>::from_data(TensorData::new(x0, [1, 2, 1, 2]), &device);
    let noise = Tensor::<4>::from_data(TensorData::new(noise, [1, 2, 1, 2]), &device);

    for t in ["0", "7", "199"] {
        let expected: Vec<f32> = serde_json::from_value(reference["q_sample"][t].clone()).unwrap();
        let actual = diffusion
            .q_sample(x0.clone(), &[t.parse().unwrap()], noise.clone(), &device)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "q_sample t={t} [{index}]: ours {a} reference {b}"
            );
        }
    }
}

#[test]
fn the_unet_preserves_the_image_shape() {
    let device = Device::flex();
    let config = UnetConfig {
        dim: 8,
        dim_mults: vec![1, 2, 4],
        channels: 3,
        ..UnetConfig::default()
    };
    let model = Unet::new(&config, &device);
    assert_eq!(model.downsample_factor(), 4);

    let images = data::batch(2, 16, 3, &device);
    let out = model.forward(images, &[0, 5]);
    assert_eq!(out.dims(), [2, 3, 16, 16]);
}

#[test]
fn ddim_sampling_is_deterministic_on_flex() {
    let device = Device::flex();
    let diffusion = diffusion_with(BetaSchedule::Sigmoid);

    device.seed(7);
    let first = diffusion
        .sample(2, &device)
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    device.seed(7);
    let second = diffusion
        .sample(2, &device)
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(first, second);
}

#[test]
fn the_ema_blends_towards_the_live_weights() {
    let device = Device::flex();
    let mut model = Unet::new(
        &UnetConfig {
            dim: 8,
            dim_mults: vec![1, 2],
            ..UnetConfig::default()
        },
        &device,
    );
    let mut ema = Ema::new(&model, 0.5);

    // Push every parameter to a constant, then check one blend step lands
    // halfway between the shadow and the new value.
    let shadow_before = diffusion::test_support::first_param(&ema.model);
    model = diffusion::test_support::fill(&model, 1.0, &device);
    ema.update(&model);
    let blended = diffusion::test_support::first_param(&ema.model);
    let expected = 0.5 * shadow_before + 0.5;
    assert!((blended - expected).abs() < 1e-5, "{blended} vs {expected}");
    assert_eq!(ema.updates(), 1);
}
