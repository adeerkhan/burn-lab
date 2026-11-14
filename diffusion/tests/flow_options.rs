//! Every flow option the reference exposes: predict modes, losses, the cosmap
//! schedule, immiscible pairing, consistency matching and reflow sampling.

use burn::prelude::*;
use burn::tensor::Distribution;
use diffusion::flow::{FlowConfig, FlowLoss, NoiseSchedule, Predict, RectifiedFlow};
use diffusion::unet::{Unet, UnetConfig};

fn network(device: &Device) -> Unet {
    Unet::new(
        &UnetConfig {
            dim: 8,
            dim_mults: vec![1, 2],
            ..UnetConfig::default()
        },
        device,
    )
}

fn tensors(device: &Device) -> (Tensor<4>, Tensor<4>) {
    (
        Tensor::<4>::random([2, 3, 16, 16], Distribution::Normal(0.0, 0.6), device),
        Tensor::<4>::random([2, 3, 16, 16], Distribution::Normal(0.0, 1.0), device),
    )
}

#[test]
fn cosmap_maps_the_endpoints() {
    let schedule = NoiseSchedule::Cosmap;
    assert!((schedule.map(0.0) - 0.0).abs() < 1e-6);
    assert!((schedule.map(1.0) - 1.0).abs() < 1e-5);
    // Symmetric around the midpoint: above the identity below it, below above.
    assert!((schedule.map(0.5) - 0.5).abs() < 1e-6);
    let half = std::f32::consts::FRAC_1_SQRT_2;
    assert!(
        (schedule.map(0.25) - (1.0 - half)).abs() < 1e-4,
        "{}",
        schedule.map(0.25)
    );
    assert!(
        (schedule.map(0.75) - half).abs() < 1e-4,
        "{}",
        schedule.map(0.75)
    );
}

#[test]
fn every_predict_and_loss_combination_is_finite() {
    let device = Device::flex();
    for predict in [Predict::Flow, Predict::Noise, Predict::Clean] {
        for loss in [FlowLoss::Mse, FlowLoss::PseudoHuber] {
            for immiscible in [false, true] {
                let flow = RectifiedFlow::new(network(&device), 16).with_config(FlowConfig {
                    predict,
                    loss,
                    immiscible,
                    ..FlowConfig::default()
                });
                let (data, noise) = tensors(&device);
                let value = flow
                    .p_losses(data, &[0.2, 0.7], noise, &device, None)
                    .into_scalar::<f32>();
                assert!(
                    value.is_finite() && value >= 0.0,
                    "{predict:?}/{loss:?}/immiscible={immiscible}: {value}"
                );
            }
        }
    }
}

#[test]
fn consistency_matching_needs_and_accepts_an_ema() {
    let device = Device::flex();
    let shadow = network(&device);
    let flow = RectifiedFlow::new(network(&device), 16).with_config(FlowConfig {
        consistency: true,
        ..FlowConfig::default()
    });
    let (data, noise) = tensors(&device);
    let value = flow
        .p_losses(data, &[0.3, 0.6], noise, &device, Some(&shadow))
        .into_scalar::<f32>();
    assert!(
        value.is_finite() && value >= 0.0,
        "consistency loss {value}"
    );
}

#[test]
fn a_single_step_reflow_sample_returns_an_image() {
    let device = Device::flex();
    let flow = RectifiedFlow::new(network(&device), 16);
    let noise = Tensor::<4>::random([2, 3, 16, 16], Distribution::Normal(0.0, 1.0), &device);
    device.seed(11);
    let once = flow.sample_from(noise.clone(), 1, &device);
    device.seed(11);
    let twice = flow.sample_from(noise, 1, &device);
    assert_eq!(once.dims(), [2, 3, 16, 16]);
    let a = once
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    let b = twice
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(a, b);
    assert!(a.iter().all(|value| value.is_finite()));
}
