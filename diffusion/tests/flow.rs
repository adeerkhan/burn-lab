//! Rectified-flow tests: the interpolation, the loss, and deterministic sampling.

use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::TensorData;
use diffusion::flow::RectifiedFlow;
use diffusion::unet::{Unet, UnetConfig};

fn values(tensor: Tensor<4>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap()
}

#[test]
fn interpolation_hits_both_endpoints() {
    let device = Device::flex();
    let data = Tensor::<4>::from_data(TensorData::new(vec![1.0f32; 12], [1, 3, 2, 2]), &device);
    let noise = Tensor::<4>::from_data(TensorData::new(vec![-1.0f32; 12], [1, 3, 2, 2]), &device);

    // t = 0 is pure noise, t = 1 is the data, t = 0.5 is exactly between.
    assert!(
        values(RectifiedFlow::interpolate(data.clone(), noise.clone(), 0.0))
            .iter()
            .all(|value| *value == -1.0)
    );
    assert!(
        values(RectifiedFlow::interpolate(data.clone(), noise, 1.0))
            .iter()
            .all(|value| *value == 1.0)
    );
    let noise = Tensor::<4>::from_data(TensorData::new(vec![-1.0f32; 12], [1, 3, 2, 2]), &device);
    assert!(
        values(RectifiedFlow::interpolate(data, noise, 0.5))
            .iter()
            .all(|value| value.abs() < 1e-6)
    );
}

#[test]
fn the_flow_loss_is_finite_and_non_negative() {
    let device = Device::flex();
    let flow = RectifiedFlow::new(
        Unet::new(
            &UnetConfig {
                dim: 8,
                dim_mults: vec![1, 2],
                ..UnetConfig::default()
            },
            &device,
        ),
        16,
    );
    let data = Tensor::<4>::random([2, 3, 16, 16], Distribution::Normal(0.0, 1.0), &device);
    let noise = Tensor::<4>::random([2, 3, 16, 16], Distribution::Normal(0.0, 1.0), &device);

    let loss = flow.p_losses(data, &[0.1, 0.9], noise, &device, None);
    let value = loss.into_scalar::<f32>();
    assert!(value.is_finite(), "loss was {value}");
    assert!(value >= 0.0, "loss was {value}");
}

#[test]
fn sampling_is_deterministic_when_the_device_is_seeded() {
    let device = Device::flex();
    let flow = RectifiedFlow::new(
        Unet::new(
            &UnetConfig {
                dim: 8,
                dim_mults: vec![1, 2],
                ..UnetConfig::default()
            },
            &device,
        ),
        16,
    );

    device.seed(7);
    let first = flow.sample(2, 4, &device);
    device.seed(7);
    let second = flow.sample(2, 4, &device);

    assert_eq!(first.dims(), [2, 3, 16, 16]);
    assert_eq!(values(first), values(second));
}
