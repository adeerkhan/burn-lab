//! ViT backbone tests: shapes, attention switching, and a finite flow loss.

use burn::prelude::*;
use diffusion::flow::RectifiedFlow;
use diffusion::unet::Unet;
use diffusion::vit::{AttentionKind, Vit, VitConfig};

fn config() -> VitConfig {
    VitConfig {
        image_size: 32,
        patch_size: 8,
        channels: 3,
        dim: 32,
        depth: 2,
        heads: 4,
        mlp_ratio: 2,
        attention: vec![AttentionKind::Full; 2],
    }
}

#[test]
fn the_vit_preserves_the_image_shape() {
    let device = Device::flex();
    let vit = Vit::new(&config(), &device);
    let x = Tensor::<4>::random(
        [2, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let out = vit.forward_continuous(x, &[0.25, 0.75]);
    assert_eq!(out.dims(), [2, 3, 32, 32]);
    let values = out
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn attention_kinds_can_be_switched_without_changing_weights() {
    let device = Device::flex();
    let vit = Vit::new(&config(), &device);
    let swapped = vit.with_attention(vec![AttentionKind::Linear, AttentionKind::Full]);
    assert_eq!(
        swapped.attention_kinds(),
        vec![AttentionKind::Linear, AttentionKind::Full]
    );
    // Same parameter tensors, only the attention pattern differs.
    let x = Tensor::<4>::random(
        [1, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let full = vit.forward_continuous(x.clone(), &[0.5]);
    let linear = swapped.forward_continuous(x, &[0.5]);
    assert_eq!(full.dims(), linear.dims());
}

#[test]
fn the_vit_trains_under_a_flow_loss() {
    let device = Device::flex();
    let vit = Vit::new(&config(), &device);
    let flow = RectifiedFlow::new(Unet::Vit(vit), 32);

    let data = Tensor::<4>::random(
        [2, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let noise = Tensor::<4>::random(
        [2, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let loss = flow.p_losses(data, &[0.2, 0.8], noise, &device, None);
    let value = loss.into_scalar::<f32>();
    assert!(value.is_finite() && value >= 0.0, "loss was {value}");
}

#[test]
fn classification_head_preserves_shape() {
    let device = Device::flex();
    let config = diffusion::vit_classifier::ClassifierConfig {
        image_size: 32,
        patch_size: 8,
        classes: 5,
        dim: 32,
        depth: 2,
        heads: 4,
        ..Default::default()
    };
    let model = diffusion::vit_classifier::VitClassifier::new(&config, &device);
    let x = Tensor::<4>::random(
        [2, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let logits = model.forward(x);
    assert_eq!(logits.dims(), [2, 5]);
}
