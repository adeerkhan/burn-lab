//! A tiny residual detail decoder that sharpens VQ-VAE reconstructions.
//!
//! The tokenizer compresses each 64x64 frame to a 16x16 grid of 4x4-pixel
//! patches, so decoded frames lose edges and small markers. This net learns a
//! correction on top of a decoded frame: `out = clamp(x + residual(x))`, trained
//! with L1 and an edge loss against the real frame while the VQ-VAE stays
//! frozen. It is applied to dream frames at playback time, so the dynamics
//! model is unchanged.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation;

use crate::world_vq::edge_loss;

#[derive(Module, Debug)]
pub struct DetailNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    #[module(skip)]
    channels: usize,
}

fn conv(ch_in: usize, ch_out: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([ch_in, ch_out], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

impl DetailNet {
    pub fn new(channels: usize, device: &Device) -> Self {
        Self {
            conv1: conv(3, channels, device),
            conv2: conv(channels, channels, device),
            conv3: conv(channels, 3, device),
            channels,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let h = activation::relu(self.conv1.forward(x.clone()));
        let h = activation::relu(self.conv2.forward(h));
        let residual = self.conv3.forward(h);
        x.add(residual).clamp(-1.0, 1.0)
    }
}

/// L1 plus the same edge loss the tokenizer uses, so boundaries stay crisp.
pub fn detail_loss(prediction: &Tensor<4>, target: &Tensor<4>) -> Tensor<1> {
    prediction
        .clone()
        .sub(target.clone())
        .abs()
        .mean()
        .add(edge_loss(prediction, target).mul_scalar(0.5))
}

/// Saving/loading helpers keep the checkpoint shape in one place.
pub fn save_detail(model: DetailNet, path: &str) {
    let record = model.valid().into_record();
    record
        .save(format!("{path}/detail"))
        .expect("save detail net");
    std::fs::write(format!("{path}/detail-config.json"), "{\"channels\": 32}")
        .expect("write detail config");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_net_preserves_shape_and_is_bounded() {
        let device = Device::flex();
        let net = DetailNet::new(8, &device);
        let x = Tensor::<4>::random(
            [2, 3, 16, 16],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let out = net.forward(x.clone());
        assert_eq!(out.dims(), [2, 3, 16, 16]);
        let max = out.abs().max().into_scalar::<f32>();
        assert!(max <= 1.0 + 1e-5);
    }
}
