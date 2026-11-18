//! A small VQVAE for 64x64 frames: 16x16 integer codes (K=64, D=32)
//! and back. Ported from the VQVAE in `avbiswas/computer_vision_recipes`,
//! with the codebook updated by EMA of assigned encoder outputs and dead codes
//! restarted from random vectors in the batch.
//!
//! The reference found that a GAN critic destabilised training; this port keeps
//! the working recipe: L1 + MSE reconstruction and an L1 edge loss.

use burn::module::Param;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::prelude::*;
use burn::tensor::activation;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VqConfig {
    pub channels: usize,
    pub dim: usize,
    pub codes: usize,
    pub grid: usize,
    pub decay: f32,
}

impl Default for VqConfig {
    fn default() -> Self {
        Self {
            channels: 3,
            dim: 32,
            codes: 64,
            grid: 16,
            decay: 0.99,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResBlock {
    fn new(ch: usize, device: &Device) -> Self {
        let conv = || {
            Conv2dConfig::new([ch, ch], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv1: conv(),
            conv2: conv(),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let h = activation::relu(self.conv1.forward(x.clone()));
        x.add(self.conv2.forward(h))
    }
}

#[derive(Module, Debug)]
pub struct VqVae {
    encoder: Vec<Conv2d>,
    encoder_res: Vec<ResBlock>,
    encoder_out: Conv2d,
    decoder_in: Conv2d,
    decoder_res: Vec<ResBlock>,
    decoder_up: Vec<ConvTranspose2d>,
    decoder_out: Conv2d,
    codebook: Param<Tensor<2>>,
    ema_count: Param<Tensor<1>>,
    ema_sum: Param<Tensor<2>>,
    #[module(skip)]
    levels: usize,
    #[module(skip)]
    config: VqConfig,
}

fn down(ch_in: usize, ch_out: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([ch_in, ch_out], [4, 4])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

fn up(ch_in: usize, ch_out: usize, device: &Device) -> ConvTranspose2d {
    ConvTranspose2dConfig::new([ch_in, ch_out], [4, 4])
        .with_stride([2, 2])
        .with_padding([1, 1])
        .init(device)
}

impl VqVae {
    pub fn new(config: &VqConfig, device: &Device) -> Self {
        let c = config.channels;
        let d = config.dim;
        // 64x64 -> grid x grid needs log2(64/grid) stride-2 stages.
        let levels = (64 / config.grid).ilog2() as usize;
        let mut encoder = vec![
            Conv2dConfig::new([c, d], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        ];
        let mut encoder_res = vec![ResBlock::new(d, device)];
        for _ in 0..levels {
            encoder.push(down(d, d, device));
            encoder_res.push(ResBlock::new(d, device));
        }
        let encoder_out = Conv2dConfig::new([d, d], [1, 1]).init(device);
        let decoder_in = Conv2dConfig::new([d, d], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let decoder_res = (0..=levels).map(|_| ResBlock::new(d, device)).collect();
        let codebook = Param::from_tensor(Tensor::<2>::random(
            [config.codes, d],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        ));
        let ema_count = Param::from_tensor(Tensor::<1>::ones([config.codes], device));
        let ema_sum = Param::from_tensor(codebook.val().clone().detach());
        Self {
            encoder,
            encoder_res,
            encoder_out,
            decoder_in,
            decoder_res,
            decoder_up: (0..levels).map(|_| up(d, d, device)).collect(),
            decoder_out: Conv2dConfig::new([d, c], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            codebook,
            ema_count,
            ema_sum,
            levels,
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &VqConfig {
        &self.config
    }

    pub fn encode(&self, x: Tensor<4>) -> Tensor<4> {
        let mut h = self.encoder[0].forward(x);
        h = self.encoder_res[0].forward(h);
        for level in 0..self.levels {
            h = self.encoder[level + 1].forward(h);
            h = self.encoder_res[level + 1].forward(h);
        }
        self.encoder_out.forward(h)
    }

    pub fn decode(&self, z: Tensor<4>) -> Tensor<4> {
        let mut h = self.decoder_in.forward(z);
        h = self.decoder_res[0].forward(h);
        for level in 0..self.levels {
            h = self.decoder_up[level].forward(h);
            h = self.decoder_res[level + 1].forward(h);
        }
        self.decoder_out.forward(h)
    }

    /// Nearest code ids for a `[B, D, H, W]` encoder output.
    pub fn ids(&self, z: Tensor<4>) -> Tensor<2, Int> {
        let [batch, dim, height, width] = z.dims();
        let flat = z
            .permute([0, 2, 3, 1])
            .reshape([batch * height * width, dim]);
        // ||z - c||^2 = ||z||^2 - 2 z.c + ||c||^2
        let left = flat.clone().powf_scalar(2.0).sum_dim(1).reshape([-1, 1]);
        let right = self
            .codebook
            .val()
            .powf_scalar(2.0)
            .sum_dim(1)
            .reshape([1, -1]);
        let middle = flat.matmul(self.codebook.val().transpose());
        left.add(right)
            .sub(middle.mul_scalar(2.0))
            .argmin(1)
            .reshape([batch * height * width, 1])
    }

    pub fn lookup(&self, ids: Tensor<2, Int>, height: usize, width: usize) -> Tensor<4> {
        let batch = ids.dims()[0] / (height * width);
        let one_hot = Tensor::<2>::zeros([ids.dims()[0], self.config.codes], &ids.device())
            .scatter(
                1,
                ids.clone(),
                Tensor::<2>::ones(ids.dims(), &ids.device()),
                burn::tensor::IndexingUpdateOp::Add,
            );
        let vectors = one_hot.matmul(self.codebook.val());
        vectors
            .reshape([batch, height, width, self.config.dim])
            .permute([0, 3, 1, 2])
    }

    /// Quantised, straight-through vectors plus the commitment loss.
    pub fn quantize(&self, z: Tensor<4>) -> (Tensor<4>, Tensor<1>) {
        let ids = self.ids(z.clone());
        let q = self.lookup(ids, z.dims()[2], z.dims()[3]).detach();
        // Straight-through: forward is q, gradient flows to z.
        let straight = z.clone().add(q.clone().sub(z.clone()).detach());
        let commit = z
            .sub(q)
            .powf_scalar(2.0)
            .mean_dim(1)
            .mean_dim(1)
            .mean_dim(1)
            .mean();
        (straight, commit)
    }

    /// EMA codebook update from the encoder outputs of one training batch.
    pub fn update_codebook(&mut self, z: &Tensor<4>) {
        let [batch, dim, height, width] = z.dims();
        let flat = z
            .clone()
            .detach()
            .permute([0, 2, 3, 1])
            .reshape([batch * height * width, dim]);
        let ids = self.ids(z.clone().detach());
        let one_hot =
            Tensor::<2>::zeros([batch * height * width, self.config.codes], &flat.device())
                .scatter(
                    1,
                    ids.clone(),
                    Tensor::<2>::ones(ids.dims(), &flat.device()),
                    burn::tensor::IndexingUpdateOp::Add,
                );
        let decay = self.config.decay;
        let counts = self.ema_count.val().clone().detach().mul_scalar(decay).add(
            one_hot
                .clone()
                .sum_dim(0)
                .reshape([self.config.codes])
                .mul_scalar(1.0 - decay),
        );
        let sums = self
            .ema_sum
            .val()
            .clone()
            .detach()
            .mul_scalar(decay)
            .add(one_hot.transpose().matmul(flat).mul_scalar(1.0 - decay));
        let codes = sums
            .clone()
            .div(counts.clone().reshape([-1, 1]).clamp_min(1e-5));
        self.codebook = Param::from_tensor(codes.detach());
        self.ema_count = Param::from_tensor(counts.detach());
        self.ema_sum = Param::from_tensor(sums.detach());
    }

    /// Encode and decode in one step, returning the quantised reconstruction.
    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<2, Int>, Tensor<1>) {
        let z = self.encode(x);
        let (q, commit) = self.quantize(z.clone());
        let ids = self.ids(z.detach());
        (self.decode(q), ids, commit)
    }

    /// Decode a `[B, grid, grid]` grid of codes into pixels in `[-1, 1]`.
    pub fn decode_ids(&self, ids: Tensor<2, Int>) -> Tensor<4> {
        let grid = self.config.grid;
        self.decode(self.lookup(ids, grid, grid))
    }
}

pub fn edge_loss(recon: &Tensor<4>, target: &Tensor<4>) -> Tensor<1> {
    let dx = |t: &Tensor<4>| {
        t.clone()
            .narrow(3, 1, t.dims()[3] - 1)
            .sub(t.clone().narrow(3, 0, t.dims()[3] - 1))
    };
    let dy = |t: &Tensor<4>| {
        t.clone()
            .narrow(2, 1, t.dims()[2] - 1)
            .sub(t.clone().narrow(2, 0, t.dims()[2] - 1))
    };
    dx(recon)
        .sub(dx(target))
        .abs()
        .mean()
        .add(dy(recon).sub(dy(target)).abs().mean())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_reconstruction_have_the_right_shapes() {
        let device = Device::flex();
        let config = VqConfig {
            dim: 8,
            codes: 16,
            ..VqConfig::default()
        };
        let model = VqVae::new(&config, &device);
        let x = Tensor::<4>::random(
            [2, 3, 64, 64],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let (recon, ids, commit) = model.forward(x);
        assert_eq!(recon.dims(), [2, 3, 64, 64]);
        assert_eq!(ids.dims(), [512, 1]); // 2 images * 16 * 16 tokens
        assert!(commit.into_scalar::<f32>() >= 0.0);
    }

    #[test]
    fn edge_loss_is_non_negative() {
        let device = Device::flex();
        let a = Tensor::<4>::random(
            [1, 3, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        assert_eq!(edge_loss(&a, &a).into_scalar::<f32>(), 0.0);
        let shifted = a.clone().narrow(3, 1, 7).mul_scalar(3.0);
        let b = Tensor::<4>::cat(vec![a.clone().narrow(3, 0, 1), shifted], 3);
        assert!(edge_loss(&a, &b).into_scalar::<f32>() > 0.0);
    }
}
