//! A small Vision Transformer classifier for flowers.
//!
//! Shares the design of `src/vit.rs` — patch embedding, learned positions,
//! pre-norm transformer blocks — but ends in a class head instead of a pixel
//! head. Kept deliberately small (`dim 64`, depth 4, 4 heads at 64x64) so
//! training needs little memory.

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClassifierConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    pub classes: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            image_size: 64,
            patch_size: 8,
            channels: 3,
            classes: 5,
            dim: 64,
            depth: 4,
            heads: 4,
        }
    }
}

#[derive(Module, Debug)]
pub struct VitClassifier {
    patch_embed: Conv2d,
    position: Param<Tensor<2>>,
    norm: LayerNorm,
    blocks: Vec<Block>,
    head: Linear,
    #[module(skip)]
    config: ClassifierConfig,
}

#[derive(Module, Debug)]
pub struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    to_qkv: Linear,
    to_out: Linear,
    mlp1: Linear,
    mlp2: Linear,
    residual_scale: Param<Tensor<1>>,
    #[module(skip)]
    heads: usize,
}

impl Block {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, dim] = x.dims();
        let head_dim = dim / self.heads;

        let h = self.norm1.forward(x.clone());
        let qkv = self
            .to_qkv
            .forward(h)
            .reshape([batch, tokens, 3, self.heads, head_dim]);
        let q = qkv
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch, tokens, self.heads, head_dim])
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .narrow(2, 1, 1)
            .reshape([batch, tokens, self.heads, head_dim])
            .swap_dims(1, 2);
        let v = qkv
            .narrow(2, 2, 1)
            .reshape([batch, tokens, self.heads, head_dim])
            .swap_dims(1, 2);
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((head_dim as f32).powf(-0.5));
        let out = activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, dim]);
        let scale = self
            .residual_scale
            .val()
            .reshape([1, 1, dim])
            .add_scalar(1.0);
        let x = x.add(self.to_out.forward(out).mul(scale.clone()));

        let h = self.norm2.forward(x.clone());
        let h = self.mlp2.forward(activation::gelu(self.mlp1.forward(h)));
        x.add(
            h.mul(
                self.residual_scale
                    .val()
                    .reshape([1, 1, dim])
                    .add_scalar(1.0),
            ),
        )
    }
}

impl VitClassifier {
    pub fn new(config: &ClassifierConfig, device: &Device) -> Self {
        let patch = config.patch_size;
        let grid = config.image_size / patch;
        let tokens = grid * grid;
        // A 1x1-style projection of flattened patches acts as a linear patch
        // embedding; Conv2d with stride=kernel is the same map with less host work.
        let patch_embed = Conv2dConfig::new([config.channels, config.dim], [patch, patch])
            .with_stride([patch, patch])
            .init(device);
        let position = Param::from_tensor(Tensor::<2>::random(
            [tokens, config.dim],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        ));
        let blocks = (0..config.depth)
            .map(|_| Block {
                norm1: LayerNormConfig::new(config.dim).init(device),
                norm2: LayerNormConfig::new(config.dim).init(device),
                to_qkv: LinearConfig::new(config.dim, config.dim * 3).init(device),
                to_out: LinearConfig::new(config.dim, config.dim).init(device),
                mlp1: LinearConfig::new(config.dim, config.dim * 2).init(device),
                mlp2: LinearConfig::new(config.dim * 2, config.dim).init(device),
                residual_scale: Param::from_tensor(Tensor::<1>::from_data(
                    TensorData::new(vec![0.1f32; config.dim], [config.dim]),
                    device,
                )),
                heads: config.heads,
            })
            .collect();
        Self {
            patch_embed,
            position,
            norm: LayerNormConfig::new(config.dim).init(device),
            blocks,
            head: LinearConfig::new(config.dim, config.classes).init(device),
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Class logits for a batch of images.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let [batch, _, height, _] = x.dims();
        let patch = self.config.patch_size;
        let grid = height / patch;
        let tokens = grid * grid;

        let embedded = self.patch_embed.forward(x);
        let mut x = embedded
            .reshape([batch, self.config.dim, tokens])
            .swap_dims(1, 2);
        x = x.add(self.position.val().reshape([1, tokens, self.config.dim]));
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self
            .norm
            .forward(x)
            .mean_dim(1)
            .reshape([batch, self.config.dim]);
        self.head.forward(x)
    }
}
