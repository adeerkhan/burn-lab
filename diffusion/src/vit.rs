//! A Vision Transformer for diffusion, ported from
//! [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch), with
//! the post-training attention search idea from *JetViT: Efficient
//! High-Resolution Vision Transformer with Post-Training Attention Search*
//! (CVPR 2026 Findings).
//!
//! The `vit-pytorch` `ViT` is an image *classifier*: patch embedding, a class
//! token, learned positions, a stack of pre-norm transformer blocks, a final
//! norm and an MLP head. Two changes turn it into a generative backbone, both
//! marked in the code:
//!
//! - **Time conditioning.** The reference has none. Every block modulates its
//!   normalised activations with a scale and shift derived from a sinusoidal
//!   time embedding (the DiT recipe); without it the model cannot know where it
//!   is on the flow's path.
//! - **Pixel output.** The classifier head becomes a linear projection back to
//!   `patch * patch * channels` values, unpatchified into an image.
//!
//! # JetViT
//!
//! JetViT starts from a trained full-attention ViT and replaces redundant
//! attention blocks with cheaper ones while *inheriting the weights*, then keeps
//! only the critical full-attention blocks. This module implements the parts
//! that are parameter-compatible:
//!
//! - `AttentionKind::Full` — the reference's softmax attention.
//! - `AttentionKind::Linear` — ReLU linear attention reusing the same
//!   `to_qkv`/`to_out` weights, normalised by the key sum, so a trained block
//!   can be switched without new parameters.
//!
//! Not implemented: the JetViT linear block's squeeze dynamic convolution (it
//! adds parameters, so it cannot inherit a full-attention checkpoint), window
//! attention, and the distillation stage of the search. The search here is a
//! greedy loss-based scan, documented in `scripts/` and the README.

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

use crate::unet::SinusoidalPosEmb;

/// Which attention a block uses. Both variants share `to_qkv` and `to_out`, so
/// a checkpoint trained with `Full` loads into a model with `Linear` blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    Linear,
}

#[derive(Clone, Debug)]
pub struct VitConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    /// Ratio of the MLP hidden width to `dim`, as in the reference's `mlp_dim`.
    pub mlp_ratio: usize,
    /// One attention kind per block; `depth` entries.
    pub attention: Vec<AttentionKind>,
}

impl Default for VitConfig {
    fn default() -> Self {
        Self {
            image_size: 64,
            patch_size: 8,
            channels: 3,
            dim: 256,
            depth: 8,
            heads: 8,
            mlp_ratio: 4,
            attention: vec![AttentionKind::Full; 8],
        }
    }
}

#[derive(Module, Debug)]
pub struct Vit {
    patch_embed: Conv2d,
    position: Param<Tensor<2>>,
    time_mlp1: Linear,
    time_mlp2: Linear,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    #[module(skip)]
    config: VitConfig,
}

#[derive(Module, Debug)]
pub struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Attention,
    mlp1: Linear,
    mlp2: Linear,
    /// `[dim * 6]` from the time embedding: two scale/shift pairs for attention
    /// and MLP.
    adaln: Linear,
    #[module(skip)]
    kind: AttentionKind,
    #[module(skip)]
    heads: usize,
}

#[derive(Module, Debug)]
pub struct Attention {
    to_qkv: Linear,
    to_out: Linear,
}

impl Attention {
    /// Softmax attention, exactly the reference's `Attention`.
    fn full(&self, x: Tensor<3>, heads: usize) -> Tensor<3> {
        let [batch, tokens, dim] = x.dims();
        let head_dim = dim / heads;
        let qkv = self.to_qkv.forward(x);
        let qkv = qkv.reshape([batch, tokens, 3, heads, head_dim]);
        let q = qkv
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch, tokens, heads, head_dim]);
        let k = qkv
            .clone()
            .narrow(2, 1, 1)
            .reshape([batch, tokens, heads, head_dim]);
        let v = qkv
            .narrow(2, 2, 1)
            .reshape([batch, tokens, heads, head_dim]);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scale = (head_dim as f32).powf(-0.5);
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let weights = activation::softmax(scores, 3);
        let out = weights.matmul(v);
        let out = out.swap_dims(1, 2).reshape([batch, tokens, dim]);
        self.to_out.forward(out)
    }

    /// ReLU linear attention with the same weights: `(q k^T) v` is reassociated
    /// as `q (k^T v)`, normalised by the summed keys. O(tokens) instead of
    /// O(tokens^2).
    fn linear(&self, x: Tensor<3>, heads: usize) -> Tensor<3> {
        let [batch, tokens, dim] = x.dims();
        let head_dim = dim / heads;
        let qkv = self.to_qkv.forward(x);
        let qkv = qkv.reshape([batch, tokens, 3, heads, head_dim]);
        let q = qkv
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch, tokens, heads, head_dim]);
        let k = qkv
            .clone()
            .narrow(2, 1, 1)
            .reshape([batch, tokens, heads, head_dim]);
        let v = qkv
            .narrow(2, 2, 1)
            .reshape([batch, tokens, heads, head_dim]);

        let q = activation::relu(q).swap_dims(1, 2);
        let k = activation::relu(k).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // context[b, h, d, e] = sum_n k[b, h, n, d] * v[b, h, n, e]
        let context = k.clone().swap_dims(2, 3).matmul(v);
        let numerator = q.clone().matmul(context);
        // denominator[b, h, d] = sum_n k[b, h, n, d]
        let denominator = k.sum_dim(2).reshape([batch, heads, 1, head_dim]);
        let out = numerator.div(denominator.add_scalar(1e-6));
        let out = out.swap_dims(1, 2).reshape([batch, tokens, dim]);
        self.to_out.forward(out)
    }

    fn forward(&self, x: Tensor<3>, heads: usize, kind: AttentionKind) -> Tensor<3> {
        match kind {
            AttentionKind::Full => self.full(x, heads),
            AttentionKind::Linear => self.linear(x, heads),
        }
    }
}

impl Block {
    fn forward(&self, x: Tensor<3>, time: Tensor<2>) -> Tensor<3> {
        let [batch, _, dim] = x.dims();
        let modulation = self.adaln.forward(time);
        let modulation = modulation.reshape([batch, 6, dim]);

        let shift_attn = modulation.clone().narrow(1, 0, 1).reshape([batch, 1, dim]);
        let scale_attn = modulation.clone().narrow(1, 1, 1).reshape([batch, 1, dim]);
        let gate_attn = modulation.clone().narrow(1, 2, 1).reshape([batch, 1, dim]);
        let shift_mlp = modulation.clone().narrow(1, 3, 1).reshape([batch, 1, dim]);
        let scale_mlp = modulation.clone().narrow(1, 4, 1).reshape([batch, 1, dim]);
        let gate_mlp = modulation.narrow(1, 5, 1).reshape([batch, 1, dim]);

        let h = self
            .norm1
            .forward(x.clone())
            .mul(scale_attn.add_scalar(1.0))
            .add(shift_attn);
        let attention = self.attn.forward(h, self.heads, self.kind);
        let x = x.add(attention.mul(gate_attn));

        let h = self
            .norm2
            .forward(x.clone())
            .mul(scale_mlp.add_scalar(1.0))
            .add(shift_mlp);
        let mlp = self.mlp2.forward(activation::gelu(self.mlp1.forward(h)));
        x.add(mlp.mul(gate_mlp))
    }
}

impl Vit {
    pub fn new(config: &VitConfig, device: &Device) -> Self {
        let patch = config.patch_size;
        let grid = config.image_size / patch;
        let tokens = grid * grid;

        let patch_embed = Conv2dConfig::new([config.channels, config.dim], [patch, patch])
            .with_stride([patch, patch])
            .init(device);
        let position = Param::from_tensor(Tensor::<2>::random(
            [tokens, config.dim],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        ));

        let time_dim = config.dim * 4;
        let time_mlp1 = LinearConfig::new(config.dim, time_dim).init(device);
        let time_mlp2 = LinearConfig::new(time_dim, time_dim).init(device);

        let blocks = (0..config.depth)
            .map(|index| Block {
                norm1: LayerNormConfig::new(config.dim).init(device),
                norm2: LayerNormConfig::new(config.dim).init(device),
                attn: Attention {
                    to_qkv: LinearConfig::new(config.dim, config.dim * 3).init(device),
                    to_out: LinearConfig::new(config.dim, config.dim).init(device),
                },
                mlp1: LinearConfig::new(config.dim, config.dim * config.mlp_ratio).init(device),
                mlp2: LinearConfig::new(config.dim * config.mlp_ratio, config.dim).init(device),
                adaln: LinearConfig::new(time_dim, config.dim * 6).init(device),
                kind: config.attention[index],
                heads: config.heads,
            })
            .collect();

        Self {
            patch_embed,
            position,
            time_mlp1,
            time_mlp2,
            blocks,
            norm: LayerNormConfig::new(config.dim).init(device),
            head: LinearConfig::new(config.dim, patch * patch * config.channels).init(device),
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &VitConfig {
        &self.config
    }

    /// Velocity prediction at continuous time `t` in `[0, 1]`.
    pub fn forward_continuous(&self, x: Tensor<4>, times: &[f32]) -> Tensor<4> {
        let [batch, channels, height, width] = x.dims();
        let patch = self.config.patch_size;
        let grid = height / patch;

        // Patch embedding: conv with stride = kernel, then [b, dim, h, w] ->
        // [b, h*w, dim].
        let x = self.patch_embed.forward(x);
        let tokens = grid * grid;
        let mut x = x.reshape([batch, self.config.dim, tokens]).swap_dims(1, 2);

        // Positions and time conditioning.
        let position = self.position.val().reshape([1, tokens, self.config.dim]);
        x = x.add(position);
        // DiT scales continuous time by 1000 before the sinusoidal embedding;
        // on [0, 1] alone most frequencies are nearly constant.
        let scaled: Vec<f32> = times.iter().map(|time| time * 1000.0).collect();
        let time = SinusoidalPosEmb::new(self.config.dim, 10_000.0)
            .forward_continuous(&scaled, &x.device());
        let time = activation::gelu(self.time_mlp1.forward(time));
        let time = self.time_mlp2.forward(time);

        for block in &self.blocks {
            x = block.forward(x, time.clone());
        }
        let x = self.norm.forward(x);

        // Back to pixels: [b, n, patch*patch*c] -> [b, c, h, w].
        let x = self.head.forward(x);
        // [b, gh, gw, ph, pw, c] -> [b, c, gh, ph, gw, pw] -> [b, c, h, w].
        // Each swap moves one axis into place; mixing grid and patch axes here
        // would permute pixels across tokens, which no per-token head can undo.
        let x = x.reshape([batch, grid, grid, patch, patch, channels]);
        let x = x.swap_dims(1, 5).swap_dims(2, 5).swap_dims(4, 5);
        x.reshape([batch, channels, height, width])
    }

    /// Which attention each block uses, in order.
    pub fn attention_kinds(&self) -> Vec<AttentionKind> {
        self.blocks.iter().map(|block| block.kind).collect()
    }

    /// A copy with a different attention schedule, for the post-training search.
    pub fn with_attention(&self, attention: Vec<AttentionKind>) -> Self {
        let mut config = self.config.clone();
        config.attention = attention;
        let mut copy = self.clone();
        for (block, kind) in copy.blocks.iter_mut().zip(config.attention.clone()) {
            block.kind = kind;
        }
        copy.config = config;
        copy
    }
}
