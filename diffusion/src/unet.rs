//! The UNet from `denoising_diffusion_pytorch`, module for module.
//!
//! Differences from the reference are listed in `README.md`; the important ones
//! are that RMSNorm is implemented directly (Burn has no channel-wise
//! `F.normalize`) and that the pixel-unshuffle downsample is a reshape and
//! permute rather than a `Rearrange`.

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;

/// `F.normalize(x, dim = 1) * g * sqrt(dim)` — normalises across channels at
/// every pixel, which is not the same as Burn's `RmsNorm` over the last axis.
#[derive(Module, Debug)]
pub struct RmsNorm {
    g: Param<Tensor<1>>,
    dim: usize,
    #[module(skip)]
    scale: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, device: &Device) -> Self {
        Self {
            g: Param::from_tensor(Tensor::ones([dim], device)),
            dim,
            scale: (dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        let normalised = x.div(norm);
        let g = self.g.val().reshape([1, self.dim, 1, 1]);
        normalised.mul(g).mul_scalar(self.scale)
    }
}

/// Sinusoidal timestep embedding, computed on the host then uploaded.
#[derive(Clone, Debug)]
pub struct SinusoidalPosEmb {
    dim: usize,
    theta: f64,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize, theta: f64) -> Self {
        Self { dim, theta }
    }

    /// `[batch, dim]` for the given integer timesteps.
    pub fn forward(&self, times: &[i64], device: &Device) -> Tensor<2> {
        let half = self.dim / 2;
        let log_theta = self.theta.ln();
        let frequencies: Vec<f64> = (0..half)
            .map(|index| (-(index as f64) * log_theta / (half as f64 - 1.0)).exp())
            .collect();

        let batch = times.len();
        let mut data = Vec::with_capacity(batch * self.dim);
        for time in times {
            let values: Vec<f64> = frequencies.iter().map(|f| *time as f64 * f).collect();
            for value in &values {
                data.push(value.sin() as f32);
            }
            for value in &values {
                data.push(value.cos() as f32);
            }
        }
        Tensor::<2>::from_data(TensorData::new(data, [batch, self.dim]), device)
    }
}

#[derive(Module, Debug)]
pub struct Block {
    proj: Conv2d,
    norm: RmsNorm,
    dropout: Dropout,
}

impl Block {
    pub fn new(dim: usize, dim_out: usize, dropout: f64, device: &Device) -> Self {
        Self {
            proj: Conv2dConfig::new([dim, dim_out], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm: RmsNorm::new(dim_out, device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<4>, scale_shift: Option<(Tensor<4>, Tensor<4>)>) -> Tensor<4> {
        let x = self.proj.forward(x);
        let x = self.norm.forward(x);
        let x = match scale_shift {
            Some((scale, shift)) => x.clone().mul(scale.add_scalar(1.0)).add(shift),
            None => x,
        };
        let x = burn::tensor::activation::silu(x);
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ResnetBlock {
    mlp: Option<Linear>,
    block1: Block,
    block2: Block,
    res_conv: Option<Conv2d>,
    dim_out: usize,
}

impl ResnetBlock {
    pub fn new(
        dim: usize,
        dim_out: usize,
        time_emb_dim: Option<usize>,
        dropout: f64,
        device: &Device,
    ) -> Self {
        let mlp =
            time_emb_dim.map(|time_dim| LinearConfig::new(time_dim, dim_out * 2).init(device));
        let res_conv = if dim != dim_out {
            Some(Conv2dConfig::new([dim, dim_out], [1, 1]).init(device))
        } else {
            None
        };
        Self {
            mlp,
            block1: Block::new(dim, dim_out, dropout, device),
            block2: Block::new(dim_out, dim_out, 0.0, device),
            res_conv,
            dim_out,
        }
    }

    pub fn forward(&self, x: Tensor<4>, time_emb: Option<Tensor<2>>) -> Tensor<4> {
        let scale_shift = match (&self.mlp, time_emb) {
            (Some(mlp), Some(embedding)) => {
                let projected = mlp.forward(embedding);
                let [batch, _] = projected.dims();
                let projected = projected.reshape([batch, self.dim_out * 2, 1, 1]);
                let scale = projected
                    .clone()
                    .slice([0..batch, 0..self.dim_out, 0..1, 0..1]);
                let shift = projected.slice([0..batch, self.dim_out..self.dim_out * 2, 0..1, 0..1]);
                Some((scale, shift))
            }
            _ => None,
        };

        let h = self.block1.forward(x.clone(), scale_shift);
        let h = self.block2.forward(h, None);
        let residual = match &self.res_conv {
            Some(conv) => conv.forward(x),
            None => x,
        };
        h.add(residual)
    }
}

/// Linear attention over the flattened spatial grid, with learnable memory keys
/// and values.
#[derive(Module, Debug)]
pub struct LinearAttention {
    norm: RmsNorm,
    mem_kv: Param<Tensor<4>>,
    to_qkv: Conv2d,
    to_out: Conv2d,
    out_norm: RmsNorm,
    heads: usize,
    dim_head: usize,
    dim: usize,
}

impl LinearAttention {
    pub fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        num_mem_kv: usize,
        device: &Device,
    ) -> Self {
        let hidden = heads * dim_head;
        // Deterministic memory: the reference uses `randn`, but the values are
        // learnable, so a small fixed ramp gives the same function class.
        let count = 2 * heads * dim_head * num_mem_kv;
        let values: Vec<f32> = (0..count)
            .map(|index| ((index * 37 % 101) as f32 / 101.0 - 0.5) * 0.04)
            .collect();
        Self {
            norm: RmsNorm::new(dim, device),
            mem_kv: Param::from_tensor(Tensor::from_data(
                TensorData::new(values, [2, heads, dim_head, num_mem_kv]),
                device,
            )),
            to_qkv: Conv2dConfig::new([dim, hidden * 3], [1, 1])
                .with_bias(false)
                .init(device),
            to_out: Conv2dConfig::new([hidden, dim], [1, 1]).init(device),
            out_norm: RmsNorm::new(dim, device),
            heads,
            dim_head,
            dim,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, _, height, width] = x.dims();
        let x = self.norm.forward(x);
        let qkv = self.to_qkv.forward(x);
        let hidden = self.heads * self.dim_head;

        let split = |offset: usize| {
            qkv.clone()
                .slice([0..batch, offset..offset + hidden, 0..height, 0..width])
        };
        // `b (h c) x y -> b h c (x y)`
        let to_heads = |t: Tensor<4>| t.reshape([batch, self.heads, self.dim_head, height * width]);
        let q = to_heads(split(0));
        let k = to_heads(split(hidden));
        let v = to_heads(split(hidden * 2));

        let [_, _, _, memory] = self.mem_kv.val().dims();
        let mem = self.mem_kv.val();
        let mk = mem
            .clone()
            .slice([0..1, 0..self.heads, 0..self.dim_head, 0..memory])
            .repeat_dim(0, batch);
        let mv = mem
            .slice([1..2, 0..self.heads, 0..self.dim_head, 0..memory])
            .repeat_dim(0, batch);

        let k = Tensor::cat(vec![mk, k], 3);
        let v = Tensor::cat(vec![mv, v], 3);

        let q = burn::tensor::activation::softmax(q, 2);
        let k = burn::tensor::activation::softmax(k, 3);
        let scale = (self.dim_head as f32).powf(-0.5);
        let q = q.mul_scalar(scale);

        // context = einsum('b h d n, b h e n -> b h d e', k, v)
        let context = k.matmul(v.transpose());
        // out = einsum('b h d e, b h d n -> b h e n', context, q)
        let out = context.transpose().matmul(q);

        let out = out.reshape([batch, self.heads * self.dim_head, height, width]);
        let out = self.to_out.forward(out);
        self.out_norm.forward(out)
    }
}

/// Full attention over the flattened spatial grid, with memory keys and values.
#[derive(Module, Debug)]
pub struct Attention {
    norm: RmsNorm,
    mem_kv: Param<Tensor<4>>,
    to_qkv: Conv2d,
    to_out: Conv2d,
    heads: usize,
    dim_head: usize,
    dim: usize,
}

impl Attention {
    pub fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        num_mem_kv: usize,
        device: &Device,
    ) -> Self {
        let hidden = heads * dim_head;
        let count = 2 * heads * num_mem_kv * dim_head;
        let values: Vec<f32> = (0..count)
            .map(|index| ((index * 53 % 97) as f32 / 97.0 - 0.5) * 0.04)
            .collect();
        Self {
            norm: RmsNorm::new(dim, device),
            mem_kv: Param::from_tensor(Tensor::from_data(
                TensorData::new(values, [2, heads, num_mem_kv, dim_head]),
                device,
            )),
            to_qkv: Conv2dConfig::new([dim, hidden * 3], [1, 1])
                .with_bias(false)
                .init(device),
            to_out: Conv2dConfig::new([hidden, dim], [1, 1]).init(device),
            heads,
            dim_head,
            dim,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, _, height, width] = x.dims();
        let x = self.norm.forward(x);
        let qkv = self.to_qkv.forward(x);
        let hidden = self.heads * self.dim_head;

        let split = |offset: usize| {
            qkv.clone()
                .slice([0..batch, offset..offset + hidden, 0..height, 0..width])
        };
        // `b (h c) x y -> b h (x y) c`
        let to_heads = |t: Tensor<4>| {
            t.reshape([batch, self.heads, self.dim_head, height * width])
                .permute([0, 1, 3, 2])
        };
        let q = to_heads(split(0));
        let k = to_heads(split(hidden));
        let v = to_heads(split(hidden * 2));

        let [_, _, memory, _] = self.mem_kv.val().dims();
        let mem = self.mem_kv.val();
        let mk = mem
            .clone()
            .slice([0..1, 0..self.heads, 0..memory, 0..self.dim_head])
            .repeat_dim(0, batch);
        let mv = mem
            .slice([1..2, 0..self.heads, 0..memory, 0..self.dim_head])
            .repeat_dim(0, batch);
        let k = Tensor::cat(vec![mk, k], 2);
        let v = Tensor::cat(vec![mv, v], 2);

        let scores = q
            .matmul(k.transpose())
            .mul_scalar((self.dim_head as f32).powf(-0.5));
        let weights = burn::tensor::activation::softmax(scores, 3);
        let out = weights.matmul(v);

        // `b h (x y) c -> b (h d) x y`
        let out =
            out.permute([0, 1, 3, 2])
                .reshape([batch, self.heads * self.dim_head, height, width]);
        self.to_out.forward(out)
    }
}

/// `Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w')` then a 1x1 convolution.
#[derive(Module, Debug)]
pub struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    pub fn new(dim: usize, dim_out: usize, device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([dim * 4, dim_out], [1, 1]).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, channels, height, width] = x.dims();
        let x = x
            .reshape([batch, channels, height / 2, 2, width / 2, 2])
            .permute([0, 1, 3, 5, 2, 4])
            .reshape([batch, channels * 4, height / 2, width / 2]);
        self.conv.forward(x)
    }
}

/// Nearest-neighbour x2 then a 3x3 convolution.
#[derive(Module, Debug)]
pub struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    pub fn new(dim: usize, dim_out: usize, device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([dim, dim_out], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [batch, channels, height, width] = x.dims();
        let x = burn::tensor::module::interpolate(
            x,
            burn::tensor::ops::InterpolateOptions::new(burn::tensor::ops::InterpolateMode::Nearest)
                .with_output_size([height * 2, width * 2]),
        );
        let _ = (batch, channels);
        self.conv.forward(x)
    }
}

/// The reference's `Unet`.
#[derive(Module, Debug)]
pub struct Unet {
    init_conv: Conv2d,
    time_mlp_linear1: Linear,
    time_mlp_linear2: Linear,
    #[module(skip)]
    position: SinusoidalPosEmb,
    #[module(skip)]
    time_dim: usize,
    downs: Vec<DownStage>,
    mid_block1: ResnetBlock,
    mid_attn: Attention,
    mid_block2: ResnetBlock,
    ups: Vec<UpStage>,
    final_res_block: ResnetBlock,
    final_conv: Conv2d,
    channels: usize,
}

#[derive(Module, Debug)]
pub struct DownStage {
    block1: ResnetBlock,
    block2: ResnetBlock,
    attention: StageAttention,
    downsample: StageDown,
}

#[derive(Module, Debug)]
pub enum StageAttention {
    Linear(LinearAttention),
    Full(Attention),
}

impl StageAttention {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Linear(attention) => attention.forward(x),
            Self::Full(attention) => attention.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub enum StageDown {
    Downsample(Downsample),
    Conv(Conv2d),
}

impl StageDown {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Downsample(down) => down.forward(x),
            Self::Conv(conv) => conv.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct UpStage {
    block1: ResnetBlock,
    block2: ResnetBlock,
    attention: StageAttention,
    upsample: StageUp,
}

#[derive(Module, Debug)]
pub enum StageUp {
    Upsample(Upsample),
    Conv(Conv2d),
}

impl StageUp {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Upsample(up) => up.forward(x),
            Self::Conv(conv) => conv.forward(x),
        }
    }
}

/// Configuration mirroring the reference's constructor defaults.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    pub dim: usize,
    pub dim_mults: Vec<usize>,
    pub channels: usize,
    pub dropout: f64,
    pub attn_dim_head: usize,
    pub attn_heads: usize,
    pub full_attn: Vec<bool>,
    pub num_mem_kv: usize,
}

impl Default for UnetConfig {
    fn default() -> Self {
        Self {
            dim: 32,
            dim_mults: vec![1, 2, 4, 8],
            channels: 3,
            dropout: 0.0,
            attn_dim_head: 32,
            attn_heads: 4,
            full_attn: vec![false, false, false, true],
            num_mem_kv: 4,
        }
    }
}

impl Unet {
    pub fn new(config: &UnetConfig, device: &Device) -> Self {
        let input_channels = config.channels;
        let init_dim = config.dim;
        let init_conv = Conv2dConfig::new([input_channels, init_dim], [7, 7])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .init(device);

        let dims: Vec<usize> = core::iter::once(init_dim)
            .chain(config.dim_mults.iter().map(|mult| config.dim * mult))
            .collect();
        let in_out: Vec<(usize, usize)> = dims.windows(2).map(|pair| (pair[0], pair[1])).collect();
        let time_dim = config.dim * 4;

        let time_mlp_linear1 = LinearConfig::new(config.dim, time_dim).init(device);
        let time_mlp_linear2 = LinearConfig::new(time_dim, time_dim).init(device);

        let mut downs = Vec::new();
        let resolutions = in_out.len();
        for (index, (dim_in, dim_out)) in in_out.iter().enumerate() {
            let is_last = index >= resolutions - 1;
            let attention = if config.full_attn[index] {
                StageAttention::Full(Attention::new(
                    *dim_in,
                    config.attn_heads,
                    config.attn_dim_head,
                    config.num_mem_kv,
                    device,
                ))
            } else {
                StageAttention::Linear(LinearAttention::new(
                    *dim_in,
                    config.attn_heads,
                    config.attn_dim_head,
                    config.num_mem_kv,
                    device,
                ))
            };
            let downsample = if is_last {
                StageDown::Conv(
                    Conv2dConfig::new([*dim_in, *dim_out], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                        .init(device),
                )
            } else {
                StageDown::Downsample(Downsample::new(*dim_in, *dim_out, device))
            };
            downs.push(DownStage {
                block1: ResnetBlock::new(*dim_in, *dim_in, Some(time_dim), config.dropout, device),
                block2: ResnetBlock::new(*dim_in, *dim_in, Some(time_dim), config.dropout, device),
                attention,
                downsample,
            });
        }

        let mid_dim = *dims.last().expect("at least one stage");
        let mid_block1 = ResnetBlock::new(mid_dim, mid_dim, Some(time_dim), config.dropout, device);
        let mid_attn = Attention::new(
            mid_dim,
            config.attn_heads,
            config.attn_dim_head,
            config.num_mem_kv,
            device,
        );
        let mid_block2 = ResnetBlock::new(mid_dim, mid_dim, Some(time_dim), config.dropout, device);

        let mut ups = Vec::new();
        for (index, (dim_in, dim_out)) in in_out.iter().rev().enumerate() {
            let is_last = index == resolutions - 1;
            let attention = if config.full_attn[resolutions - 1 - index] {
                StageAttention::Full(Attention::new(
                    *dim_out,
                    config.attn_heads,
                    config.attn_dim_head,
                    config.num_mem_kv,
                    device,
                ))
            } else {
                StageAttention::Linear(LinearAttention::new(
                    *dim_out,
                    config.attn_heads,
                    config.attn_dim_head,
                    config.num_mem_kv,
                    device,
                ))
            };
            let upsample = if is_last {
                StageUp::Conv(
                    Conv2dConfig::new([*dim_out, *dim_in], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                        .init(device),
                )
            } else {
                StageUp::Upsample(Upsample::new(*dim_out, *dim_in, device))
            };
            ups.push(UpStage {
                block1: ResnetBlock::new(
                    *dim_out + *dim_in,
                    *dim_out,
                    Some(time_dim),
                    config.dropout,
                    device,
                ),
                block2: ResnetBlock::new(
                    *dim_out + *dim_in,
                    *dim_out,
                    Some(time_dim),
                    config.dropout,
                    device,
                ),
                attention,
                upsample,
            });
        }

        let final_res_block = ResnetBlock::new(
            init_dim * 2,
            init_dim,
            Some(time_dim),
            config.dropout,
            device,
        );
        let final_conv = Conv2dConfig::new([init_dim, config.channels], [1, 1]).init(device);

        Self {
            init_conv,
            time_mlp_linear1,
            time_mlp_linear2,
            position: SinusoidalPosEmb::new(config.dim, 10_000.0),
            downs,
            mid_block1,
            mid_attn,
            mid_block2,
            ups,
            final_res_block,
            final_conv,
            channels: config.channels,
            time_dim,
        }
    }

    /// Spatial reduction factor, `2 ** (len(downs) - 1)`.
    pub fn downsample_factor(&self) -> usize {
        1 << (self.downs.len() - 1)
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn forward(&self, x: Tensor<4>, times: &[i64]) -> Tensor<4> {
        let position = self.position.forward(times, &x.device());
        let embedding = burn::tensor::activation::gelu(self.time_mlp_linear1.forward(position));
        let embedding = self.time_mlp_linear2.forward(embedding);

        let mut x = self.init_conv.forward(x);
        let residual = x.clone();
        let mut skip: Vec<Tensor<4>> = Vec::new();

        for stage in &self.downs {
            x = stage.block1.forward(x, Some(embedding.clone()));
            skip.push(x.clone());
            x = stage.block2.forward(x, Some(embedding.clone()));
            x = stage.attention.forward(x.clone()).add(x);
            skip.push(x.clone());
            x = stage.downsample.forward(x);
        }

        x = self.mid_block1.forward(x, Some(embedding.clone()));
        x = self.mid_attn.forward(x.clone()).add(x);
        x = self.mid_block2.forward(x, Some(embedding.clone()));

        for stage in &self.ups {
            x = Tensor::cat(vec![x, skip.pop().expect("skip connection")], 1);
            x = stage.block1.forward(x, Some(embedding.clone()));
            x = Tensor::cat(vec![x, skip.pop().expect("skip connection")], 1);
            x = stage.block2.forward(x, Some(embedding.clone()));
            x = stage.attention.forward(x.clone()).add(x);
            x = stage.upsample.forward(x);
        }

        x = Tensor::cat(vec![x, residual], 1);
        let x = self.final_res_block.forward(x, Some(embedding));
        self.final_conv.forward(x)
    }
}
