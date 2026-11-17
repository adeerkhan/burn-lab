//! A tiny Genie-style world model over VQ-VAE tokens.
//!
//! Each frame is a 16x16 grid of integer codes. Frames are stacked into a
//! context window; every frame is embedded as code + action that produced it,
//! then a stack of spatio-temporal blocks mixes the grid and the time axis:
//!
//! - spatial: all 256 tokens of a frame attend to each other
//! - temporal: each grid cell attends to the same cell in earlier frames, causal
//!
//! A head predicts a distribution over N_CODES for every token of every frame.
//! Training masks a random fraction of tokens (MaskGIT) and only predicts the
//! hidden ones, with some context tokens corrupted so the model learns to cope
//! with its own imperfect rollouts. Positions are learned (the v1 design in the
//! reference); 3D RoPE is deliberately left out of this small port.

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

pub const TOKENS: usize = 256;
pub const ACTIONS: usize = 5;
pub const NO_ACTION: i64 = ACTIONS as i64;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorldConfig {
    pub codes: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub context: usize,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            codes: 64,
            dim: 64,
            layers: 2,
            heads: 4,
            context: 4,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention {
    to_qkv: Linear,
    to_out: Linear,
    #[module(skip)]
    heads: usize,
}

impl Attention {
    fn new(dim: usize, heads: usize, device: &Device) -> Self {
        Self {
            to_qkv: LinearConfig::new(dim, dim * 3).init(device),
            to_out: LinearConfig::new(dim, dim).init(device),
            heads,
        }
    }

    fn forward(&self, x: Tensor<3>, causal: bool) -> Tensor<3> {
        let [batch, tokens, dim] = x.dims();
        let head_dim = dim / self.heads;
        let qkv = self
            .to_qkv
            .forward(x.clone())
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
        let weights = if causal {
            let mask = Tensor::<2>::ones([tokens, tokens], &x.device())
                .tril(0)
                .reshape([1, 1, tokens, tokens]);
            activation::softmax(scores.mul(mask).div_scalar(1.0), 3)
        } else {
            activation::softmax(scores, 3)
        };
        let out = weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, dim]);
        self.to_out.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: Attention,
    mlp1: Linear,
    mlp2: Linear,
}

impl Block {
    fn new(dim: usize, heads: usize, device: &Device) -> Self {
        Self {
            norm1: LayerNormConfig::new(dim).init(device),
            norm2: LayerNormConfig::new(dim).init(device),
            attention: Attention::new(dim, heads, device),
            mlp1: LinearConfig::new(dim, dim * 4).init(device),
            mlp2: LinearConfig::new(dim * 4, dim).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct WorldModel {
    code: Embedding,
    action: Embedding,
    position: Param<Tensor<2>>,
    frame: Param<Tensor<2>>,
    blocks: Vec<(Block, Block)>,
    norm: LayerNorm,
    head: Linear,
    #[module(skip)]
    config: WorldConfig,
}

impl WorldModel {
    pub fn new(config: &WorldConfig, device: &Device) -> Self {
        let dim = config.dim;
        let codes = config.codes;
        let blocks = (0..config.layers)
            .map(|_| {
                (
                    Block::new(dim, config.heads, device),
                    Block::new(dim, config.heads, device),
                )
            })
            .collect();
        Self {
            code: EmbeddingConfig::new(codes + 1, dim).init(device),
            action: EmbeddingConfig::new(ACTIONS + 1, dim).init(device),
            position: Param::from_tensor(Tensor::<2>::random(
                [TOKENS, dim],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            )),
            frame: Param::from_tensor(Tensor::<2>::random(
                [config.context + 1, dim],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks,
            norm: LayerNormConfig::new(dim).init(device),
            head: LinearConfig::new(dim, codes).init(device),
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// Logits for every token of every frame: `[B, frames, 256, CODES]`.
    pub fn forward_seq(&self, seq: Tensor<3, Int>, actions: Tensor<2, Int>) -> Tensor<4> {
        let [batch, frames, tokens] = seq.dims();
        assert_eq!(tokens, TOKENS);
        let code_emb = self.code.forward(seq.reshape([batch * frames, tokens]));
        let mut x = code_emb.reshape([batch, frames, tokens, self.config.dim]);
        x = x
            .add(self.position.val().reshape([1, 1, tokens, self.config.dim]))
            .add(
                self.frame
                    .val()
                    .narrow(0, 0, frames)
                    .reshape([1, frames, 1, self.config.dim]),
            );
        let act = self
            .action
            .forward(actions.clone())
            .unsqueeze::<4>()
            .reshape([batch, frames, 1, self.config.dim]);
        x = x.add(act);

        for (spatial, temporal) in &self.blocks {
            let h =
                spatial
                    .norm1
                    .forward(x.clone().reshape([batch * frames, tokens, self.config.dim]));
            let attended = spatial.attention.forward(h, false);
            let x1 = x
                .clone()
                .add(attended.reshape([batch, frames, tokens, self.config.dim]));
            let h = spatial.norm2.forward(x1.clone().reshape([
                batch * frames,
                tokens,
                self.config.dim,
            ]));
            let h = spatial
                .mlp2
                .forward(activation::gelu(spatial.mlp1.forward(h)));
            let x1 = x1.add(h.reshape([batch, frames, tokens, self.config.dim]));

            // Transpose to [B, tokens, frames, dim] so temporal attention is
            // over frames for each fixed grid cell, causally.
            let xt = x1
                .swap_dims(1, 2)
                .reshape([batch * tokens, frames, self.config.dim]);
            let h = temporal.norm1.forward(xt.clone());
            let attended = temporal.attention.forward(h, true);
            let xt1 = xt.add(attended);
            let h = temporal.norm2.forward(xt1.clone());
            let h = temporal
                .mlp2
                .forward(activation::gelu(temporal.mlp1.forward(h)));
            let xt1 = xt1.add(h);
            x = xt1
                .reshape([batch, tokens, frames, self.config.dim])
                .swap_dims(1, 2);
        }
        self.head
            .forward(
                self.norm
                    .forward(x.reshape([batch * frames, tokens, self.config.dim])),
            )
            .reshape([batch, frames, tokens, self.config.codes])
    }

    /// Predict one frame from a clean context, in a single pass (MaskGIT 1 pass).
    pub fn predict(&self, seq: Tensor<3, Int>, actions: Tensor<2, Int>) -> Tensor<2, Int> {
        let [batch, frames, _] = seq.dims();
        let codes = self.config.codes;
        let mask = codes as i64;
        // Copy the context into the window: Burn has no index-assign, so rebuild
        // with gather-free concatenation on the host for this tiny case.
        let context_values = seq
            .clone()
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap();
        let mut values = Vec::with_capacity(batch * (frames + 1) * TOKENS);
        let target = vec![mask; batch * TOKENS];
        for index in 0..batch {
            values.extend_from_slice(
                &context_values[index * frames * TOKENS..(index + 1) * frames * TOKENS],
            );
            values.extend_from_slice(&target[index * TOKENS..(index + 1) * TOKENS]);
        }
        let full = Tensor::<3, Int>::from_data(
            TensorData::new(values, [batch, frames + 1, TOKENS]),
            &seq.device(),
        );
        let mut acts = actions
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap();
        acts.extend(vec![NO_ACTION; batch]);
        let actions_full =
            Tensor::<2, Int>::from_data(TensorData::new(acts, [batch, frames + 1]), &full.device());
        let logits = self.forward_seq(full, actions_full);
        logits
            .narrow(1, frames, 1)
            .reshape([batch * TOKENS, codes])
            .argmax(1)
            .reshape([batch, TOKENS])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_predict_have_the_right_shapes() {
        let device = Device::flex();
        let config = WorldConfig {
            codes: 16,
            dim: 32,
            layers: 1,
            heads: 4,
            context: 3,
        };
        let model = WorldModel::new(&config, &device);
        let seq = Tensor::<3, Int>::zeros([2, 3, TOKENS], &device);
        let actions = Tensor::<2, Int>::zeros([2, 3], &device);
        let logits = model.forward_seq(seq.clone(), actions.clone());
        assert_eq!(logits.dims(), [2, 3, TOKENS, 16]);
        let predicted = model.predict(seq, actions);
        assert_eq!(predicted.dims(), [2, TOKENS]);
    }
}
