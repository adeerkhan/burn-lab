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

fn default_tokens() -> usize {
    TOKENS
}

fn default_change_prior() -> f32 {
    -2.0
}
pub const ACTIONS: usize = 5;
pub const NO_ACTION: i64 = ACTIONS as i64;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorldConfig {
    pub codes: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub context: usize,
    /// Tokens per frame: grid x grid (256 for 16x16, 1024 for 32x32).
    #[serde(default = "default_tokens")]
    pub tokens: usize,
    /// Initial logit bias against changing a token. Negative = copy.
    #[serde(default = "default_change_prior")]
    pub change_prior: f32,
    /// Predict a change flag plus a new code, falling back to copying the
    /// previous frame's code. Much stronger on a slowly changing room.
    #[serde(default)]
    pub residual: bool,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            codes: 64,
            dim: 64,
            layers: 2,
            heads: 4,
            context: 4,
            tokens: default_tokens(),
            change_prior: default_change_prior(),
            residual: false,
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
    change_head: Linear,
    change_prior: Param<Tensor<1>>,
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
                [config.tokens, dim],
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
            // Bias toward "no change": at init the model copies the last frame
            // unless it learns evidence to the contrary. This makes the copy
            // baseline the starting point rather than something to discover.
            change_head: LinearConfig::new(dim, 1).init(device),
            change_prior: Param::from_tensor(Tensor::<1>::full([1], config.change_prior, device)),
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// Code logits `[B, frames, 256, codes]` and change logits
    /// `[B, frames, 256, 1]` for every token of every frame.
    pub fn forward_full(
        &self,
        seq: Tensor<3, Int>,
        actions: Tensor<2, Int>,
    ) -> (Tensor<4>, Tensor<4>) {
        let [batch, frames, tokens] = seq.dims();
        assert_eq!(tokens, self.config.tokens);
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
        let h = self
            .norm
            .forward(x.reshape([batch * frames, tokens, self.config.dim]));
        let logits =
            self.head
                .forward(h.clone())
                .reshape([batch, frames, tokens, self.config.codes]);
        let change = self
            .change_head
            .forward(h)
            .add(self.change_prior.val().reshape([1, 1, 1]))
            .reshape([batch, frames, tokens, 1]);
        (logits, change)
    }

    /// Absolute-code logits only, for the non-residual path and the shape test.
    pub fn forward_seq(&self, seq: Tensor<3, Int>, actions: Tensor<2, Int>) -> Tensor<4> {
        self.forward_full(seq, actions).0
    }

    /// One decoding pass with a given (possibly partially revealed) target
    /// frame. Returns new codes, change probabilities and token confidences.
    fn predict_pass(
        &self,
        seq: &Tensor<3, Int>,
        actions: &Tensor<2, Int>,
        target: Option<&Tensor<2, Int>>,
    ) -> (Tensor<2, Int>, Tensor<2>, Tensor<2>) {
        let device = seq.device();
        let [batch, frames, _] = seq.dims();
        let tokens = self.config.tokens;
        let codes = self.config.codes;
        let mask = codes as i64;
        let context_values = seq
            .clone()
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap();
        let target_values = match target {
            Some(t) => t
                .clone()
                .into_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap(),
            None => vec![mask; batch * tokens],
        };
        let mut values = Vec::with_capacity(batch * (frames + 1) * tokens);
        for index in 0..batch {
            values.extend_from_slice(
                &context_values[index * frames * tokens..(index + 1) * frames * tokens],
            );
            values.extend_from_slice(&target_values[index * tokens..(index + 1) * tokens]);
        }
        let full = Tensor::<3, Int>::from_data(
            TensorData::new(values, [batch, frames + 1, tokens]),
            &device,
        );
        // The oldest context frame has no known action; every later frame,
        // including the target, is preceded by the action that produced it.
        let mut acts = vec![NO_ACTION; batch];
        acts.extend(
            actions
                .clone()
                .into_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap(),
        );
        let actions_full =
            Tensor::<2, Int>::from_data(TensorData::new(acts, [batch, frames + 1]), &device);
        let (logits, change_logits) = self.forward_full(full, actions_full);
        let target_logits = logits.narrow(1, frames, 1).reshape([batch * tokens, codes]);
        let probabilities = activation::softmax(target_logits.clone(), 1);
        let new_codes = target_logits.argmax(1).reshape([batch * tokens]);
        let confidence = probabilities
            .gather(1, new_codes.clone().reshape([-1, 1]))
            .reshape([batch * tokens]);
        let change = change_logits
            .narrow(1, frames, 1)
            .reshape([batch * tokens])
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let change_probs: Vec<f32> = change.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        (
            new_codes.reshape([batch, tokens]),
            Tensor::<2>::from_data(TensorData::new(change_probs, [batch, tokens]), &device),
            confidence.reshape([batch, tokens]),
        )
    }

    /// Predict the next frame's codes and a per-token probability that each
    /// token changed relative to the last context frame.
    pub fn predict_detail(
        &self,
        seq: Tensor<3, Int>,
        actions: Tensor<2, Int>,
    ) -> (Tensor<2, Int>, Tensor<2>) {
        let (codes, probs, _) = self.predict_pass(&seq, &actions, None);
        (codes, probs)
    }

    /// Iterative MaskGIT decoding. Start with every target token masked, then
    /// on each pass commit the most confident still-masked tokens and re-run.
    /// One pass reduces to plain argmax; more passes sharpen stochastic
    /// fireflies and beam edges without retraining.
    pub fn predict_multipass(
        &self,
        seq: Tensor<3, Int>,
        actions: Tensor<2, Int>,
        passes: usize,
    ) -> (Tensor<2, Int>, Tensor<2>) {
        let passes = passes.max(1);
        if passes == 1 {
            return self.predict_detail(seq, actions);
        }
        let device = seq.device();
        let batch = seq.dims()[0];
        let tokens = self.config.tokens;
        let mask = self.config.codes as i64;
        let mut target = vec![mask; batch * tokens];
        let mut codes = vec![0i64; batch * tokens];
        let mut change_probs = vec![0f32; batch * tokens];
        for pass in 1..=passes {
            let target_tensor = Tensor::<2, Int>::from_data(
                TensorData::new(target.clone(), [batch, tokens]),
                &device,
            );
            let (new_codes, probs, confidence) =
                self.predict_pass(&seq, &actions, Some(&target_tensor));
            codes = new_codes
                .into_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap();
            change_probs = probs
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            if pass == passes {
                break;
            }
            // Cosine schedule: the fraction still masked after pass i is
            // cos(i*pi/(2*passes)). Reveal the top-confidence tokens needed to
            // move from this pass's ratio to the next one's.
            let now = (pass as f32 * std::f32::consts::PI / (2.0 * passes as f32))
                .cos()
                .max(0.0);
            let next = ((pass + 1) as f32 * std::f32::consts::PI / (2.0 * passes as f32))
                .cos()
                .max(0.0);
            let fraction = ((now - next) / now.max(1e-6)).min(1.0);
            let conf = confidence
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            for item in 0..batch {
                let mut masked: Vec<(f32, usize)> = (0..tokens)
                    .filter(|&i| target[item * tokens + i] == mask)
                    .map(|i| (conf[item * tokens + i], i))
                    .collect();
                let count = (fraction * masked.len() as f32).ceil() as usize;
                masked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for &(_, index) in masked.iter().take(count) {
                    target[item * tokens + index] = codes[item * tokens + index];
                }
            }
        }
        (
            Tensor::<2, Int>::from_data(TensorData::new(codes, [batch, tokens]), &device),
            Tensor::<2>::from_data(TensorData::new(change_probs, [batch, tokens]), &device),
        )
    }

    pub fn predict(&self, seq: Tensor<3, Int>, actions: Tensor<2, Int>) -> Tensor<2, Int> {
        self.predict_detail(seq, actions).0
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
            tokens: 256,
            change_prior: -2.0,
            residual: false,
        };
        let model = WorldModel::new(&config, &device);
        let tokens = config.tokens;
        let seq = Tensor::<3, Int>::zeros([2, 3, tokens], &device);
        let actions = Tensor::<2, Int>::zeros([2, 3], &device);
        let logits = model.forward_seq(seq.clone(), actions.clone());
        assert_eq!(logits.dims(), [2, 3, tokens, 16]);
        let predicted = model.predict(seq.clone(), actions.clone());
        assert_eq!(predicted.dims(), [2, tokens]);

        let residual = WorldConfig {
            residual: true,
            ..config
        };
        let model = WorldModel::new(&residual, &device);
        let (logits, change) = model.forward_full(seq.clone(), actions.clone());
        assert_eq!(logits.dims(), [2, 3, tokens, 16]);
        assert_eq!(change.dims(), [2, 3, tokens, 1]);
        assert_eq!(model.predict(seq, actions).dims(), [2, tokens]);
    }
}
