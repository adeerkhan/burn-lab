//! The hybrid block and the four-stream hyper-connection.
//!
//! Mirrors `glm53_flash/model.py::HybridBlock` and `::HyperConnection`.

use burn::{
    module::Param,
    nn::{RmsNorm, RmsNormConfig},
    prelude::*,
    tensor::activation,
};

use crate::{
    attention::{LinearAttention, SparseAttention},
    config::ModelConfig,
    moe::SparseMoE,
};

/// A hybrid block: one attention kind, one MoE, two RMSNorms.
#[derive(Module, Debug)]
pub struct HybridBlock {
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attention: Attention,
    moe: SparseMoE,
}

/// Either attention kind, kept in one enum-like field.
#[derive(Module, Debug)]
pub enum Attention {
    Linear(LinearAttention),
    Sparse(SparseAttention),
}

impl Attention {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Attention::Linear(attention) => attention.forward(x),
            Attention::Sparse(attention) => attention.forward(x),
        }
    }
}

impl HybridBlock {
    pub fn new(config: &ModelConfig, sparse: bool, device: &Device) -> Self {
        let attention = if sparse {
            Attention::Sparse(SparseAttention::new(config, device))
        } else {
            Attention::Linear(LinearAttention::new(config, device))
        };

        Self {
            attention_norm: RmsNormConfig::new(config.dim)
                .with_epsilon(1e-6)
                .init(device),
            ffn_norm: RmsNormConfig::new(config.dim)
                .with_epsilon(1e-6)
                .init(device),
            attention,
            moe: SparseMoE::new(config, device),
        }
    }

    /// # Shapes
    ///   - Input `[batch, time, dim]`
    ///   - Output `[batch, time, dim]`
    ///   - Usage `[experts]`
    pub fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<1>) {
        let attended = self
            .attention
            .forward(self.attention_norm.forward(x.clone()));
        let x = x.add(attended);

        let (ffn, usage) = self.moe.forward(self.ffn_norm.forward(x.clone()));

        (x.add(ffn), usage)
    }

    /// Parameters held by the routed experts, excluding the shared expert.
    pub fn routed_expert_params(&self) -> usize {
        self.moe.routed_expert_params()
    }
}

/// Four residual streams with learned input mixing and output routing.
///
/// The reference calls this "mHC" but simplifies the manifold-constrained
/// Sinkhorn formulation to a plain softmax over the streams.
#[derive(Module, Debug)]
pub struct HyperConnection {
    block: HybridBlock,
    input_logits: Param<Tensor<1>>,
    output_logits: Param<Tensor<1>>,
    #[module(skip)]
    streams: usize,
}

impl HyperConnection {
    pub fn new(config: &ModelConfig, sparse: bool, device: &Device) -> Self {
        Self {
            block: HybridBlock::new(config, sparse, device),
            input_logits: Param::from_tensor(linspace(0.1, -0.1, config.streams, device)),
            output_logits: Param::from_tensor(linspace(-0.1, 0.1, config.streams, device)),
            streams: config.streams,
        }
    }

    /// # Shapes
    ///   - Input `[batch, time, streams, dim]`
    ///   - Output `[batch, time, streams, dim]`
    ///   - Usage `[experts]`
    pub fn forward(&self, streams: Tensor<4>) -> (Tensor<4>, Tensor<1>) {
        // Mix the streams into one, weighted by the learned input routing.
        let input_weights = activation::softmax(self.input_logits.val(), 0);
        let input_weights = input_weights.reshape::<4, _>([1, 1, self.streams, 1]);
        let mixed = streams
            .clone()
            .mul(input_weights)
            .sum_dim(2)
            .squeeze_dim::<3>(2);

        let (transformed, usage) = self.block.forward(mixed);

        // Scatter the block output back into every stream, weighted by the
        // learned output routing.
        let output_weights = activation::softmax(self.output_logits.val(), 0);
        let output_weights = output_weights.reshape::<4, _>([1, 1, self.streams, 1]);
        let routed = transformed.unsqueeze_dim::<4>(2).mul(output_weights);

        (streams.add(routed), usage)
    }

    /// Parameters held by the routed experts, excluding the shared expert.
    pub fn routed_expert_params(&self) -> usize {
        self.block.routed_expert_params()
    }
}

/// `linspace(start, end, count)` over a device.
fn linspace(start: f32, end: f32, count: usize, device: &Device) -> Tensor<1> {
    assert!(count > 1, "linspace needs at least two points");
    let steps = Tensor::<1, Int>::arange(0..count as i64, device).float();
    steps
        .div_scalar((count - 1) as f32)
        .mul_scalar(end - start)
        .add_scalar(start)
}
