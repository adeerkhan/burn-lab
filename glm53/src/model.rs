//! The GLM-5.3-Flash-inspired language model.
//!
//! Mirrors `glm53_flash/model.py::GLM53FlashFromScratch`.
//!
//! ```text
//! bytes -> embedding -> four residual streams
//!                       |
//!        [linear attention + MoE] x 3
//!                       |
//!        [sparse attention + MoE] x 1
//!                       |
//!                  repeat x 3
//!                       |
//!              RMSNorm -> tied LM head
//! ```

use burn::{
    nn::{Embedding, EmbeddingConfig, Initializer, RmsNorm, RmsNormConfig},
    prelude::*,
};

use crate::{block::HyperConnection, config::ModelConfig};

/// The language model.
#[derive(Module, Debug)]
pub struct GLM53Flash {
    embedding: Embedding,
    layers: Vec<HyperConnection>,
    final_norm: RmsNorm,
    #[module(skip)]
    config: ModelConfig,
}

impl GLM53Flash {
    pub fn new(config: &ModelConfig, device: &Device) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.dim)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let layers = (0..config.layers)
            .map(|index| HyperConnection::new(config, config.is_sparse(index), device))
            .collect();

        let final_norm = RmsNormConfig::new(config.dim)
            .with_epsilon(1e-6)
            .init(device);

        Self {
            embedding,
            layers,
            final_norm,
            config: config.clone(),
        }
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Run the model over token ids.
    ///
    /// # Shapes
    ///   - Input `[batch, time]`
    ///   - Logits `[batch, time, vocab_size]`
    ///   - Usage `[layers, experts]`
    pub fn forward(&self, input_ids: Tensor<2, Int>) -> (Tensor<3>, Tensor<2>) {
        let embedded = self.embedding.forward(input_ids);
        self.forward_embeddings(embedded)
    }

    /// Run the model over precomputed embeddings.
    ///
    /// # Shapes
    ///   - Input `[batch, time, dim]`
    ///   - Logits `[batch, time, vocab_size]`
    ///   - Usage `[layers, experts]`
    pub fn forward_embeddings(&self, embedded: Tensor<3>) -> (Tensor<3>, Tensor<2>) {
        let [batch, time, dim] = embedded.dims();
        assert!(
            time <= self.config.max_sequence_length,
            "sequence of {time} exceeds the maximum of {}",
            self.config.max_sequence_length
        );

        // Four streams, all starting from the same embedding.
        let mut streams =
            embedded
                .unsqueeze_dim::<4>(2)
                .expand([batch, time, self.config.streams, dim]);

        let mut usages = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, usage) = layer.forward(streams);
            streams = next;
            usages.push(usage);
        }

        let hidden = self
            .final_norm
            .forward(streams.mean_dim(2).squeeze_dim::<3>(2));
        let logits = self.lm_head(hidden);

        (logits, Tensor::stack(usages, 0))
    }

    /// The tied language-model head.
    ///
    /// The reference ties the output projection to the embedding matrix, so
    /// both share one parameter. Burn's `Module` tree has no weight-sharing
    /// mechanism, so the head reads the embedding weight directly.
    fn lm_head(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, time, dim] = hidden.dims();
        let weight = self.embedding.weight.val();

        hidden
            .reshape::<2, _>([batch * time, dim])
            .matmul(weight.transpose())
            .reshape::<3, _>([batch, time, self.config.vocab_size])
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.num_params_recursive()
    }

    /// The parameter group of the last transformer block.
    pub fn last_block_group(&self) -> burn::module::ParamGroup {
        let block = self
            .layers
            .last()
            .expect("the model has at least one layer")
            .clone();
        burn::module::ParamGroup::ids_from_module(block)
    }

    /// The parameter group of the final norm.
    pub fn final_norm_group(&self) -> burn::module::ParamGroup {
        burn::module::ParamGroup::ids_from_module(self.final_norm.clone())
    }

    /// The parameter group of the embedding. This is also the tied LM head.
    pub fn embedding_group(&self) -> burn::module::ParamGroup {
        burn::module::ParamGroup::ids_from_module(self.embedding.clone())
    }

    /// The number of parameters that currently require a gradient.
    pub fn num_trainable_params(&self) -> usize {
        use burn::module::{ModuleVisitor, Param};

        #[derive(Default)]
        struct Counter {
            total: usize,
        }

        impl ModuleVisitor for Counter {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                if param.val().is_require_grad() {
                    self.total += param.lazy_shape().num_elements();
                }
            }
        }

        let mut counter = Counter::default();
        self.visit(&mut counter);
        counter.total
    }

    fn num_params_recursive(&self) -> usize {
        let mut total = 0usize;

        total += self.embedding.weight.num_params();
        total += self.final_norm.num_params();
        for layer in &self.layers {
            total += layer.num_params();
        }

        total
    }

    /// The number of parameters active for a single token.
    ///
    /// An architectural estimate, matching the reference: every parameter
    /// except the routed experts, plus the shared expert and `top_k` of the
    /// routed experts.
    pub fn num_active_params(&self) -> usize {
        let total = self.num_params();
        let routed = self.routed_expert_params();

        total - routed + routed * self.config.top_k / self.config.experts
    }

    fn routed_expert_params(&self) -> usize {
        let mut total = 0usize;
        for layer in &self.layers {
            total += layer.routed_expert_params();
        }
        total
    }
}
