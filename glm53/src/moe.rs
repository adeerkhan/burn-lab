//! Sparse mixture of experts: top-k routed experts plus one shared expert.
//!
//! Mirrors `glm53_flash/model.py::SparseMoE`.
//!
//! Two dispatches are implemented. [`SparseMoE::forward`] evaluates only the
//! experts each token selected, which is what the reference does and what keeps
//! activation memory proportional to `top_k` rather than to `experts`.
//! [`SparseMoE::forward_dense`] evaluates every expert on every token and weights
//! the results; it is numerically equivalent and retained because it is the
//! formulation the port shipped with and the reference point for the equivalence
//! test.

use burn::{
    nn::{Initializer, Linear, LinearConfig},
    prelude::*,
    tensor::{IndexingUpdateOp, activation},
};

use crate::config::ModelConfig;

/// One expert: a gated MLP with a doubled up-projection.
#[derive(Module, Debug)]
pub struct Expert {
    up: Linear,
    down: Linear,
    #[module(skip)]
    hidden: usize,
}

impl Expert {
    fn new(dim: usize, hidden: usize, device: &Device) -> Self {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let up = LinearConfig::new(dim, hidden * 2)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);
        let down = LinearConfig::new(hidden, dim)
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);

        Self { up, down, hidden }
    }

    /// The output width of this expert: the model width.
    fn down_weight_dim(&self) -> usize {
        self.down.weight.val().dims()[1]
    }

    /// # Shapes
    ///   - Input `[tokens, dim]`
    ///   - Output `[tokens, dim]`
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let [tokens, _] = x.dims();
        let up = self.up.forward(x);
        let gate = up.clone().slice([0..tokens, 0..self.hidden]);
        let value = up.slice([0..tokens, self.hidden..2 * self.hidden]);

        self.down.forward(activation::silu(gate).mul(value))
    }
}

/// Top-k routed experts plus one always-active shared expert.
#[derive(Module, Debug)]
pub struct SparseMoE {
    router: Linear,
    experts: Vec<Expert>,
    shared: Expert,
    #[module(skip)]
    num_experts: usize,
    #[module(skip)]
    top_k: usize,
}

/// The routing decision shared by both dispatches.
struct Routing {
    top_indices: Tensor<2, Int>,
    top_weights: Tensor<2>,
}

impl SparseMoE {
    pub fn new(config: &ModelConfig, device: &Device) -> Self {
        let router = LinearConfig::new(config.dim, config.experts)
            .with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let experts = (0..config.experts)
            .map(|_| Expert::new(config.dim, config.expert_hidden, device))
            .collect();
        let shared = Expert::new(config.dim, config.expert_hidden, device);

        Self {
            router,
            experts,
            shared,
            num_experts: config.experts,
            top_k: config.top_k,
        }
    }

    fn route(&self, flat: &Tensor<2>) -> Routing {
        let logits = self.router.forward(flat.clone());
        let (top_values, top_indices) = logits.topk_with_indices(self.top_k, 1);
        let top_weights = activation::softmax(top_values, 1);
        Routing {
            top_indices,
            top_weights,
        }
    }

    /// The share of routing slots each expert took, from a one-hot of the
    /// selected indices. Exact, and independent of the dispatch.
    fn usage(&self, routing: &Routing, tokens: usize) -> Tensor<1> {
        let one_hot = routing
            .top_indices
            .clone()
            .one_hot::<3>(self.num_experts)
            .float();
        let counts = one_hot.sum_dim(1).sum_dim(0);
        counts
            .reshape::<1, _>([self.num_experts])
            .div_scalar((tokens * self.top_k) as f32)
    }

    /// Evaluate only the experts each token selected.
    ///
    /// # Shapes
    ///   - Input `[batch, time, dim]`
    ///   - Output `[batch, time, dim]`
    ///   - Usage `[experts]`: the share of routing slots each expert took.
    ///
    /// # Performance
    ///
    /// Measured on this port, ~30x slower than [`forward`](Self::forward) at
    /// `batch 16 x 128` on wgpu. Every expert gets a different row count, so
    /// each step presents new tensor shapes to the JIT backend and it
    /// recompiles instead of reusing kernels. The memory saving is real: batch
    /// 32 fits here and did not with the dense path.
    pub fn forward_sparse(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<1>) {
        let [batch, time, dim] = x.dims();
        let tokens = batch * time;
        let flat = x.reshape::<2, _>([tokens, dim]);

        let routing = self.route(&flat);
        let mut output = self.shared.forward(flat.clone());

        // Only the tokens that picked this expert are evaluated on it. This is
        // the reference's gather/scatter dispatch; the host round-trip that
        // `argwhere` needs is the price of variable-length batching.
        for (expert_index, expert) in self.experts.iter().enumerate() {
            let selects = routing
                .top_indices
                .clone()
                .equal_scalar(expert_index as i64);
            // `any_dim` keeps the reduced axis, like `sum_dim`; flatten it so
            // `argwhere` reports row indices rather than (row, column) pairs.
            let rows = selects.clone().any_dim(1).reshape::<1, _>([tokens]);

            let count = rows.clone().int().sum().into_scalar::<i32>();
            if count == 0 {
                continue;
            }

            let indices = rows.argwhere().reshape::<1, _>([count as usize]);
            let selected = flat.clone().select(0, indices.clone());

            // The weight this expert was given, per token, then gathered.
            let weights = selects
                .float()
                .mul(routing.top_weights.clone())
                .sum_dim(1)
                .reshape::<1, _>([tokens])
                .select(0, indices.clone());

            let contribution = expert.forward(selected).mul(weights.unsqueeze_dim::<2>(1));

            output = output.select_assign(0, indices, contribution, IndexingUpdateOp::Add);
        }

        let usage = self.usage(&routing, tokens);

        (output.reshape::<3, _>([batch, time, dim]), usage)
    }

    /// Evaluate every expert on every token and weight the results.
    ///
    /// This is the default. Numerically equivalent to
    /// [`forward_sparse`](Self::forward_sparse), but it keeps
    /// `[tokens, experts, dim]` and every expert's up-projection alive for the
    /// backward pass, which is what limits the batch size.
    pub fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<1>) {
        let [batch, time, dim] = x.dims();
        let tokens = batch * time;
        let flat = x.reshape::<2, _>([tokens, dim]);

        let routing = self.route(&flat);

        // Dense routing weights: [tokens, experts], zero where a token did not
        // route. Built from a one-hot so the weights land in the right column.
        let one_hot = routing
            .top_indices
            .clone()
            .one_hot::<3>(self.num_experts)
            .float();
        let scaled = one_hot.clone().mul(
            routing
                .top_weights
                .clone()
                .reshape::<3, _>([tokens, self.top_k, 1]),
        );
        let dense_weights = scaled
            .sum_dim(1)
            .reshape::<2, _>([tokens, self.num_experts]);

        let outputs = self
            .experts
            .iter()
            .map(|expert| expert.forward(flat.clone()))
            .collect::<Vec<_>>();
        let stacked = Tensor::stack(outputs, 1);
        let routed = stacked
            .mul(dense_weights.unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape::<2, _>([tokens, self.config_dim()]);

        let shared = self.shared.forward(flat);
        let usage = self.usage(&routing, tokens);

        (
            shared.add(routed).reshape::<3, _>([batch, time, dim]),
            usage,
        )
    }

    fn config_dim(&self) -> usize {
        self.experts
            .first()
            .map(|expert| expert.down_weight_dim())
            .unwrap_or(0)
    }

    /// Parameters held by the routed experts, excluding the shared expert.
    pub fn routed_expert_params(&self) -> usize {
        self.experts.iter().map(|expert| expert.num_params()).sum()
    }
}
