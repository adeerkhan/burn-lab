//! Reproducer: one training step on wgpu is not bit-reproducible when the token
//! embedding is tied to the output projection.
//!
//! Run it twice and compare the printed hash:
//!
//! ```sh
//! cargo run -p glm53 --release --example determinism -- wgpu linear
//! cargo run -p glm53 --release --example determinism -- wgpu sparse
//! cargo run -p glm53 --release --example determinism -- flex linear
//! ```
//!
//! `wgpu` prints a different `embedding_grad` on every run while `loss` stays
//! identical; `flex` prints the same hash every time. The model is f32, so this
//! is not a half-precision accumulation effect.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, RmsNorm, RmsNormConfig};
use burn::optim::GradientsParams;
use burn::prelude::*;
use std::hash::{Hash, Hasher};

#[derive(Module, Debug)]
struct Model {
    embedding: Embedding,
    block: glm53::block::HyperConnection,
    norm: RmsNorm,
    streams: usize,
    dim: usize,
}

impl Model {
    fn forward(&self, ids: Tensor<2, Int>) -> Tensor<2> {
        let hidden = self.embedding.forward(ids);
        let [batch, time, dim] = hidden.dims();
        let streams = hidden
            .reshape([batch, time, 1, dim])
            .repeat_dim(2, self.streams);
        let (mixed, _usage) = self.block.forward(streams);
        let [_, _, _streams_out, dim_out] = mixed.dims();
        let mixed = mixed.sum_dim(2).squeeze_dim::<3>(2);
        let normed = self.norm.forward(mixed.reshape([batch * time, dim_out]));
        let weight = self.embedding.weight.val();
        normed.matmul(weight.transpose())
    }
}

struct Fingerprint(Option<u64>);

impl ModuleVisitor for Fingerprint {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let data = param
            .val()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for value in data {
            value.to_bits().hash(&mut hasher);
        }
        if self.0.is_none() {
            self.0 = Some(hasher.finish());
        }
    }
}

fn main() {
    let which = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "wgpu".to_string());
    let sparse = std::env::args()
        .nth(2)
        .map(|v| v == "sparse")
        .unwrap_or(false);
    let device = if which == "wgpu" {
        Device::wgpu(Default::default()).autodiff()
    } else {
        Device::flex().autodiff()
    };
    let d = &device;

    let config = glm53::ModelConfig::new()
        .with_dim(48)
        .with_layers(1)
        .with_heads(3)
        .with_expert_hidden(96)
        .with_experts(4)
        .with_top_k(1)
        .with_streams(2)
        .with_sparse_window(24)
        .with_sparse_stride(24)
        .with_max_sequence_length(128);

    let mut module = Model {
        embedding: EmbeddingConfig::new(260, 48)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(d),
        block: glm53::block::HyperConnection::new(&config, sparse, d),
        norm: RmsNormConfig::new(48).init(d),
        streams: 2,
        dim: 48,
    };
    module = glm53::seeding::initialize(module, 11, d);

    let mut weights = Fingerprint(None);
    module.visit(&mut weights);

    let ids = Tensor::<2, Int>::from_data(
        TensorData::new(
            (0..24 * 128)
                .map(|i| 10 + (i % 50) as i64)
                .collect::<Vec<_>>(),
            [24, 128],
        ),
        d,
    );
    let targets = Tensor::<1, Int>::from_data(
        TensorData::new(
            (0..24 * 128)
                .map(|i| 11 + (i % 50) as i64)
                .collect::<Vec<_>>(),
            [24 * 128],
        ),
        d,
    );

    let logits = module.forward(ids);
    let loss = CrossEntropyLossConfig::new()
        .init(d)
        .forward(logits, targets);
    let loss_value = loss.clone().into_scalar::<f32>();
    let grads = GradientsParams::from_grads(loss.backward(), &module);
    let gradient = grads.get::<2>(module.embedding.weight.id).unwrap();
    let data = gradient
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in data {
        value.to_bits().hash(&mut hasher);
    }

    println!(
        "{which} sparse={sparse} loss={loss_value:.9} embedding_grad={:016x}",
        hasher.finish()
    );
}
