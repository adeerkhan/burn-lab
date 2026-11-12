//! Burn-only reduction: embedding -> one causal attention block -> RMSNorm ->
//! tied head. Nothing from glm53 except the seeding of the weights.
//!
//! Run twice; a different `embedding_grad` means wgpu is not bit-reproducible.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::optim::GradientsParams;
use burn::prelude::*;
use std::hash::{Hash, Hasher};

#[derive(Module, Debug)]
struct Block {
    qkv: Linear,
    out: Linear,
}

impl Block {
    fn new(dim: usize, device: &Device) -> Self {
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        Self {
            qkv: LinearConfig::new(dim, dim * 3)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            out: LinearConfig::new(dim, dim)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, time, dim] = x.dims();
        let qkv = self.qkv.forward(x);
        let q = qkv.clone().slice([0..batch, 0..time, 0..dim]);
        let k = qkv.clone().slice([0..batch, 0..time, dim..2 * dim]);
        let v = qkv.slice([0..batch, 0..time, 2 * dim..3 * dim]);

        let scores = q.matmul(k.transpose()).mul_scalar((dim as f32).powf(-0.5));
        let mut mask = Vec::with_capacity(time * time);
        for query in 0..time {
            for key in 0..time {
                mask.push(if key <= query { 0.0f32 } else { -1.0e30 });
            }
        }
        let mask = Tensor::<3>::from_data(TensorData::new(mask, [1, time, time]), &scores.device());
        let weights = burn::tensor::activation::softmax(scores.add(mask), 2);
        self.out.forward(weights.matmul(v))
    }
}

#[derive(Module, Debug)]
struct Model {
    embedding: Embedding,
    block: Block,
    norm: RmsNorm,
}

impl Model {
    fn new(vocab: usize, dim: usize, device: &Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab, dim)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            block: Block::new(dim, device),
            norm: RmsNormConfig::new(dim).init(device),
        }
    }

    fn forward(&self, ids: Tensor<2, Int>) -> Tensor<2> {
        let hidden = self.embedding.forward(ids);
        let [batch, time, dim] = hidden.dims();
        let mixed = self.block.forward(hidden);
        let normed = self.norm.forward(mixed.reshape([batch * time, dim]));
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
    let device = if which == "wgpu" {
        Device::wgpu(Default::default()).autodiff()
    } else {
        Device::flex().autodiff()
    };
    let d = &device;

    d.seed(7);
    let module = Model::new(260, 48, d);
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
        "{which} loss={loss_value:.9} weights={:016x} embedding_grad={:016x}",
        weights.0.unwrap(),
        hasher.finish()
    );
}
