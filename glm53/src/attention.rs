//! The two attention kinds of the hybrid model.
//!
//! * [`LinearAttention`] — causal linear attention with an ELU feature map.
//! * [`SparseAttention`] — gathered causal attention over a local window plus
//!   strided anchors.
//!
//! Both mirror `glm53_flash/model.py`.

use burn::{
    nn::{Initializer, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig},
    prelude::*,
    tensor::{FloatDType, activation},
};

use crate::config::ModelConfig;

/// Added to masked-out attention positions before the softmax.
const MASK_VALUE: f32 = -1e9;

/// The projections both attention kinds share.
#[derive(Module, Debug)]
pub struct AttentionBase {
    qkv: Linear,
    out: Linear,
    rope: RotaryEncoding,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl AttentionBase {
    fn new(config: &ModelConfig, device: &Device) -> Self {
        let qkv = LinearConfig::new(config.dim, 3 * config.dim)
            .with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);
        let out = LinearConfig::new(config.dim, config.dim)
            .with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);
        let rope =
            RotaryEncodingConfig::new(config.max_sequence_length, config.head_dim()).init(device);

        Self {
            qkv,
            out,
            rope,
            heads: config.heads,
            head_dim: config.head_dim(),
        }
    }

    /// Project to q, k, v and rotate q and k.
    ///
    /// Returns three tensors shaped `[batch, heads, time, head_dim]`.
    fn qkv_heads(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch, time, dim] = x.dims();
        let qkv = self.qkv.forward(x);

        let q = qkv.clone().slice([0..batch, 0..time, 0..dim]);
        let k = qkv.clone().slice([0..batch, 0..time, dim..2 * dim]);
        let v = qkv.slice([0..batch, 0..time, 2 * dim..3 * dim]);

        let shape = [batch, time, self.heads, self.head_dim];
        // Burn's RotaryEncoding expects (batch, heads, time, head_dim).
        let q = q.reshape::<4, _>(shape).swap_dims(1, 2);
        let k = k.reshape::<4, _>(shape).swap_dims(1, 2);
        let v = v.reshape::<4, _>(shape).swap_dims(1, 2);

        (self.rope.forward(q), self.rope.forward(k), v)
    }

    /// Apply the output projection to `[batch, heads, time, head_dim]`.
    fn project_out(&self, x: Tensor<4>) -> Tensor<3> {
        let [batch, heads, time, head_dim] = x.dims();
        let merged = x
            .swap_dims(1, 2)
            .reshape::<3, _>([batch, time, heads * head_dim]);
        self.out.forward(merged)
    }
}

/// Causal linear attention with an ELU feature map.
///
/// The reference materialises a cumulative sum of outer products,
/// `kv[t] = sum_{s<=t} k_s (x) v_s`, and contracts it with `q[t]`. Writing the
/// same quantity as a masked contraction gives the identical result with a
/// `[batch, heads, time, time]` intermediate instead of
/// `[batch, time, heads, head_dim, head_dim]`, which is what makes this
/// trainable at all.
#[derive(Module, Debug)]
pub struct LinearAttention {
    base: AttentionBase,
}

impl LinearAttention {
    pub fn new(config: &ModelConfig, device: &Device) -> Self {
        Self {
            base: AttentionBase::new(config, device),
        }
    }

    /// # Shapes
    ///   - Input `[batch, time, dim]`
    ///   - Output `[batch, time, dim]`
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, time, _] = x.dims();
        let (q, k, v) = self.base.qkv_heads(x);

        // ELU feature map: `elu(x) + 1` is positive, which is what makes the
        // denominator of the attention positive.
        let q = activation::elu(q, 1.0).add_scalar(1.0);
        let k = activation::elu(k, 1.0).add_scalar(1.0);

        // scores[b, h, t, s] = q_t . k_s
        let scores = q.matmul(k.clone().transpose());

        // Keep only the causal prefix: s <= t.
        let mask = causal_mask(time, &scores.device());
        let scores = scores.mul(mask);

        // numerator[t] = sum_{s<=t} (q_t . k_s) v_s
        let numerator = scores.clone().matmul(v);
        // denominator[t] = sum_{s<=t} (q_t . k_s)
        let denominator = scores.sum_dim(3);

        let output = numerator.div(denominator.clamp_min(1e-6));

        self.base.project_out(output)
    }
}

/// Gathered causal attention over a local window plus strided anchors.
///
/// The reference gathers, for every query position, the keys at multiples of
/// `stride` and the `window` keys ending at the query. That gathered softmax is
/// the same as a full softmax with every non-selected key masked out, so this
/// implementation masks instead of gathering — Burn's `gather` cannot produce
/// the rank-5 intermediate the reference needs.
#[derive(Module, Debug)]
pub struct SparseAttention {
    base: AttentionBase,
    #[module(skip)]
    window: usize,
    #[module(skip)]
    stride: usize,
}

impl SparseAttention {
    pub fn new(config: &ModelConfig, device: &Device) -> Self {
        Self {
            base: AttentionBase::new(config, device),
            window: config.sparse_window,
            stride: config.sparse_stride,
        }
    }

    /// # Shapes
    ///   - Input `[batch, time, dim]`
    ///   - Output `[batch, time, dim]`
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, time, _] = x.dims();
        let (q, k, v) = self.base.qkv_heads(x);

        let scores = q
            .matmul(k.transpose())
            .mul_scalar((self.base.head_dim as f32).powf(-0.5));

        let mask = sparse_mask(time, self.window, self.stride, &scores.device());
        let scores = scores.add(mask);

        // The reference softmaxes in float32 and casts back.
        let dtype = scores.dtype();
        let weights = activation::softmax(scores.cast(FloatDType::F32), 3).cast(dtype);
        let output = weights.matmul(v);

        self.base.project_out(output)
    }
}

/// A `[1, 1, time, time]` mask that is 1 where a query may attend a key.
fn causal_mask(time: usize, device: &Device) -> Tensor<4> {
    let mut data = Vec::with_capacity(time * time);
    for query in 0..time {
        for key in 0..time {
            data.push(if key <= query { 1.0f32 } else { 0.0f32 });
        }
    }
    Tensor::<4>::from_data(TensorData::new(data, [1, 1, time, time]), device)
}

/// A `[1, 1, time, time]` additive mask for the gathered sparse attention.
///
/// A key is visible to a query when it is a strided anchor at or before the
/// query, or inside the local window ending at the query.
fn sparse_mask(time: usize, window: usize, stride: usize, device: &Device) -> Tensor<4> {
    let mut data = Vec::with_capacity(time * time);
    for query in 0..time {
        for key in 0..time {
            let anchor = key % stride == 0;
            let local = key + window > query;
            let visible = key <= query && (anchor || local);
            data.push(if visible { 0.0f32 } else { MASK_VALUE });
        }
    }
    Tensor::<4>::from_data(TensorData::new(data, [1, 1, time, time]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Device::flex()
    }

    #[test]
    fn causal_mask_only_keeps_the_prefix() {
        let mask = causal_mask(4, &device());
        let data = mask.to_data();
        let values = data.as_slice::<f32>().unwrap();

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        // Row 2: keys 0 and 1 visible, 2 and 3 masked.
        assert_eq!(values[2 * 4], 1.0);
        assert_eq!(values[2 * 4 + 1], 1.0);
        // Key 2 is the query's own position, so it is visible.
        assert_eq!(values[2 * 4 + 2], 1.0);
        assert_eq!(values[2 * 4 + 3], 0.0);
    }

    #[test]
    fn sparse_mask_keeps_anchors_and_the_local_window() {
        // stride 4, window 2: anchors are keys 0 and 4, the window covers the
        // two keys ending at the query.
        let mask = sparse_mask(6, 2, 4, &device());
        let data = mask.to_data();
        let values = data.as_slice::<f32>().unwrap();

        // Query 5 sees anchor 0, anchor 4, and the local window {4, 5}.
        assert_eq!(values[5 * 6], 0.0);
        assert_eq!(values[5 * 6 + 4], 0.0);
        assert_eq!(values[5 * 6 + 5], 0.0);
        // Key 3 is neither an anchor nor in the window.
        assert_eq!(values[5 * 6 + 3], MASK_VALUE);
        // Key 1 is not an anchor (1 % 4 != 0) and outside the window.
        assert_eq!(values[5 * 6 + 1], MASK_VALUE);
    }

    #[test]
    fn every_sparse_row_has_at_least_one_visible_key() {
        let mask = sparse_mask(16, 4, 4, &device());
        let data = mask.to_data();
        let values = data.as_slice::<f32>().unwrap();
        for query in 0..16 {
            let visible = (0..16)
                .filter(|key| values[query * 16 + key] == 0.0)
                .count();
            assert!(visible > 0, "row {query} has no visible key");
        }
    }

    /// The reference materialises `kv[t] = sum_{s<=t} k_s (x) v_s` and contracts
    /// `q[t]` against it. This implementation instead masks a full contraction.
    /// The two must agree; this checks it against a direct transcription of the
    /// reference's definition.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn linear_attention_matches_the_cumulative_sum_definition() {
        let device = device();
        let config = ModelConfig::new()
            .with_dim(4)
            .with_heads(1)
            .with_layers(1)
            .with_experts(2)
            .with_expert_hidden(8)
            .with_streams(2)
            .with_max_sequence_length(8);

        let attention = LinearAttention::new(&config, &device);

        #[rustfmt::skip]
        let input = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    0.5f32, -1.0, 0.25, 2.0,
                    1.5, 0.75, -0.5, 0.1,
                    -0.2, 0.3, 1.1, -1.4,
                    0.9, -0.6, 0.4, 0.8,
                ],
                [1, 4, 4],
            ),
            &device,
        );

        // The Burn path.
        let expected = attention.forward(input.clone());

        // A direct transcription of the reference, on the same q/k/v.
        let (q, k, v) = attention.base.qkv_heads(input);
        let q = q.to_data();
        let k = k.to_data();
        let v = v.to_data();
        let q = q.as_slice::<f32>().unwrap();
        let k = k.as_slice::<f32>().unwrap();
        let v = v.as_slice::<f32>().unwrap();

        let (batch, heads, time, head_dim) = (1usize, 1usize, 4usize, 4usize);
        let mut reference = vec![0.0f32; batch * heads * time * head_dim];

        // The reference applies `elu(x) + 1` to q and k before contracting.
        let feature = |x: f32| if x > 0.0 { x + 1.0 } else { x.exp() };
        let q: Vec<f32> = q.iter().map(|x| feature(*x)).collect();
        let k: Vec<f32> = k.iter().map(|x| feature(*x)).collect();
        let q = &q[..];
        let k = &k[..];

        for b in 0..batch {
            for h in 0..heads {
                for t in 0..time {
                    // kv = sum_{s<=t} k_s (x) v_s, contracted with q_t.
                    let mut numerator = vec![0.0f32; head_dim];
                    let mut denominator = 0.0f32;
                    for s in 0..=t {
                        let score: f32 = (0..head_dim)
                            .map(|d| {
                                let at = ((b * heads + h) * time + s) * head_dim + d;
                                let bt = ((b * heads + h) * time + t) * head_dim + d;
                                q[bt] * k[at]
                            })
                            .sum();
                        denominator += score;
                        for d in 0..head_dim {
                            let at = ((b * heads + h) * time + s) * head_dim + d;
                            numerator[d] += score * v[at];
                        }
                    }
                    let denominator = denominator.max(1e-6);
                    for d in 0..head_dim {
                        let out = ((b * heads + h) * time + t) * head_dim + d;
                        reference[out] = numerator[d] / denominator;
                    }
                }
            }
        }

        let reference = Tensor::<4>::from_data(
            TensorData::new(reference, [batch, heads, time, head_dim]),
            &device,
        );
        let expected_reference = attention.base.project_out(reference);

        let actual = expected.to_data();
        let reference = expected_reference.to_data();
        actual.assert_approx_eq::<f32>(&reference, burn::tensor::Tolerance::default());
    }
}
