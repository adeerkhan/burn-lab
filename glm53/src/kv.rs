//! Per-layer decode state.
//!
//! The full-sequence path recomputes the whole prefix at every generated token.
//! This keeps only what the next token needs:
//!
//! - linear attention: the running sum of `k v^T` and the running sum of `k`;
//! - sparse attention: the keys and values seen so far.

use burn::prelude::*;

/// What one layer keeps between tokens.
#[derive(Clone, Debug)]
pub enum LayerState {
    /// `(sum of k v^T, sum of k)`, both accumulated over the prefix.
    Linear(Option<(Tensor<4>, Tensor<4>)>),
    /// `(keys, values)` for positions `0..=t`.
    Sparse(Option<(Tensor<4>, Tensor<4>)>),
}

/// One entry per layer, in model order.
#[derive(Clone, Debug)]
pub struct Cache {
    pub layers: Vec<LayerState>,
}

impl Cache {
    /// An empty cache with the right state kind for every layer.
    pub fn new(config: &crate::ModelConfig) -> Self {
        Self {
            layers: (0..config.layers)
                .map(|index| {
                    if config.is_sparse(index) {
                        LayerState::Sparse(None)
                    } else {
                        LayerState::Linear(None)
                    }
                })
                .collect(),
        }
    }

    /// Forget everything, so the cache can be reused for another prompt.
    pub fn reset(&mut self) {
        for state in &mut self.layers {
            *state = match state {
                LayerState::Linear(_) => LayerState::Linear(None),
                LayerState::Sparse(_) => LayerState::Sparse(None),
            };
        }
    }
}
