//! Model configuration, mirroring `glm53_flash/model.py::ModelConfig`.

use burn::config::Config;

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Number of byte tokens (4 special tokens + 256 byte values).
    #[config(default = 260)]
    pub vocab_size: usize,
    /// Model width.
    #[config(default = 192)]
    pub dim: usize,
    /// Number of hybrid blocks.
    #[config(default = 12)]
    pub layers: usize,
    /// Number of attention heads.
    #[config(default = 6)]
    pub heads: usize,
    /// Hidden width of a single expert.
    #[config(default = 384)]
    pub expert_hidden: usize,
    /// Number of routed experts.
    #[config(default = 8)]
    pub experts: usize,
    /// How many routed experts each token activates.
    #[config(default = 2)]
    pub top_k: usize,
    /// Number of residual streams (the "mHC" simplification).
    #[config(default = 4)]
    pub streams: usize,
    /// Local window of the gathered sparse attention.
    #[config(default = 32)]
    pub sparse_window: usize,
    /// Stride of the sparse attention anchors.
    #[config(default = 32)]
    pub sparse_stride: usize,
    /// Maximum sequence length.
    #[config(default = 192)]
    pub max_sequence_length: usize,
}

impl ModelConfig {
    /// Width of a single attention head.
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    /// Whether the block at `index` uses the gathered sparse attention.
    ///
    /// The reference picks sparse attention when `(index + 1) % 4 == 0`, which
    /// gives 9 linear blocks and 3 sparse blocks for the default 12 layers.
    pub fn is_sparse(&self, index: usize) -> bool {
        (index + 1).is_multiple_of(4)
    }
}
