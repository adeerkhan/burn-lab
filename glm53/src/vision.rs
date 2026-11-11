//! The miniature GLM-5.3-Flash image path.
//!
//! Mirrors `glm53_flash/vision.py`: RGB patch embedding with a learned position
//! table, non-causal vision transformer blocks, a 2x2 spatial merge, and a gated
//! projector that emits language-model-width image tokens. A clearly labelled
//! direct-patch baseline is kept for comparison.
//!
//! The vision path appends three ids after the 260-token byte vocabulary, so it
//! needs `vocab_size >= 263`. It never modifies the language model: text-only
//! behaviour with `vocab_size = 260` is untouched.

use burn::{
    module::Param,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::{activation, module::attention, ops::AttentionModuleOptions},
};

use crate::model::GLM53Flash;

/// `<image_start>`, appended after the byte vocabulary.
pub const IMAGE_START: usize = 260;
/// The placeholder replaced by a visual embedding.
pub const IMAGE_PLACEHOLDER: usize = 261;
/// `<image_end>`.
pub const IMAGE_END: usize = 262;
/// The smallest vocabulary that can express the image span.
pub const REQUIRED_VOCAB: usize = 263;
/// The byte tokenizer's beginning-of-sequence id.
pub const BOS: usize = 1;

/// Geometry and width of the vision tower.
#[derive(Config, Debug)]
pub struct VisionConfig {
    #[config(default = 32)]
    pub image_size: usize,
    #[config(default = 4)]
    pub patch_size: usize,
    #[config(default = 3)]
    pub in_channels: usize,
    #[config(default = 32)]
    pub hidden_size: usize,
    #[config(default = 2)]
    pub depth: usize,
    #[config(default = 4)]
    pub heads: usize,
    #[config(default = 64)]
    pub intermediate_size: usize,
    #[config(default = 2)]
    pub spatial_merge_size: usize,
    #[config(default = 64)]
    pub projection_intermediate_size: usize,
    #[config(default = 1e-5)]
    pub rms_norm_eps: f64,
}

impl VisionConfig {
    /// Patches per side.
    pub fn patch_grid_size(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Merged tokens per side.
    pub fn merged_grid_size(&self) -> usize {
        self.patch_grid_size() / self.spatial_merge_size
    }

    /// Visual tokens produced per image.
    pub fn visual_token_count(&self) -> usize {
        self.merged_grid_size() * self.merged_grid_size()
    }

    /// Reject geometries the topology cannot express.
    pub fn validate(&self) -> Result<(), String> {
        if self.in_channels != 3 {
            return Err("the miniature vision path requires RGB input".to_string());
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err("image_size must be divisible by patch_size".to_string());
        }
        if !self
            .patch_grid_size()
            .is_multiple_of(self.spatial_merge_size)
        {
            return Err("patch grid must be divisible by spatial_merge_size".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.heads) {
            return Err("vision hidden_size must be divisible by heads".to_string());
        }
        if self.depth < 1 {
            return Err("vision depth must be positive".to_string());
        }
        if self.spatial_merge_size != 2 {
            return Err("this teaching merger implements 2x2 merging only".to_string());
        }
        Ok(())
    }
}

/// Embed non-overlapping RGB patches and add a learned 2D position table.
#[derive(Module, Debug)]
pub struct PatchEmbedding {
    proj: Conv2d,
    position: Param<Tensor<3>>,
}

impl PatchEmbedding {
    fn new(config: &VisionConfig, device: &Device) -> Self {
        config.validate().expect("valid vision config");
        let proj = Conv2dConfig::new(
            [config.in_channels, config.hidden_size],
            [config.patch_size; 2],
        )
        .with_stride([config.patch_size; 2])
        .with_bias(true)
        .init(device);

        let tokens = config.patch_grid_size() * config.patch_grid_size();
        let position = Tensor::<3>::random(
            [1, tokens, config.hidden_size],
            burn::tensor::Distribution::Default,
            device,
        )
        .mul_scalar(0.02);

        Self {
            proj,
            position: Param::from_tensor(position),
        }
    }

    /// # Shapes
    ///   - Images `[batch, 3, image, image]`
    ///   - Tokens `[batch, patches, hidden]`
    fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let [batch, channels, height, width] = images.dims();
        assert_eq!(channels, 3, "images must have three channels");

        let feature_map = self.proj.forward(images);
        let [_, hidden, grid_h, grid_w] = feature_map.dims();
        let _ = (height, width);
        let tokens = feature_map
            .reshape::<3, _>([batch, hidden, grid_h * grid_w])
            .swap_dims(1, 2);

        tokens.add(self.position.val())
    }
}

/// Full bidirectional vision attention with per-head query and key RMSNorm.
#[derive(Module, Debug)]
pub struct VisionAttention {
    qkv: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    proj: Linear,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl VisionAttention {
    fn new(config: &VisionConfig, device: &Device) -> Self {
        let hidden = config.hidden_size;
        Self {
            qkv: LinearConfig::new(hidden, hidden * 3)
                .with_bias(true)
                .init(device),
            q_norm: RmsNormConfig::new(hidden / config.heads)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(hidden / config.heads)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            proj: LinearConfig::new(hidden, hidden)
                .with_bias(true)
                .init(device),
            heads: config.heads,
            head_dim: hidden / config.heads,
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let [batch, length, hidden] = hidden_states.dims();

        let qkv = self.qkv.forward(hidden_states);
        let q = qkv.clone().slice([0..batch, 0..length, 0..hidden]);
        let k = qkv.clone().slice([0..batch, 0..length, hidden..2 * hidden]);
        let v = qkv.slice([0..batch, 0..length, 2 * hidden..3 * hidden]);

        let shape = [batch, length, self.heads, self.head_dim];
        let q = self
            .q_norm
            .forward(q.reshape::<4, _>(shape))
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(k.reshape::<4, _>(shape))
            .swap_dims(1, 2);
        let v = v.reshape::<4, _>(shape).swap_dims(1, 2);

        // Non-causal, unmasked, and scaled by `1/sqrt(head_dim)`.
        let attention = attention(q, k, v, None, None, None, AttentionModuleOptions::default());

        let merged = attention
            .swap_dims(1, 2)
            .reshape::<3, _>([batch, length, hidden]);
        self.proj.forward(merged)
    }
}

/// Gated feed-forward with a SwiGLU activation.
#[derive(Module, Debug)]
pub struct VisionSwiGlu {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionSwiGlu {
    fn new(config: &VisionConfig, device: &Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(true)
                .init(device),
            up_proj: LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(true)
                .init(device),
            down_proj: LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let gate = activation::silu(self.gate_proj.forward(hidden_states.clone()));
        self.down_proj
            .forward(gate.mul(self.up_proj.forward(hidden_states)))
    }
}

/// Pre-normalised non-causal attention followed by a SwiGLU MLP.
#[derive(Module, Debug)]
pub struct VisionBlock {
    norm1: RmsNorm,
    attention: VisionAttention,
    norm2: RmsNorm,
    mlp: VisionSwiGlu,
}

impl VisionBlock {
    fn new(config: &VisionConfig, device: &Device) -> Self {
        Self {
            norm1: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            attention: VisionAttention::new(config, device),
            norm2: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            mlp: VisionSwiGlu::new(config, device),
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let attended = hidden_states.clone()
            + self
                .attention
                .forward(self.norm1.forward(hidden_states.clone()));
        attended.clone() + self.mlp.forward(self.norm2.forward(attended))
    }
}

/// Learn one output token from each non-overlapping 2x2 neighbourhood.
#[derive(Module, Debug)]
pub struct SpatialMerger2x2 {
    downsample: Linear,
}

impl SpatialMerger2x2 {
    fn new(vision_dim: usize, output_dim: usize, device: &Device) -> Self {
        Self {
            downsample: LinearConfig::new(vision_dim * 4, output_dim)
                .with_bias(true)
                .init(device),
        }
    }

    /// # Shapes
    ///   - Tokens `[batch, grid_h * grid_w, vision_dim]`
    ///   - Merged `[batch, (grid_h/2) * (grid_w/2), output_dim]`
    fn forward(&self, hidden_states: Tensor<3>, grid_h: usize, grid_w: usize) -> Tensor<3> {
        let [batch, tokens, hidden] = hidden_states.dims();
        assert_eq!(tokens, grid_h * grid_w, "vision tokens must form the grid");
        assert!(
            grid_h.is_multiple_of(2) && grid_w.is_multiple_of(2),
            "vision tokens must form an even 2D grid"
        );

        let feature_map = hidden_states
            .swap_dims(1, 2)
            .reshape::<4, _>([batch, hidden, grid_h, grid_w]);
        let neighbourhoods = feature_map
            .reshape::<6, _>([batch, hidden, grid_h / 2, 2, grid_w / 2, 2])
            .permute([0, 2, 4, 1, 3, 5])
            .reshape::<3, _>([batch, (grid_h / 2) * (grid_w / 2), hidden * 4]);

        self.downsample.forward(neighbourhoods)
    }
}

/// GLM-style projection MLP emitting language-model-width image tokens.
#[derive(Module, Debug)]
pub struct MultimodalProjector {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MultimodalProjector {
    fn new(dim: usize, intermediate: usize, device: &Device) -> Self {
        Self {
            proj: LinearConfig::new(dim, dim).with_bias(false).init(device),
            post_projection_norm: LayerNormConfig::new(dim).init(device),
            gate_proj: LinearConfig::new(dim, intermediate)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(dim, intermediate)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate, dim)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let x = activation::gelu(
            self.post_projection_norm
                .forward(self.proj.forward(hidden_states)),
        );
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate.mul(self.up_proj.forward(x)))
    }
}

/// The architecture-faithful miniature vision tower.
#[derive(Module, Debug)]
pub struct VisionEncoder {
    patch_embedding: PatchEmbedding,
    blocks: Vec<VisionBlock>,
    post_layernorm: RmsNorm,
    spatial_merger: SpatialMerger2x2,
    projector: MultimodalProjector,
    #[module(skip)]
    config: VisionConfig,
}

impl VisionEncoder {
    pub fn new(config: &VisionConfig, language_dim: usize, device: &Device) -> Self {
        config.validate().expect("valid vision config");
        Self {
            patch_embedding: PatchEmbedding::new(config, device),
            blocks: (0..config.depth)
                .map(|_| VisionBlock::new(config, device))
                .collect(),
            post_layernorm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            spatial_merger: SpatialMerger2x2::new(config.hidden_size, language_dim, device),
            projector: MultimodalProjector::new(
                language_dim,
                config.projection_intermediate_size,
                device,
            ),
            config: config.clone(),
        }
    }

    /// Visual tokens produced per image.
    pub fn visual_token_count(&self) -> usize {
        self.config.visual_token_count()
    }

    /// # Shapes
    ///   - Images `[batch, 3, image, image]`
    ///   - Tokens `[batch, visual_tokens, language_dim]`
    pub fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let mut hidden_states = self.patch_embedding.forward(images);
        for block in &self.blocks {
            hidden_states = block.forward(hidden_states);
        }
        let hidden_states = self.post_layernorm.forward(hidden_states);
        let grid = self.config.patch_grid_size();
        let merged = self.spatial_merger.forward(hidden_states, grid, grid);
        self.projector.forward(merged)
    }
}

/// A language model with the faithful vision tower attached.
#[derive(Module, Debug)]
pub struct VisionLanguageModel {
    language_model: GLM53Flash,
    vision_encoder: VisionEncoder,
}

impl VisionLanguageModel {
    /// # Panics
    ///
    /// Panics if the language model's vocabulary cannot express the image span.
    pub fn new(language_model: GLM53Flash, vision_config: VisionConfig, device: &Device) -> Self {
        assert!(
            language_model.config().vocab_size >= REQUIRED_VOCAB,
            "language model vocab_size must be at least {REQUIRED_VOCAB}"
        );
        let encoder = VisionEncoder::new(&vision_config, language_model.config().dim, device);
        Self {
            language_model,
            vision_encoder: encoder,
        }
    }

    /// Visual tokens produced per image.
    pub fn visual_token_count(&self) -> usize {
        self.vision_encoder.visual_token_count()
    }

    /// Visual tokens plus the two boundary tokens.
    pub fn inserted_token_count(&self) -> usize {
        self.visual_token_count() + 2
    }

    /// The language model underneath.
    pub fn language_model(&self) -> &GLM53Flash {
        &self.language_model
    }

    /// The vision tower's parameters.
    pub fn vision_group(&self) -> burn::module::ParamGroup {
        burn::module::ParamGroup::ids_from_module(self.vision_encoder.clone())
    }

    /// The last language block's parameters.
    pub fn last_block_group(&self) -> burn::module::ParamGroup {
        self.language_model.last_block_group()
    }

    /// The language model's final norm.
    pub fn final_norm_group(&self) -> burn::module::ParamGroup {
        self.language_model.final_norm_group()
    }

    /// The tied embedding, which is also the language head.
    pub fn embedding_group(&self) -> burn::module::ParamGroup {
        self.language_model.embedding_group()
    }

    /// # Shapes
    ///   - Images `[batch, 3, image, image]`
    ///   - Text ids `[batch, time]`
    pub fn forward(&self, images: Tensor<4>, text_ids: Tensor<2, Int>) -> (Tensor<3>, Tensor<2>) {
        let device = text_ids.device();
        let multimodal = build_multimodal_input_ids(text_ids, self.visual_token_count(), &device);

        // The placeholders start after the optional BOS and <image_start>.
        let inset = self.placeholder_offset(&multimodal);

        let embedded = self.language_model.embed(multimodal);
        let visual = self.vision_encoder.forward(images);
        let spliced = splice(embedded, visual, inset);
        self.language_model.forward_embeddings(spliced)
    }

    /// How many tokens precede the placeholders: 2 with a BOS, otherwise 1.
    fn placeholder_offset(&self, multimodal: &Tensor<2, Int>) -> usize {
        let [batch, _] = multimodal.dims();
        let first = multimodal
            .clone()
            .slice([0..batch, 0..1])
            .into_data()
            .convert::<i32>();
        let leading = first.as_slice::<i32>().expect("int token ids")[0];
        if leading as usize == BOS { 2 } else { 1 }
    }
}

/// Insert a contiguous image span after the first `inset` text tokens.
///
/// The reference scatters visual tokens into `placeholder` positions; the span is
/// always contiguous and starts right after the optional BOS, so splicing is the
/// same operation without a host round-trip.
fn splice(text: Tensor<3>, visual: Tensor<3>, inset: usize) -> Tensor<3> {
    let [batch, time, dim] = text.dims();
    let placeholders = visual.dims()[1];
    assert!(
        inset + placeholders <= time,
        "the multimodal sequence has no room for the image span"
    );

    let head = text.clone().slice([0..batch, 0..inset, 0..dim]);
    // The placeholders are replaced, not kept alongside the visual tokens.
    let tail = text.slice([0..batch, inset + placeholders..time, 0..dim]);
    Tensor::cat(vec![head, visual, tail], 1)
}

/// Build `<image_start> placeholder x n <image_end>` text ids.
pub fn build_multimodal_input_ids(
    text_ids: Tensor<2, Int>,
    visual_token_count: usize,
    device: &Device,
) -> Tensor<2, Int> {
    let [batch, time] = text_ids.dims();
    assert!(time >= 1, "text ids need at least one token");
    assert!(
        visual_token_count >= 1,
        "visual token count must be positive"
    );

    let span_len = visual_token_count + 2;
    let mut span = vec![IMAGE_PLACEHOLDER as i64; span_len];
    span[0] = IMAGE_START as i64;
    span[span_len - 1] = IMAGE_END as i64;
    let span_data: Vec<i64> = (0..batch).flat_map(|_| span.clone()).collect();
    let span_tensor =
        Tensor::<2, Int>::from_data(TensorData::new(span_data, [batch, span_len]), device);

    let first = text_ids.clone().slice([0..batch, 0..1]);
    let data = first.into_data().convert::<i32>();
    let values = data.as_slice::<i32>().expect("int token ids");
    let all_bos = values.iter().all(|value| *value as usize == BOS);
    let any_bos = values.iter().any(|value| *value as usize == BOS);
    assert!(
        all_bos || !any_bos,
        "all rows must agree on whether a BOS token is present"
    );

    if all_bos {
        let head = text_ids.clone().slice([0..batch, 0..1]);
        let tail = text_ids.slice([0..batch, 1..time]);
        Tensor::cat(vec![head, span_tensor, tail], 1)
    } else {
        Tensor::cat(vec![span_tensor, text_ids], 1)
    }
}

/// Baseline only: project RGB patches straight to language-model tokens.
#[derive(Module, Debug)]
pub struct DirectPatchEncoder {
    patch_projection: Conv2d,
    position: Param<Tensor<3>>,
    modality: Param<Tensor<3>>,
    #[module(skip)]
    token_count: usize,
}

impl DirectPatchEncoder {
    fn new(dim: usize, patch_size: usize, image_size: usize, device: &Device) -> Self {
        assert!(
            image_size.is_multiple_of(patch_size),
            "image_size must be divisible by patch_size"
        );
        let token_count = (image_size / patch_size) * (image_size / patch_size);
        let patch_projection = Conv2dConfig::new([3, dim], [patch_size; 2])
            .with_stride([patch_size; 2])
            .with_bias(true)
            .init(device);
        let position = Tensor::<3>::random(
            [1, token_count, dim],
            burn::tensor::Distribution::Default,
            device,
        )
        .mul_scalar(0.02);
        let modality =
            Tensor::<3>::random([1, 1, dim], burn::tensor::Distribution::Default, device)
                .mul_scalar(0.02);

        Self {
            patch_projection,
            position: Param::from_tensor(position),
            modality: Param::from_tensor(modality),
            token_count,
        }
    }

    fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let [batch, channels, _, _] = images.dims();
        assert_eq!(channels, 3, "images must have three channels");

        let patches = self.patch_projection.forward(images);
        let [_, dim, grid_h, grid_w] = patches.dims();
        patches
            .reshape::<3, _>([batch, dim, grid_h * grid_w])
            .swap_dims(1, 2)
            .add(self.position.val())
            .add(self.modality.val())
    }
}

/// Clearly labelled reproduction of the direct-patch teaching baseline.
#[derive(Module, Debug)]
pub struct DirectPatchVisionLanguageModel {
    language_model: GLM53Flash,
    image_encoder: DirectPatchEncoder,
}

impl DirectPatchVisionLanguageModel {
    pub fn new(language_model: GLM53Flash, patch_size: usize, device: &Device) -> Self {
        let dim = language_model.config().dim;
        let image_encoder = DirectPatchEncoder::new(dim, patch_size, 32, device);
        Self {
            language_model,
            image_encoder,
        }
    }

    /// Visual tokens produced per image.
    pub fn visual_token_count(&self) -> usize {
        self.image_encoder.token_count
    }

    /// This baseline inserts exactly its visual tokens.
    pub fn inserted_token_count(&self) -> usize {
        self.visual_token_count()
    }

    pub fn forward(&self, images: Tensor<4>, text_ids: Tensor<2, Int>) -> (Tensor<3>, Tensor<2>) {
        let visual = self.image_encoder.forward(images);
        let text = self.language_model.embed(text_ids);
        let embedded = Tensor::cat(vec![visual, text], 1);
        self.language_model.forward_embeddings(embedded)
    }
}

/// Build next-byte labels that score only the answer tokens.
///
/// Positions outside the answer are `-100`. The multimodal sequence is
/// `inserted + text` long, so it yields `inserted + text - 1` predictions.
pub fn answer_only_labels(
    text_ids: Tensor<2, Int>,
    prompt_length: usize,
    visual_token_count: usize,
    boundary_token_count: usize,
    device: &Device,
) -> Tensor<2, Int> {
    let [batch, time] = text_ids.dims();
    assert!(
        prompt_length >= 1 && prompt_length < time,
        "prompt_length must leave at least one answer token"
    );

    let inserted = visual_token_count + boundary_token_count;
    let width = inserted + time - 1;
    let answer_len = time - prompt_length;
    let start = inserted + prompt_length - 1;

    let mut labels = vec![-100i64; batch * width];
    let answers = text_ids
        .slice([0..batch, prompt_length..time])
        .into_data()
        .convert::<i32>();
    let answers = answers.as_slice::<i32>().expect("int labels");

    for row in 0..batch {
        for column in 0..answer_len {
            labels[row * width + start + column] = answers[row * answer_len + column] as i64;
        }
    }

    Tensor::<2, Int>::from_data(TensorData::new(labels, [batch, width]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    fn language_config() -> ModelConfig {
        ModelConfig::new()
            .with_vocab_size(REQUIRED_VOCAB)
            .with_dim(32)
            .with_heads(4)
            .with_layers(1)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_sparse_window(8)
            .with_sparse_stride(8)
            .with_max_sequence_length(64)
    }

    fn vision_config() -> VisionConfig {
        VisionConfig::new()
            .with_hidden_size(24)
            .with_intermediate_size(48)
    }

    #[test]
    fn config_geometry_is_consistent() {
        let config = vision_config();
        assert_eq!(config.patch_grid_size(), 8);
        assert_eq!(config.merged_grid_size(), 4);
        assert_eq!(config.visual_token_count(), 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_rejects_bad_geometry() {
        let config = VisionConfig::new().with_heads(5).with_hidden_size(24);
        assert!(config.validate().is_err());
    }

    #[test]
    fn image_span_is_inserted_after_bos() {
        let device = Device::flex();
        let ids: Vec<i64> = vec![BOS as i64, 14, 40, 20];
        let text = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, 4]), &device);
        let multimodal = build_multimodal_input_ids(text, 16, &device);

        assert_eq!(multimodal.dims(), [1, 4 + 18]);
        let data = multimodal.into_data().convert::<i32>();
        let values = data.as_slice::<i32>().unwrap();
        assert_eq!(values[0] as usize, BOS);
        assert_eq!(values[1] as usize, IMAGE_START);
        assert_eq!(values[2] as usize, IMAGE_PLACEHOLDER);
        assert_eq!(values[17] as usize, IMAGE_PLACEHOLDER);
        assert_eq!(values[18] as usize, IMAGE_END);
        assert_eq!(values[19], 14);
    }

    #[test]
    fn answer_only_labels_mask_everything_but_the_answer() {
        let device = Device::flex();
        // "Digit: " is 7 bytes plus BOS; the answer is the digit byte.
        let ids: Vec<i64> = vec![BOS as i64, 14, 40, 20, 30];
        let text = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, 5]), &device);
        let labels = answer_only_labels(text, 4, 16, 2, &device);

        let [batch, width] = labels.dims();
        assert_eq!(batch, 1);
        assert_eq!(width, 16 + 2 + 5 - 1);

        let data = labels.into_data().convert::<i32>();
        let values = data.as_slice::<i32>().unwrap();
        let start = 16 + 2 + 4 - 1;
        assert!(values[..start].iter().all(|value| *value == -100));
        assert_eq!(values[start], 30);
        assert!(values[start + 1..].iter().all(|value| *value == -100));
    }

    #[test]
    fn vision_forward_produces_visual_tokens() {
        let device = Device::flex();
        let language = GLM53Flash::new(&language_config(), &device);
        let model = VisionLanguageModel::new(language, vision_config(), &device);

        let images = Tensor::<4>::zeros([2, 3, 32, 32], &device);
        let ids = vec![BOS as i64, 14, 40, 20];
        let text = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, 4]), &device);
        let text = Tensor::cat(vec![text.clone(), text], 0);

        let (logits, _) = model.forward(images, text);
        let [batch, time, _] = logits.dims();
        assert_eq!(batch, 2);
        assert_eq!(time, 4 + 18);
    }

    #[test]
    fn text_only_behaviour_is_unchanged_by_the_vision_path() {
        let device = Device::flex();

        // A 260-token language model, as the text-only port uses.
        let plain_config = ModelConfig::new()
            .with_dim(32)
            .with_heads(4)
            .with_layers(1)
            .with_experts(2)
            .with_expert_hidden(16)
            .with_streams(2)
            .with_sparse_window(8)
            .with_sparse_stride(8)
            .with_max_sequence_length(64);
        assert_eq!(plain_config.vocab_size, 260);

        let model = GLM53Flash::new(&plain_config, &device).valid();
        let ids: Vec<i64> = vec![BOS as i64, 14, 40, 20, 30];
        let text = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, 5]), &device);

        let (first, _) = model.clone().forward(text.clone());
        let (second, _) = model.forward(text);
        first.into_data().assert_approx_eq::<f32>(
            &second.into_data(),
            burn::tensor::Tolerance::rel_abs(1e-5, 1e-5),
        );
    }

    #[test]
    fn direct_patch_baseline_produces_sixteen_tokens() {
        let device = Device::flex();
        let language = GLM53Flash::new(&language_config(), &device);
        let model = DirectPatchVisionLanguageModel::new(language, 8, &device);
        assert_eq!(model.visual_token_count(), 16);

        let images = Tensor::<4>::zeros([1, 3, 32, 32], &device);
        let text =
            Tensor::<2, Int>::from_data(TensorData::new(vec![BOS as i64, 14], [1, 2]), &device);
        let (logits, _) = model.forward(images, text);
        assert_eq!(logits.dims()[1], 16 + 2);
    }
}
