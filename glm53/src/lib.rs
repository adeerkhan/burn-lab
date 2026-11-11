//! A Burn port of `vukrosic/glm-5.3-flash-from-scratch`.
//!
//! The reference is a 25.7M-parameter GLM-5.3-Flash-inspired language model:
//! hybrid linear/sparse attention, a sparse mixture of experts, and four
//! residual streams, trained from scratch on a synthetic Python corpus.
//!
//! This crate reimplements the model and its data pipeline on Burn, so the
//! framework can be exercised on real language-model work rather than on CNNs.

pub mod analysis;
pub mod arithmetic;
pub mod attention;
pub mod block;
pub mod config;
pub mod data;
pub mod evaluation;
pub mod evaluator;
pub mod experiments;
pub mod generate;
pub mod metrics;
pub mod model;
pub mod moe;
pub mod receipt;
pub mod rl;
pub mod tasks;
pub mod tokenizer;
pub mod training;
pub mod vision;
pub mod vision_data;
pub mod vision_pilot;
pub mod vision_train;

pub use config::ModelConfig;
pub use model::GLM53Flash;
pub use tokenizer::ByteTokenizer;
