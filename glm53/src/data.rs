//! Dataset and batching for next-byte prediction.

use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::{tasks, tokenizer::ByteTokenizer};

/// One training document: the token ids of a synthetic Python snippet.
#[derive(Clone, Debug)]
pub struct Document {
    pub tokens: Vec<u32>,
}

/// A corpus of synthetic documents, deterministic in `seed`.
#[derive(Clone, Debug)]
pub struct Corpus {
    documents: Vec<Document>,
}

impl Corpus {
    /// Build a corpus of `count` documents.
    pub fn new(count: usize, seed: u64) -> Self {
        let tokenizer = ByteTokenizer::new();
        let documents = (0..count)
            .map(|index| {
                let text = tasks::pretraining_text(index, seed);
                Document {
                    tokens: tokenizer.encode(&text, true, true),
                }
            })
            .collect();

        Self { documents }
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&Document> {
        self.documents.get(index)
    }
}

/// A batch of token ids with the shifted targets.
#[derive(Clone, Debug)]
pub struct ByteBatch {
    pub tokens: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
}

/// Packs documents into fixed-length sequences.
#[derive(Clone, Default)]
pub struct ByteBatcher {
    pub sequence_length: usize,
}

impl ByteBatcher {
    pub fn new(sequence_length: usize) -> Self {
        Self { sequence_length }
    }
}

impl Batcher<Document, ByteBatch> for ByteBatcher {
    fn batch(&self, items: Vec<Document>, device: &Device) -> ByteBatch {
        let width = self.sequence_length.saturating_sub(1);
        let mut rows: Vec<Vec<i64>> = Vec::with_capacity(items.len());
        let mut targets: Vec<Vec<i64>> = Vec::with_capacity(items.len());

        for item in items {
            let mut tokens = item.tokens.clone();
            // Truncate or pad to the fixed sequence length.
            tokens.truncate(self.sequence_length);
            while tokens.len() < self.sequence_length {
                tokens.push(0);
            }
            if width == 0 {
                continue;
            }

            // Next-byte prediction: the target at position i is the token at i+1.
            let inputs: Vec<i64> = tokens[..width].iter().map(|t| *t as i64).collect();
            let outputs: Vec<i64> = tokens[1..=width].iter().map(|t| *t as i64).collect();
            rows.push(inputs);
            targets.push(outputs);
        }

        let shape = [rows.len(), width];
        let flat_tokens: Vec<i64> = rows.into_iter().flatten().collect();
        let flat_targets: Vec<i64> = targets.into_iter().flatten().collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(flat_tokens, shape), device);
        let targets = Tensor::<2, Int>::from_data(TensorData::new(flat_targets, shape), device);

        ByteBatch { tokens, targets }
    }
}
