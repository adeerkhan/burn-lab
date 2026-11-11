//! A deterministic byte tokenizer: one token per UTF-8 byte.
//!
//! Mirrors `glm53_flash/tokenizer.py`. Special tokens occupy ids 0..3 and byte
//! `b` is token `b + 4`.

pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const SEP_ID: u32 = 3;
pub const BYTE_OFFSET: u32 = 4;
pub const VOCAB_SIZE: usize = 260;

#[derive(Clone, Debug)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Encode text as byte tokens.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut ids: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|b| *b as u32 + BYTE_OFFSET)
            .collect();
        if bos {
            ids.insert(0, BOS_ID);
        }
        if eos {
            ids.push(EOS_ID);
        }
        ids
    }

    /// Decode byte tokens back into text, stopping at the first EOS.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len());
        for &token in ids {
            if token == EOS_ID {
                break;
            }
            if token >= BYTE_OFFSET {
                bytes.push((token - BYTE_OFFSET) as u8);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ascii() {
        let tokenizer = ByteTokenizer::new();
        let text = "def increment(x):\n    return x + 1\n";
        let ids = tokenizer.encode(text, true, true);
        assert_eq!(ids[0], BOS_ID);
        assert_eq!(*ids.last().unwrap(), EOS_ID);
        assert_eq!(tokenizer.decode(&ids), text);
    }

    #[test]
    fn encodes_ascii_as_seventeen_tokens() {
        // The reference README states `def increment(x):` takes 17 byte tokens.
        let tokenizer = ByteTokenizer::new();
        let ids = tokenizer.encode("def increment(x):", false, false);
        assert_eq!(ids.len(), 17);
    }

    #[test]
    fn stops_at_eos() {
        let tokenizer = ByteTokenizer::new();
        let ids = tokenizer.encode("hi", false, true);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hi");
    }
}
