//! Generated RGB seven-segment digits, and the text batch the vision path reads.
//!
//! Mirrors the generator in `scripts/train_vision.py`: 32x32 RGB, segments drawn
//! from fixed boxes, a per-image shift of up to two pixels, a label-independent
//! random colour, and additive noise. The rendering uses this crate's PRNG, so
//! the pixels differ from the reference's while the distribution does not.

use burn::prelude::*;

use crate::{tasks::Rng, tokenizer::ByteTokenizer};

/// Segment boxes: `(y0, y1, x0, x1)`.
const HORIZONTAL: [(u8, (usize, usize, usize, usize)); 3] = [
    (b'a', (4, 7, 8, 24)),
    (b'g', (14, 17, 8, 24)),
    (b'd', (25, 28, 8, 24)),
];
const VERTICAL: [(u8, (usize, usize, usize, usize)); 4] = [
    (b'f', (6, 15, 5, 8)),
    (b'b', (6, 15, 24, 27)),
    (b'e', (16, 26, 5, 8)),
    (b'c', (16, 26, 24, 27)),
];

/// Which segments each digit lights.
fn segments(digit: u8) -> &'static [u8] {
    match digit {
        0 => b"abcdef",
        1 => b"bc",
        2 => b"abged",
        3 => b"abgcd",
        4 => b"fgbc",
        5 => b"afgcd",
        6 => b"afgecd",
        7 => b"abc",
        8 => b"abcdefg",
        9 => b"abfgcd",
        _ => b"",
    }
}

fn segment_box(segment: u8) -> (usize, usize, usize, usize) {
    for (name, bounds) in HORIZONTAL.iter().chain(VERTICAL.iter()) {
        if *name == segment {
            return *bounds;
        }
    }
    panic!("unknown segment {}", segment as char);
}

/// Render `labels.len()` digits as `[n, 3, 32, 32]` float images in `[0, 1]`.
pub fn render_digits(labels: &[u8], seed: u64, noise: f32, device: &Device) -> Tensor<4> {
    const SIZE: usize = 32;
    let mut rng = Rng::new(seed);
    let mut pixels = vec![0.0f32; labels.len() * 3 * SIZE * SIZE];

    for (row, digit) in labels.iter().enumerate() {
        let shift_y = rng.below(5) as i32 - 2;
        let shift_x = rng.below(5) as i32 - 2;
        let colour = [
            0.55 + 0.45 * unit(&mut rng),
            0.55 + 0.45 * unit(&mut rng),
            0.55 + 0.45 * unit(&mut rng),
        ];
        let intensity = 0.80 + 0.20 * unit(&mut rng);

        for segment in segments(*digit) {
            let (y0, y1, x0, x1) = segment_box(*segment);
            let y0 = ((y0 as i32 + shift_y).max(0) as usize).min(SIZE);
            let y1 = ((y1 as i32 + shift_y).max(0) as usize).min(SIZE);
            let x0 = ((x0 as i32 + shift_x).max(0) as usize).min(SIZE);
            let x1 = ((x1 as i32 + shift_x).max(0) as usize).min(SIZE);

            for (channel, shade) in colour.iter().enumerate() {
                for y in y0..y1 {
                    for x in x0..x1 {
                        let index = ((row * 3 + channel) * SIZE + y) * SIZE + x;
                        pixels[index] = shade * intensity;
                    }
                }
            }
        }
    }

    for value in pixels.iter_mut() {
        *value = (*value + noise * unit(&mut rng)).clamp(0.0, 1.0);
    }

    Tensor::<4>::from_data(
        TensorData::new(pixels, [labels.len(), 3, SIZE, SIZE]),
        device,
    )
}

fn unit(rng: &mut Rng) -> f32 {
    // 24 bits of mantissa keeps this strictly inside [0, 1).
    (rng.below(1 << 24) as f32) / ((1 << 24) as f32)
}

/// The prompt the model answers from.
pub const PROMPT: &str = "Digit: ";

/// Encode `"Digit: <label>"` rows, returning the tokens and the prompt length.
pub fn text_batch(labels: &[u8], device: &Device) -> (Tensor<2, Int>, usize) {
    let tokenizer = ByteTokenizer::new();
    let prompt = tokenizer.encode(PROMPT, true, false);
    let prompt_length = prompt.len();

    let mut ids = Vec::with_capacity(labels.len() * (prompt_length + 1));
    for label in labels {
        ids.extend(prompt.iter().map(|token| *token as i64));
        ids.extend(
            tokenizer
                .encode(&label.to_string(), false, false)
                .iter()
                .map(|token| *token as i64),
        );
    }

    let width = ids.len() / labels.len();
    (
        Tensor::<2, Int>::from_data(TensorData::new(ids, [labels.len(), width]), device),
        prompt_length,
    )
}

/// Encode the prompt alone, for held-out evaluation.
pub fn prompt_batch(examples: usize, device: &Device) -> (Tensor<2, Int>, usize) {
    let tokenizer = ByteTokenizer::new();
    let prompt = tokenizer.encode(PROMPT, true, false);
    let width = prompt.len();
    let ids: Vec<i64> = (0..examples)
        .flat_map(|_| prompt.iter().map(|token| *token as i64))
        .collect();
    (
        Tensor::<2, Int>::from_data(TensorData::new(ids, [examples, width]), device),
        width,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendering_is_deterministic() {
        let device = Device::flex();
        let first = render_digits(&[0, 1, 2], 7, 0.06, &device);
        let second = render_digits(&[0, 1, 2], 7, 0.06, &device);
        first
            .into_data()
            .assert_approx_eq::<f32>(&second.into_data(), burn::tensor::Tolerance::strict());
    }

    #[test]
    fn rendering_stays_in_range_and_shapes() {
        let device = Device::flex();
        let images = render_digits(&[3, 8], 11, 0.06, &device);
        assert_eq!(images.dims(), [2, 3, 32, 32]);
        let data = images.into_data();
        for value in data.as_slice::<f32>().unwrap() {
            assert!((0.0..=1.0).contains(value), "pixel {value} out of range");
        }
    }

    #[test]
    fn different_digits_render_differently() {
        let device = Device::flex();
        // Same seed, so shift and colour match; only the lit segments differ.
        let one = render_digits(&[1], 3, 0.0, &device).into_data();
        let eight = render_digits(&[8], 3, 0.0, &device).into_data();
        let a = one.as_slice::<f32>().unwrap();
        let b = eight.as_slice::<f32>().unwrap();
        assert!(a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn the_text_batch_carries_one_answer_byte() {
        let device = Device::flex();
        let (tokens, prompt_length) = text_batch(&[0, 7, 9], &device);
        let [batch, width] = tokens.dims();
        assert_eq!(batch, 3);
        assert_eq!(width, prompt_length + 1);

        let data = tokens.into_data().convert::<i32>();
        let values = data.as_slice::<i32>().unwrap();
        assert_eq!(values[prompt_length], 4 + b'0' as i32);
        assert_eq!(values[width + prompt_length], 4 + b'7' as i32);
    }

    #[test]
    fn the_prompt_batch_matches_the_training_prefix() {
        let device = Device::flex();
        let (prompt, prompt_length) = prompt_batch(4, &device);
        let (tokens, training_prompt_length) = text_batch(&[1, 2, 3, 4], &device);
        assert_eq!(prompt_length, training_prompt_length);
        assert_eq!(prompt.dims(), [4, prompt_length]);
        assert_eq!(tokens.dims()[1], prompt_length + 1);
    }

    #[test]
    fn every_digit_lights_at_least_one_segment() {
        for digit in 0..10u8 {
            assert!(!segments(digit).is_empty(), "digit {digit} is blank");
        }
    }
}
