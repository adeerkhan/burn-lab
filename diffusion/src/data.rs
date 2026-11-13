//! A tiny synthetic image dataset: eight coloured shapes on a black canvas.
//!
//! The reference trains on CIFAR-10 through `torchvision`. That download is not
//! part of this example: eight templates are enough to show that the diffusion
//! model learns a distribution, and they keep the example self-contained.

use burn::prelude::*;

/// Deterministic SplitMix64, so the crate has no RNG dependency.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// The eight shapes, drawn into an RGB buffer of `size x size`.
pub fn template(index: usize, size: usize) -> Vec<f32> {
    let mut pixels = vec![0.0f32; 3 * size * size];
    let put = |pixels: &mut Vec<f32>, x: usize, y: usize, colour: [f32; 3]| {
        if x < size && y < size {
            let at = (y * size + x) * 3;
            pixels[at] = colour[0];
            pixels[at + 1] = colour[1];
            pixels[at + 2] = colour[2];
        }
    };

    let palette = [
        [1.0, 0.2, 0.2],
        [0.2, 0.9, 0.3],
        [0.2, 0.4, 1.0],
        [1.0, 0.9, 0.1],
        [0.9, 0.2, 0.9],
        [0.1, 0.9, 0.9],
        [1.0, 0.5, 0.1],
        [0.9, 0.9, 0.9],
    ];
    let colour = palette[index % palette.len()];
    let margin = size / 8;
    let start = margin;
    let end = size - margin;
    let mid = size / 2;

    match index % 8 {
        // Horizontal bar.
        0 => {
            for y in mid - 2..mid + 2 {
                for x in start..end {
                    put(&mut pixels, x, y, colour);
                }
            }
        }
        // Vertical bar.
        1 => {
            for x in mid - 2..mid + 2 {
                for y in start..end {
                    put(&mut pixels, x, y, colour);
                }
            }
        }
        // Square.
        2 => {
            for y in start..end {
                for x in start..end {
                    put(&mut pixels, x, y, colour);
                }
            }
        }
        // Cross.
        3 => {
            for y in start..end {
                for x in mid - 1..mid + 1 {
                    put(&mut pixels, x, y, colour);
                }
            }
            for y in mid - 1..mid + 1 {
                for x in start..end {
                    put(&mut pixels, x, y, colour);
                }
            }
        }
        // Diagonal.
        4 => {
            for index in start..end {
                put(&mut pixels, index, index, colour);
                put(&mut pixels, size - 1 - index, index, colour);
            }
        }
        // Checkerboard.
        5 => {
            for y in start..end {
                for x in start..end {
                    if (x / 4 + y / 4) % 2 == 0 {
                        put(&mut pixels, x, y, colour);
                    }
                }
            }
        }
        // Ring.
        6 => {
            let radius = (size as f32) * 0.3;
            for y in 0..size {
                for x in 0..size {
                    let dx = x as f32 - mid as f32;
                    let dy = y as f32 - mid as f32;
                    let distance = (dx * dx + dy * dy).sqrt();
                    if (distance - radius).abs() < 2.0 {
                        put(&mut pixels, x, y, colour);
                    }
                }
            }
        }
        // Inverted square border.
        _ => {
            for y in start..end {
                for x in start..end {
                    if x < start + 2 || x >= end - 2 || y < start + 2 || y >= end - 2 {
                        put(&mut pixels, x, y, colour);
                    }
                }
            }
        }
    }
    pixels
}

/// A batch of shape images in `[-1, 1]`.
pub fn batch(batch_size: usize, size: usize, seed: u64, device: &Device) -> Tensor<4> {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(batch_size * 3 * size * size);
    for _ in 0..batch_size {
        let pixels = template(rng.below(8), size);
        data.extend(pixels.iter().map(|value| value * 2.0 - 1.0));
    }
    Tensor::<4>::from_data(TensorData::new(data, [batch_size, 3, size, size]), device)
}

/// Distance from each sample to the closest template, in pixel space.
pub fn nearest_template_distance(samples: Tensor<4>, size: usize) -> f32 {
    let [batch, _, _, _] = samples.dims();
    let flat = samples
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    let templates: Vec<Vec<f32>> = (0..8)
        .map(|index| {
            template(index, size)
                .iter()
                .map(|v| v * 2.0 - 1.0)
                .collect()
        })
        .collect();

    let mut total = 0.0f32;
    let mut count = 0usize;
    for index in 0..batch {
        let offset = index * 3 * size * size;
        let sample = &flat[offset..offset + 3 * size * size];
        let mut best = f32::MAX;
        for template in &templates {
            let distance: f32 = sample
                .iter()
                .zip(template)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / sample.len() as f32;
            best = best.min(distance);
        }
        total += best;
        count += 1;
    }
    total / count as f32
}

/// A flat file of `count` RGB images of `size x size`, packed as u8.
pub struct RawImages {
    data: Vec<u8>,
    size: usize,
    count: usize,
}

impl RawImages {
    pub fn load(path: &str, size: usize) -> Self {
        let data = std::fs::read(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
        let stride = 3 * size * size;
        assert!(
            data.len().is_multiple_of(stride),
            "{path} is not a whole number of {size}x{size} RGB images"
        );
        Self {
            count: data.len() / stride,
            data,
            size,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// A batch of random images in `[-1, 1]`.
    pub fn batch(&self, batch: usize, seed: u64, device: &Device) -> Tensor<4> {
        let mut rng = Rng::new(seed);
        let stride = 3 * self.size * self.size;
        let plane = self.size * self.size;
        let mut pixels = Vec::with_capacity(batch * stride);
        for _ in 0..batch {
            let index = rng.below(self.count);
            let offset = index * stride;
            // The file is interleaved RGB (`RGBRGB...`) but a tensor is
            // `[channels, height, width]`, so the planes have to be separated.
            // Copying the bytes straight through would hand the model an
            // interleaved permutation of the image.
            for channel in 0..3 {
                for pixel in 0..plane {
                    pixels.push(self.data[offset + 3 * pixel + channel] as f32 / 127.5 - 1.0);
                }
            }
        }
        Tensor::<4>::from_data(
            TensorData::new(pixels, [batch, 3, self.size, self.size]),
            device,
        )
    }

    /// Mean L1 distance from each sample to the closest of `subset` training
    /// images. A rough quality proxy when there are no templates.
    pub fn nearest_training_distance(&self, samples: Tensor<4>, subset: usize, seed: u64) -> f32 {
        let [batch, _, _, _] = samples.dims();
        let stride = 3 * self.size * self.size;
        let flat = samples
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        // Draw *distinct* references: with replacement, a "256-image" subset can
        // be much smaller in practice, which makes the score a noisy random
        // variable and weakens the proxy.
        let mut rng = Rng::new(seed);
        let wanted = subset.min(self.count);
        let mut taken = vec![false; self.count];
        let mut indices = Vec::with_capacity(wanted);
        while indices.len() < wanted {
            let index = rng.below(self.count);
            if !taken[index] {
                taken[index] = true;
                indices.push(index);
            }
        }
        let references: Vec<&[u8]> = indices
            .iter()
            .map(|index| &self.data[index * stride..(index + 1) * stride])
            .collect();

        let plane = self.size * self.size;
        let mut total = 0.0f32;
        for index in 0..batch {
            let sample = &flat[index * stride..(index + 1) * stride];
            let mut best = f32::MAX;
            for reference in &references {
                // Samples are planar, references are interleaved.
                let mut sum = 0.0f32;
                for pixel in 0..plane {
                    for channel in 0..3 {
                        let value = reference[3 * pixel + channel] as f32 / 127.5 - 1.0;
                        sum += (sample[channel * plane + pixel] - value).abs();
                    }
                }
                best = best.min(sum / stride as f32);
            }
            total += best;
        }
        total / batch as f32
    }
}
