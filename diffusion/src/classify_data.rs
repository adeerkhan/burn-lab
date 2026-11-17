//! A streaming loader for the flat u8 classification datasets prepared by
//! `scripts/prepare_classification.py`.
//!
//! The images stay on disk: `batch` seeks to random rows and reads only one
//! batch at a time, so the process never holds the whole dataset in memory.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use burn::prelude::*;

pub struct LabeledImages {
    images: File,
    labels: File,
    size: usize,
    count: usize,
}

impl LabeledImages {
    pub fn load(images: &str, labels: &str, size: usize) -> Self {
        let images = File::open(images).expect("open the image file");
        let labels = File::open(labels).expect("open the label file");
        let count = labels.metadata().expect("label file metadata").len() as usize;
        Self {
            images,
            labels,
            size,
            count,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// One batch of images in `[-1, 1]` and their class ids.
    pub fn batch(
        &mut self,
        batch: usize,
        seed: u64,
        device: &Device,
    ) -> (Tensor<4>, Tensor<2, Int>) {
        let stride = 3 * self.size * self.size;
        let mut rng = crate::data::Rng::new(seed);
        let mut pixels = Vec::with_capacity(batch * stride);
        let mut classes = Vec::with_capacity(batch);

        for _ in 0..batch {
            let index = rng.below(self.count);
            let mut raw = vec![0u8; stride];
            self.images
                .seek(SeekFrom::Start((index * stride) as u64))
                .expect("seek the image file");
            self.images.read_exact(&mut raw).expect("read one image");
            // The file is interleaved RGB, a tensor is [channels, height, width].
            let plane = self.size * self.size;
            for channel in 0..3 {
                for pixel in 0..plane {
                    pixels.push(raw[3 * pixel + channel] as f32 / 127.5 - 1.0);
                }
            }

            let mut label = [0u8; 1];
            self.labels
                .seek(SeekFrom::Start(index as u64))
                .expect("seek the label file");
            self.labels.read_exact(&mut label).expect("read one label");
            classes.push(label[0] as i64);
        }

        let images = Tensor::<4>::from_data(
            TensorData::new(pixels, [batch, 3, self.size, self.size]),
            device,
        );
        let labels = Tensor::<2, Int>::from_data(TensorData::new(classes, [batch, 1]), device);
        (images, labels)
    }
}
