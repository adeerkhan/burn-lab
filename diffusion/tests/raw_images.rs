//! Tests for the flat-file image loader that feeds the flowers runs.

use burn::prelude::*;
use diffusion::data::{RawImages, Rng};

fn write_images(path: &std::path::Path, images: &[[u8; 12]], size: usize) {
    let mut bytes = Vec::new();
    for image in images {
        assert_eq!(image.len(), 3 * size * size);
        bytes.extend_from_slice(image);
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn a_batch_is_seeded_and_inside_the_pixel_range() {
    let directory = std::env::temp_dir().join("diffusion-raw-batch");
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("two.bin");
    // Two 2x2 RGB images: black and white.
    write_images(&path, &[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [255; 12]], 2);

    let images = RawImages::load(path.to_str().unwrap(), 2);
    assert_eq!(images.len(), 2);
    assert!(!images.is_empty());

    let device = Device::flex();
    let first = images.batch(8, 42, &device);
    let second = images.batch(8, 42, &device);
    assert_eq!(first.dims(), [8, 3, 2, 2]);

    let values = first
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    // Every pixel is a corner of the cube: -1 or +1.
    assert!(values.iter().all(|value| value.abs() == 1.0));

    // The same seed replays the same batch, which is what makes resume exact.
    let replay = second
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(values, replay);
}

#[test]
fn the_nearest_distance_uses_the_closest_reference() {
    let directory = std::env::temp_dir().join("diffusion-raw-nearest");
    std::fs::create_dir_all(&directory).unwrap();
    let device = Device::flex();

    // The loader maps u8 to [-1, 1], so 128 becomes 128/127.5 - 1.
    let grey = 128.0f32 / 127.5 - 1.0;

    // A dataset holding only the middle grey image: a grey sample must be
    // exactly zero away, however many references are drawn.
    let path = directory.join("grey.bin");
    write_images(&path, &[[128; 12]], 2);
    let images = RawImages::load(path.to_str().unwrap(), 2);
    let middle = Tensor::<4>::from_data(
        burn::tensor::TensorData::new(vec![grey; 12], [1, 3, 2, 2]),
        &device,
    );
    let distance = images.nearest_training_distance(middle, 8, 7);
    assert!(distance < 1e-6, "expected zero, got {distance}");

    // The same sample against black and white references must be the distance
    // to the closer one, not the mean.
    let path = directory.join("corners.bin");
    write_images(&path, &[[0; 12], [255; 12]], 2);
    let images = RawImages::load(path.to_str().unwrap(), 2);
    let middle = Tensor::<4>::from_data(
        burn::tensor::TensorData::new(vec![grey; 12], [1, 3, 2, 2]),
        &device,
    );
    let distance = images.nearest_training_distance(middle, 8, 7);
    let expected = (1.0 - grey).abs();
    assert!(
        (distance - expected).abs() < 1e-6,
        "expected {expected}, got {distance}"
    );
}

#[test]
fn the_rng_is_reproducible_and_bounded() {
    let mut first = Rng::new(99);
    let mut second = Rng::new(99);
    for _ in 0..16 {
        assert_eq!(first.next_u64(), second.next_u64());
    }

    // `below` must stay in range for every bound, including zero.
    let mut rng = Rng::new(7);
    for bound in [0usize, 1, 7, 11, 100] {
        for _ in 0..32 {
            assert!(rng.below(bound) < bound.max(1));
        }
    }
}

#[test]
fn a_batch_separates_the_colour_planes() {
    let directory = std::env::temp_dir().join("diffusion-raw-layout");
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("quad.bin");

    // One 2x2 image, interleaved as a BMP/PIL would write it:
    // red, green, blue, white pixels.
    let interleaved = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
    std::fs::write(&path, interleaved).unwrap();

    let images = RawImages::load(path.to_str().unwrap(), 2);
    let device = Device::flex();
    let batch = images.batch(1, 3, &device);
    let flat = batch
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap();

    // [channels, height, width]: the red plane holds pixel 0 and pixel 3.
    assert_eq!(
        flat,
        vec![
            1.0, -1.0, -1.0, 1.0, // red
            -1.0, 1.0, -1.0, 1.0, // green
            -1.0, -1.0, 1.0, 1.0, // blue
        ]
    );
}
