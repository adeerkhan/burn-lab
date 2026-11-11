//! Forward-pass benchmark for the port.
//!
//! Reports milliseconds per forward and tokens per second for a fixed shape, so
//! the same measurement can be taken against the Python reference. Random
//! initialisation is fine: latency does not depend on the weight values.
//!
//! ```sh
//! cargo run -p glm53 --release --example bench_forward -- wgpu 16 128 20
//! ```

use std::time::Instant;

use burn::prelude::*;
use glm53::{GLM53Flash, ModelConfig};

fn main() {
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "wgpu".to_string());
    let batch: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
    let sequence: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(128);
    let iterations: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(20);

    let device = match backend.as_str() {
        "flex" => Device::flex(),
        "wgpu" => Device::wgpu(Default::default()),
        other => panic!("unknown backend `{other}`"),
    };

    let config = ModelConfig::new();
    assert!(
        sequence <= config.max_sequence_length,
        "sequence exceeds the {}-token context",
        config.max_sequence_length
    );

    let build_started = Instant::now();
    let model = GLM53Flash::new(&config, &device).valid();
    device.sync().expect("device sync");
    let build_seconds = build_started.elapsed().as_secs_f64();

    let tokens: Vec<i64> = (0..batch * sequence)
        .map(|index| 10 + (index % 60) as i64)
        .collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(tokens, [batch, sequence]), &device);

    // Warm up: first calls compile kernels and populate caches.
    for _ in 0..3 {
        let _ = model.forward(input.clone());
    }
    device.sync().expect("device sync");

    let started = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward(input.clone());
    }
    device.sync().expect("device sync");
    let seconds = started.elapsed().as_secs_f64();

    let per_forward = seconds / iterations as f64;
    let tokens_per_second = (batch * sequence) as f64 / per_forward;

    println!("backend      {backend}");
    println!("shape        batch {batch} x seq {sequence}");
    println!(
        "build        {:.1} ms (device + model init)",
        build_seconds * 1e3
    );
    println!("forward      {:.1} ms", per_forward * 1e3);
    println!("throughput   {:.0} tokens/s", tokens_per_second);
}
