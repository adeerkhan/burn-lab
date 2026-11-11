//! Inference: load a trained record and run the model.
//!
//! Parameters are lazily initialized, so `config.init(device).load_record(record)`
//! costs nothing beyond the load itself.

use std::path::{Path, PathBuf};

use crate::{data::MnistBatcher, model::Model, training::TrainingConfig};
use burn::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::vision::MnistItem,
    },
    prelude::*,
    store::ModuleRecord,
    train::{EvaluatorBuilder, EvaluatorComponentTypesMarker, metric::AccuracyMetric},
};

/// Load the trained model from `artifact_dir`.
pub fn load_model(artifact_dir: &str, device: &Device) -> Model {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = ModuleRecord::load(format!("{artifact_dir}/model"))
        .expect("Trained model should exist; run train first");

    config.model.init(device).load_record(record)
}

/// Predict the digit of a single MNIST image.
pub fn infer(artifact_dir: &str, device: impl Into<Device>, item: MnistItem) {
    let device = device.into();
    let model = load_model(artifact_dir, &device);

    let label = item.label;
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted: u8 = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {predicted} Expected {label}");
}

/// Evaluate the trained model over the whole MNIST test set.
///
/// Uses Burn's `Evaluator`, the counterpart of `SupervisedTraining` for test
/// splits. It writes the per-batch accuracy into the artifact directory;
/// [`aggregate_accuracy`] folds that into a single number.
pub fn evaluate(artifact_dir: &str, device: impl Into<Device>) {
    let device = device.into();
    let model = load_model(artifact_dir, &device);

    let dataloader = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(64)
        .shuffle(42)
        .num_workers(4)
        .build(burn::data::dataset::vision::MnistDataset::test());

    let evaluator = EvaluatorBuilder::<EvaluatorComponentTypesMarker<Model>>::new(artifact_dir)
        .metrics((AccuracyMetric::new(),))
        .summary()
        .build(model);

    let result = evaluator.eval("test", dataloader);

    if let Some(error) = result.error {
        eprintln!("Evaluation failed: {error}");
        std::process::exit(1);
    }

    match aggregate_accuracy(artifact_dir) {
        Some(accuracy) => println!("\nTest accuracy: {accuracy:.3}%"),
        None => eprintln!("\nNo accuracy metric was written to {artifact_dir}"),
    }
}

/// The metric logs the evaluator wrote, one per split.
fn accuracy_logs(artifact_dir: &str) -> Vec<PathBuf> {
    let root = Path::new(artifact_dir).join("test");
    let mut files = Vec::new();

    let Ok(entries) = std::fs::read_dir(&root) else {
        return files;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Ok(inner) = std::fs::read_dir(&path) {
                files.extend(inner.flatten().map(|e| e.path()));
            }
        } else {
            files.push(path);
        }
    }

    files
        .into_iter()
        .filter(|path| path.file_name().and_then(|n| n.to_str()) == Some("Accuracy.log"))
        .collect()
}

/// Fold the per-batch accuracy the evaluator logged into one number.
///
/// Each line is `value,count`, where `count` is the number of items in that
/// batch, so the overall accuracy is the count-weighted mean.
fn aggregate_accuracy(artifact_dir: &str) -> Option<f64> {
    let mut weighted_sum = 0.0;
    let mut total = 0.0;

    for path in accuracy_logs(artifact_dir) {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };

        for line in content.lines().filter(|line| !line.trim().is_empty()) {
            let mut parts = line.split(',');
            let value: f64 = parts.next()?.parse().ok()?;
            let count: f64 = parts.next().unwrap_or("1").parse().ok()?;
            weighted_sum += value * count;
            total += count;
        }
    }

    (total > 0.0).then(|| weighted_sum / total)
}
