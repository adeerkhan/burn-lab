//! burn-lab: a scratchpad for exercising the Burn checkout in `../burn`.
//!
//! The commands mirror the Burn Book's basic workflow: train a CNN on MNIST,
//! evaluate it on the test set, and run inference on a single image.

mod data;
mod device;
mod inference;
mod model;
mod training;

use burn::{data::dataloader::Dataset, optim::AdamConfig};
use device::BACKENDS;
use model::ModelConfig;
use training::TrainingConfig;

const USAGE: &str = "\
burn-lab — train and evaluate a Burn CNN on MNIST

USAGE:
    burn-lab <COMMAND> [OPTIONS]

COMMANDS:
    train     Train the model and save it to the artifact directory
    test      Evaluate the saved model on the MNIST test set
    infer     Predict the digit of a single MNIST test image
    print     Print the model structure

OPTIONS:
    --backend <NAME>       Backend to run on: wgpu | flex   (default: wgpu)
    --epochs <N>           Number of training epochs        (default: 10)
    --batch-size <N>       Batch size                       (default: 64)
    --learning-rate <F>    Learning rate                    (default: 0.0001)
    --artifact-dir <PATH>  Where to save config/model/logs  (default: artifacts/mnist)
    --index <N>            MNIST test image index for infer (default: 42)
    -h, --help             Show this message
";

#[derive(Debug)]
struct Options {
    backend: String,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    artifact_dir: String,
    index: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backend: "wgpu".to_string(),
            epochs: None,
            batch_size: None,
            learning_rate: None,
            artifact_dir: "artifacts/mnist".to_string(),
            index: 42,
        }
    }
}

impl Options {
    fn parse(args: &[String]) -> Self {
        let mut opts = Options::default();
        let mut i = 0;

        while i < args.len() {
            let key = args[i].as_str();
            let value = |i: &mut usize| -> String {
                *i += 1;
                args.get(*i)
                    .cloned()
                    .unwrap_or_else(|| panic!("missing value for `{key}`"))
            };

            match key {
                "--backend" => opts.backend = value(&mut i),
                "--epochs" => {
                    opts.epochs = Some(value(&mut i).parse().expect("invalid `--epochs`"))
                }
                "--batch-size" => {
                    opts.batch_size = Some(value(&mut i).parse().expect("invalid `--batch-size`"))
                }
                "--learning-rate" => {
                    opts.learning_rate =
                        Some(value(&mut i).parse().expect("invalid `--learning-rate`"))
                }
                "--artifact-dir" => opts.artifact_dir = value(&mut i),
                "--index" => opts.index = value(&mut i).parse().expect("invalid `--index`"),
                other => panic!("unknown option `{other}`\n\n{USAGE}"),
            }

            i += 1;
        }

        opts
    }

    /// The training configuration, with the CLI overrides applied.
    fn training_config(&self) -> TrainingConfig {
        let mut config = TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new());

        if let Some(epochs) = self.epochs {
            config = config.with_num_epochs(epochs);
        }
        if let Some(batch_size) = self.batch_size {
            config = config.with_batch_size(batch_size);
        }
        if let Some(learning_rate) = self.learning_rate {
            config = config.with_learning_rate(learning_rate);
        }

        config
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "train".to_string());
    let rest: Vec<String> = args.collect();

    if command == "-h" || command == "--help" || command == "help" {
        println!("{USAGE}");
        return;
    }

    let opts = Options::parse(&rest);
    assert!(
        BACKENDS.contains(&opts.backend.as_str()),
        "unknown backend `{}`, expected one of {BACKENDS:?}",
        opts.backend
    );

    let device = device::device(&opts.backend);

    match command.as_str() {
        "train" => {
            let config = opts.training_config();
            println!("Backend: {}  Device: {device:?}", opts.backend);
            let model = training::train(&opts.artifact_dir, config, device);
            println!("\nTrained model saved to {}", opts.artifact_dir);
            println!("{model}");
        }
        "test" => {
            println!("Backend: {}  Device: {device:?}", opts.backend);
            inference::evaluate(&opts.artifact_dir, device);
        }
        "infer" => {
            let item = burn::data::dataset::vision::MnistDataset::test()
                .get(opts.index)
                .expect("MNIST test image should exist");
            inference::infer(&opts.artifact_dir, device, item);
        }
        "print" => {
            let model = ModelConfig::new(10, 512).init(&device);
            println!("{model}");
        }
        other => {
            eprintln!("unknown command `{other}`\n\n{USAGE}");
            std::process::exit(1);
        }
    }
}
