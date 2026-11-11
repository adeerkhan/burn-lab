//! Checkpoint receipts and the system receipt.
//!
//! Mirrors the receipt discipline of the reference's `runtime.py` and
//! `system_receipt.py`: a checkpoint directory carries its configuration, its
//! metadata and a sha256 of the weights, and loading verifies the hash.

use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The Burn release this port is written against. Mirrors `Cargo.toml`.
pub const BURN_VERSION: &str = "0.23.0-pre.2";

/// What a checkpoint directory records about itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointReceipt {
    pub schema_version: String,
    pub stage: String,
    pub step: usize,
    pub created_unix: u64,
    pub weights: String,
    pub weights_sha256: String,
    pub weights_bytes: u64,
    /// Free-form fields: source checkpoint, seed, hyperparameters.
    #[serde(default)]
    pub notes: BTreeMap<String, String>,
}

/// Why a checkpoint failed verification.
#[derive(Debug)]
pub enum ReceiptError {
    Missing(String),
    Mismatch { recorded: String, actual: String },
    Io(String),
}

impl std::fmt::Display for ReceiptError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(path) => write!(formatter, "missing {path}"),
            Self::Mismatch { recorded, actual } => {
                write!(
                    formatter,
                    "weights hash mismatch: recorded {recorded}, actual {actual}"
                )
            }
            Self::Io(message) => write!(formatter, "io error: {message}"),
        }
    }
}

impl std::error::Error for ReceiptError {}

/// The weight file name Burn writes for a saved record.
pub const WEIGHTS: &str = "model.bpk";
/// The receipt file name.
pub const RECEIPT: &str = "metadata.json";

/// sha256 of a file, in lowercase hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Write the receipt for a checkpoint directory that already holds weights.
///
/// # Errors
///
/// Returns an error if the weights are missing or cannot be read.
pub fn write_receipt(
    directory: &Path,
    stage: &str,
    step: usize,
    notes: BTreeMap<String, String>,
) -> Result<CheckpointReceipt, ReceiptError> {
    let weights = directory.join(WEIGHTS);
    if !weights.exists() {
        return Err(ReceiptError::Missing(weights.display().to_string()));
    }
    let hash = sha256_file(&weights).map_err(|error| ReceiptError::Io(error.to_string()))?;
    let bytes = fs::metadata(&weights)
        .map_err(|error| ReceiptError::Io(error.to_string()))?
        .len();

    let receipt = CheckpointReceipt {
        schema_version: "1.0".to_string(),
        stage: stage.to_string(),
        step,
        created_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0),
        weights: WEIGHTS.to_string(),
        weights_sha256: hash,
        weights_bytes: bytes,
        notes,
    };

    let text = serde_json::to_string_pretty(&receipt).expect("serialise the receipt");
    fs::write(directory.join(RECEIPT), text + "\n")
        .map_err(|error| ReceiptError::Io(error.to_string()))?;

    Ok(receipt)
}

/// Read a receipt, if the directory has one.
pub fn read_receipt(directory: &Path) -> Option<CheckpointReceipt> {
    let text = fs::read_to_string(directory.join(RECEIPT)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Verify a checkpoint directory against its receipt.
///
/// A directory with no receipt is an error: every checkpoint this port writes
/// carries one, so a missing receipt means the checkpoint is not ours to trust.
///
/// # Errors
///
/// Returns [`ReceiptError::Missing`] when the receipt or weights are absent and
/// [`ReceiptError::Mismatch`] when the hash does not match.
pub fn verify(directory: &Path) -> Result<CheckpointReceipt, ReceiptError> {
    let receipt = read_receipt(directory)
        .ok_or_else(|| ReceiptError::Missing(directory.join(RECEIPT).display().to_string()))?;

    let weights = directory.join(&receipt.weights);
    if !weights.exists() {
        return Err(ReceiptError::Missing(weights.display().to_string()));
    }

    let actual = sha256_file(&weights).map_err(|error| ReceiptError::Io(error.to_string()))?;
    if actual != receipt.weights_sha256 {
        return Err(ReceiptError::Mismatch {
            recorded: receipt.weights_sha256.clone(),
            actual,
        });
    }

    Ok(receipt)
}

/// A directory prepared for a checkpoint, plus its notes.
pub fn prepare(directory: &Path) -> Result<(), ReceiptError> {
    fs::create_dir_all(directory).map_err(|error| ReceiptError::Io(error.to_string()))
}

/// The path of the checkpoint for `step` inside a run directory.
pub fn checkpoint_path(run_directory: &str, step: usize) -> PathBuf {
    Path::new(run_directory).join(format!("checkpoint-{step:04}"))
}

/// Count the parameters that currently require a gradient.
pub fn num_trainable_params<M: burn::module::Module>(module: &M) -> usize {
    use burn::module::{ModuleVisitor, Param};

    #[derive(Default)]
    struct Counter {
        total: usize,
    }

    impl ModuleVisitor for Counter {
        fn visit_float<const D: usize>(&mut self, param: &Param<burn::prelude::Tensor<D>>) {
            if param.val().is_require_grad() {
                self.total += param.lazy_shape().num_elements();
            }
        }
    }

    let mut counter = Counter::default();
    module.visit(&mut counter);
    counter.total
}

/// Capture the runtime and hardware that produced a result.
pub fn system_receipt() -> serde_json::Value {
    let mut facts = serde_json::Map::new();
    facts.insert("os".into(), json_string(read_os()));
    facts.insert("cpu".into(), json_string(read_cpu()));
    facts.insert("cpu_cores".into(), json_number(num_cpus()));
    facts.insert("gpu".into(), json_string(read_gpu()));
    facts.insert(
        "rustc".into(),
        json_string(read_command("rustc", &["--version"])),
    );
    facts.insert(
        "cargo".into(),
        json_string(read_command("cargo", &["--version"])),
    );
    facts.insert("burn".into(), json_string(BURN_VERSION.to_string()));
    facts.insert(
        "glm53".into(),
        json_string(env!("CARGO_PKG_VERSION").to_string()),
    );
    facts.insert("profile".into(), json_string(profile()));
    serde_json::Value::Object(facts)
}

fn json_string(value: String) -> serde_json::Value {
    serde_json::Value::String(value)
}

fn json_number(value: usize) -> serde_json::Value {
    serde_json::Number::from(value as u64).into()
}

fn profile() -> String {
    if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        "release".to_string()
    }
}

fn read_os() -> String {
    if let Ok(text) = fs::read_to_string("/etc/os-release") {
        for line in text.lines() {
            if let Some(name) = line.strip_prefix("PRETTY_NAME=") {
                return name.trim_matches('"').to_string();
            }
        }
    }
    std::env::consts::OS.to_string()
}

fn read_cpu() -> String {
    if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some(name) = line.strip_prefix("model name")
                && let Some(value) = name.split(':').nth(1)
            {
                return value.trim().to_string();
            }
        }
    }
    "unknown".to_string()
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(0)
}

fn read_gpu() -> String {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "none".to_string(),
    }
}

fn read_command(program: &str, arguments: &[&str]) -> String {
    match Command::new(program).args(arguments).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join("glm53-receipt-tests").join(name);
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    #[test]
    fn a_receipt_round_trips() {
        let directory = scratch("round-trip");
        fs::write(directory.join(WEIGHTS), b"weights").unwrap();

        let mut notes = BTreeMap::new();
        notes.insert("seed".to_string(), "42".to_string());
        let written = write_receipt(&directory, "pretrain", 100, notes).unwrap();
        assert_eq!(written.step, 100);
        assert_eq!(written.weights_bytes, 7);

        let verified = verify(&directory).unwrap();
        assert_eq!(verified.weights_sha256, written.weights_sha256);
        assert_eq!(verified.notes.get("seed").unwrap(), "42");
    }

    #[test]
    fn a_tampered_checkpoint_is_a_hard_error() {
        let directory = scratch("tampered");
        fs::write(directory.join(WEIGHTS), b"weights").unwrap();
        write_receipt(&directory, "pretrain", 1, BTreeMap::new()).unwrap();

        fs::write(directory.join(WEIGHTS), b"tampered").unwrap();
        let error = verify(&directory).expect_err("a changed weight file must not verify");
        assert!(matches!(error, ReceiptError::Mismatch { .. }));
    }

    #[test]
    fn a_missing_receipt_is_an_error() {
        let directory = scratch("no-receipt");
        fs::write(directory.join(WEIGHTS), b"weights").unwrap();
        let error = verify(&directory).expect_err("no receipt means no trust");
        assert!(matches!(error, ReceiptError::Missing(_)));
    }

    #[test]
    fn the_system_receipt_names_the_toolchain() {
        let receipt = system_receipt();
        for key in ["os", "cpu", "rustc", "burn", "profile"] {
            assert!(receipt.get(key).is_some(), "missing {key}");
        }
        assert_eq!(
            receipt["burn"],
            serde_json::Value::String(BURN_VERSION.into())
        );
    }

    #[test]
    fn the_hash_matches_a_known_digest() {
        let directory = scratch("digest");
        let path = directory.join("payload");
        fs::write(&path, b"abc").unwrap();
        // sha256("abc")
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
