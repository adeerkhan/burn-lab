//! Peak GPU memory sampling.
//!
//! Burn does not expose allocator statistics, so this polls `nvidia-smi` in the
//! background and keeps the highest reading. The number covers the whole device,
//! including whatever else is running, so it is reported alongside the baseline
//! taken when the sampler starts.

use std::{
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// A background sampler for peak used GPU memory, in MiB.
pub struct PeakMemory {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<u64>>,
    baseline: Option<u64>,
}

impl PeakMemory {
    /// Start sampling. Returns `None` inside if `nvidia-smi` is unavailable.
    pub fn start() -> Self {
        let baseline = read_used_mib();
        let Some(baseline_value) = baseline else {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
                baseline: None,
            };
        };

        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let mut peak = baseline_value;
            while !flag.load(Ordering::Relaxed) {
                if let Some(used) = read_used_mib() {
                    peak = peak.max(used);
                }
                thread::sleep(POLL_INTERVAL);
            }
            peak
        });

        Self {
            stop,
            handle: Some(handle),
            baseline: Some(baseline_value),
        }
    }

    /// Stop sampling and return `(baseline, peak)` in MiB, or `None`.
    pub fn stop(mut self) -> Option<(u64, u64)> {
        self.stop.store(true, Ordering::Relaxed);
        let baseline = self.baseline?;
        let peak = self.handle.take()?.join().ok()?;
        Some((baseline, peak))
    }
}

/// A short human-readable summary, e.g. `peak 3120 MiB (+2871)`.
pub fn describe(memory: Option<(u64, u64)>) -> String {
    match memory {
        Some((baseline, peak)) => {
            format!(
                "peak {peak} MiB of GPU memory ({:+} over baseline)",
                peak as i64 - baseline as i64
            )
        }
        None => "peak GPU memory: not measured (no nvidia-smi)".to_string(),
    }
}

fn read_used_mib() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().next()?.trim().parse::<u64>().ok()
}
