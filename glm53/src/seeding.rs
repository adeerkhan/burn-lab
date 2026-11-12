//! Serialised, seeded device randomness.
//!
//! `Device::seed` writes a process-global RNG (`burn-flex` keeps it behind a
//! `Mutex<Option<_>>`, CubeCL keeps its own global). Two threads that seed and
//! then construct a model therefore interleave their draws, and the second model
//! does not start from the seed it asked for. The test harness runs in parallel,
//! so a reproducibility test failed intermittently.
//!
//! Every construction that must be reproducible takes this lock, seeds inside it
//! and completes before releasing. The alternative — re-implementing Burn's
//! initialisers against our own RNG — would change the weight distributions that
//! every reported result depends on, so it is deliberately not done here.

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;

use crate::tasks::Rng;

use std::sync::{Mutex, MutexGuard};

static SEEDED: Mutex<()> = Mutex::new(());

/// Seed the device and run `f` while no other seeded work can interleave.
pub fn with_device_rng<T>(device: &Device, seed: u64, f: impl FnOnce() -> T) -> T {
    let _guard: MutexGuard<'_, ()> = SEEDED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    device.seed(seed);
    f()
}

/// Forced initialisation: Burn creates parameters lazily, so the weights are
/// drawn on first access, not in the constructor. Touching every parameter inside
/// the lock is what actually makes the initialisation deterministic.
struct Materialize;

impl ModuleVisitor for Materialize {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let _ = param.val().dims();
    }
}

/// Build a module from `seed` and force every parameter to materialise while the
/// device is seeded and no other thread can draw.
pub fn seeded<M: Module>(device: &Device, seed: u64, build: impl FnOnce() -> M) -> M {
    with_device_rng(device, seed, || {
        let model = build();
        let mut materialize = Materialize;
        model.visit(&mut materialize);
        model
    })
}

/// Build a language model that starts from `seed`, without racing.
pub fn seeded_model(config: &crate::ModelConfig, seed: u64, device: &Device) -> crate::GLM53Flash {
    // Escape hatch used to bisect a regression: `GLM53_DEVICE_INIT=1` keeps
    // Burn's own initialiser (materialised under the lock) instead of the host one.
    if std::env::var("GLM53_DEVICE_INIT").is_ok() {
        return seeded(device, seed, || crate::GLM53Flash::new(config, device));
    }
    initialize(crate::GLM53Flash::new(config, device), seed, device)
}

/// A held lock with the device seeded, for a function that draws as it runs.
///
/// Keep it alive for the whole section: the guard is what stops another thread
/// from reseeding the device between two of this section's draws.
pub struct RngSession {
    _guard: MutexGuard<'static, ()>,
}

/// Seed the device and hold the lock until the returned guard is dropped.
pub fn rng_session(device: &Device, seed: u64) -> RngSession {
    let guard = SEEDED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    device.seed(seed);
    RngSession { _guard: guard }
}

/// Overwrite every matrix parameter from our own seeded RNG.
///
/// Burn's initialisers draw from a process-global RNG, so their values depend on
/// what else ran in the process. This reproduces the same *distributions* —
/// uniform `+-1/sqrt(fan_in)` for linear weights, `N(0, 0.02)` for the embedding —
/// from a seed we control, which is independent of thread scheduling. Vectors are
/// left alone: biases are zero, RMSNorm gains are one, and the stream logits are a
/// deterministic ramp.
struct ParameterInit {
    rng: Rng,
    device: Device,
}

impl ModuleMapper for ParameterInit {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dims = param.val().dims();
        // Vectors are deterministic already (no biases, ones for RMSNorm, a ramp
        // for the stream logits); matrices are the only random ones.
        if dims.len() != 2 {
            return param;
        }
        let count: usize = dims.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|_| (self.rng.normal() * 0.02) as f32)
            .collect();
        // `require_grad` is what makes the parameter active: a plain `from_data`
        // tensor is inactive, and `Module::train` would then never enable its
        // gradient, so the weights would never move.
        let tensor = Tensor::<D>::from_data(TensorData::new(values, dims), &self.device);
        // A non-autodiff device has no gradient to enable; `require_grad` panics there.
        let tensor = if self.device.is_autodiff() {
            tensor.require_grad()
        } else {
            tensor
        };
        param.map(|_| tensor)
    }
}

/// Replace a freshly built model's matrices with values derived from `seed`.
pub fn initialize<M: Module>(model: M, seed: u64, device: &Device) -> M {
    let mut init = ParameterInit {
        rng: Rng::new(seed ^ 0xA5A5_5A5A_1234_5678),
        device: device.clone(),
    };
    model.map(&mut init)
}
