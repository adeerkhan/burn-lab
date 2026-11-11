use burn::tensor::Device;

/// Backend names accepted on the command line.
pub const BACKENDS: &[&str] = &["wgpu", "flex"];

/// Build a [`Device`] from a backend name.
///
/// This is the practical payoff of Burn's runtime dispatch: the model, data and
/// training code below is identical for every backend, only the device changes.
pub fn device(name: &str) -> Device {
    match name {
        "wgpu" => Device::wgpu(Default::default()),
        "flex" => Device::flex(),
        other => panic!("unknown backend '{other}', expected one of {BACKENDS:?}"),
    }
}
