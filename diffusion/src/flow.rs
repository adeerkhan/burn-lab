//! Rectified flow, ported from [`lucidrains/rectified-flow-pytorch`].
//!
//! The model learns a velocity field that transports Gaussian noise to data
//! along straight paths. At a time `t` in `[0, 1]`:
//!
//! ```text
//! x_t = (1 - t) * noise + t * data
//! v   = data - noise
//! ```
//!
//! and the network is trained against `v`. Sampling integrates `dx/dt = v(x, t)`
//! from `t = 0` (noise) to `t = 1` (data) with the midpoint rule, matching the
//! reference's `odeint(method='midpoint')`.
//!
//! Every constructor option of the reference that does not need external
//! weights is here:
//!
//! | Option | Here |
//! | --- | --- |
//! | `predict='flow' \| 'noise' \| 'clean'` | [`Predict`] |
//! | `loss_fn='mse' \| 'pseudo_huber'` | [`FlowLoss`] |
//! | `noise_schedule=identity \| cosmap` | [`NoiseSchedule`] |
//! | `immiscible` | greedy optimal pairing inside a batch |
//! | `use_consistency` | data-match and velocity-match losses against an EMA |
//!
//! Not implemented: `pseudo_huber_with_lpips` (needs VGG16 weights that are not
//! available offline), `mean_variance_net` (probabilistic output; unused by any
//! sampler here) and `auto_normalize` (the flowers are already in `[-1, 1]`, and
//! running statistics would have to be checkpointed with the weights).
//!
//! Reflow — the second rectification pass — is not an option of this type in the
//! reference either; it is [`Reflow`], trained on `(noise, sample)` pairs drawn
//! from a frozen copy of a trained model.
//!
//! [`lucidrains/rectified-flow-pytorch`]: https://github.com/lucidrains/rectified-flow-pytorch

use burn::prelude::*;
use burn::tensor::Distribution;

use crate::unet::Unet;

/// The ODE integrator used at sampling time. All three reuse the same weights;
/// only the trajectory differs. Costs are counted in network evaluations (NFE).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowSolver {
    /// One evaluation per step: `x += h v(x, t)`.
    Euler,
    /// Two evaluations per step: midpoint of the interval. The default here.
    Midpoint,
    /// Two evaluations per step: trapezoid (Heun), explicit and second order.
    Heun,
}

/// What the network's output is interpreted as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predict {
    /// Predict the velocity `data - noise`.
    Flow,
    /// Predict the noise, as in DDPM's epsilon parameterisation.
    Noise,
    /// Predict the clean data.
    Clean,
}

/// The regression loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowLoss {
    Mse,
    /// `sqrt(mse + c^2) - c`, from the reference's `PseudoHuberLoss`.
    PseudoHuber,
}

/// How the uniform time is warped before interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseSchedule {
    Identity,
    /// `1 - 1 / (tan(pi/2 * t) + 1)`, Algorithm 21 of the SD3 paper.
    Cosmap,
}

/// How the consistency time shift varies along the path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaSchedule {
    Constant,
    /// Larger shift near the noise end, vanishing at the data end.
    Linear,
    /// Cosine ramp, zero at the data end.
    Cosine,
}

#[derive(Clone, Debug)]
pub struct FlowConfig {
    pub predict: Predict,
    pub loss: FlowLoss,
    pub schedule: NoiseSchedule,
    /// Pair each noise with a data sample by optimal transport inside the batch.
    pub immiscible: bool,
    /// Add consistency flow matching's data/velocity terms against an EMA.
    pub consistency: bool,
    pub consistency_weight: f32,
    pub consistency_velocity_alpha: f32,
    pub consistency_delta_time: f32,
    pub delta_schedule: DeltaSchedule,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            predict: Predict::Flow,
            loss: FlowLoss::Mse,
            schedule: NoiseSchedule::Identity,
            immiscible: false,
            consistency: false,
            consistency_weight: 1.0,
            consistency_velocity_alpha: 1e-5,
            consistency_delta_time: 1e-3,
            delta_schedule: DeltaSchedule::Constant,
        }
    }
}

impl NoiseSchedule {
    pub fn map(&self, time: f32) -> f32 {
        match self {
            Self::Identity => time,
            Self::Cosmap => 1.0 - 1.0 / ((std::f32::consts::FRAC_PI_2 * time).tan() + 1.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RectifiedFlow {
    pub model: Unet,
    pub image_size: usize,
    pub config: FlowConfig,
}

/// The derived quantities the losses and the sampler both need.
struct FlowOutput {
    flow: Tensor<4>,
    /// `x_t + flow * (1 - t)`: the model's estimate of the clean data.
    pred_data: Tensor<4>,
}

impl RectifiedFlow {
    pub fn new(model: Unet, image_size: usize) -> Self {
        Self {
            model,
            image_size,
            config: FlowConfig::default(),
        }
    }

    pub fn with_config(mut self, config: FlowConfig) -> Self {
        self.config = config;
        self
    }

    /// `x_t = (1 - t) * noise + t * data`, the reference's `noise.lerp(data, t)`.
    pub fn interpolate(data: Tensor<4>, noise: Tensor<4>, time: f32) -> Tensor<4> {
        noise.mul_scalar(1.0 - time).add(data.mul_scalar(time))
    }

    /// Derive the velocity from the network's output, whatever it predicts.
    fn derive(&self, output: Tensor<4>, x_t: Tensor<4>, times: &[f32]) -> FlowOutput {
        let batch = times.len();
        let [_, channels, height, width] = x_t.dims();
        let time = Tensor::<1>::from_data(TensorData::new(times.to_vec(), [batch]), &x_t.device())
            .reshape([batch, 1, 1, 1]);
        let _ = (channels, height, width);

        let flow = match self.config.predict {
            Predict::Flow => output,
            Predict::Noise => x_t.clone().sub(output).div(time.clone().clamp_min(5e-3)),
            Predict::Clean => output
                .sub(x_t.clone())
                .div(time.clone().neg().add_scalar(1.0).clamp_min(5e-3)),
        };
        let pred_data = x_t.add(flow.clone().mul(time.neg().add_scalar(1.0)));
        FlowOutput { flow, pred_data }
    }

    /// Pair noise with data by optimal transport inside the batch, so each
    /// sample travels a shorter path (the immiscible-diffusion trick). Greedy
    /// instead of the reference's Hungarian solve; exact for small batches in
    /// practice and O(n^2 log n).
    fn immiscible_noise(data: &Tensor<4>, noise: Tensor<4>, device: &Device) -> Tensor<4> {
        let [batch, channels, height, width] = data.dims();
        let elements = channels * height * width;
        let flat_data = data.clone().reshape([batch, elements]);
        let flat_noise = noise.clone().reshape([batch, elements]);

        let cost = flat_data
            .reshape([batch, 1, elements])
            .sub(flat_noise.clone().reshape([1, batch, elements]))
            .powf_scalar(2.0)
            .mean_dim(2)
            .reshape([batch, batch])
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();

        let mut pairs: Vec<(f32, usize, usize)> = (0..batch)
            .flat_map(|left| (0..batch).map(move |right| (left, right)))
            .map(|(left, right)| (cost[left * batch + right], left, right))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut used_left = vec![false; batch];
        let mut used_right = vec![false; batch];
        let mut permutation = vec![0usize; batch];
        for (_, left, right) in pairs {
            if !used_left[left] && !used_right[right] {
                used_left[left] = true;
                used_right[right] = true;
                permutation[left] = right;
            }
        }

        let values = noise
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let mut reordered = Vec::with_capacity(values.len());
        for index in permutation {
            let source = index * elements;
            reordered.extend_from_slice(&values[source..source + elements]);
        }
        Tensor::<4>::from_data(
            TensorData::new(reordered, [batch, channels, height, width]),
            device,
        )
    }

    /// The flow-matching loss. `ema` is required when consistency matching is on;
    /// it is the shadow model the consistency targets come from.
    pub fn p_losses(
        &self,
        data: Tensor<4>,
        times: &[f32],
        noise: Tensor<4>,
        device: &Device,
        ema: Option<&Unet>,
    ) -> Tensor<1> {
        let batch = times.len();
        let [_, channels, height, width] = data.dims();
        let elements = (channels * height * width) as f32;

        let noise = if self.config.immiscible {
            Self::immiscible_noise(&data, noise, device)
        } else {
            noise
        };

        let scheduled: Vec<f32> = times
            .iter()
            .map(|time| self.config.schedule.map(*time))
            .collect();
        let time = Tensor::<1>::from_data(TensorData::new(scheduled.clone(), [batch]), device)
            .reshape([batch, 1, 1, 1]);
        let noised = noise
            .clone()
            .mul(time.clone().neg().add_scalar(1.0))
            .add(data.clone().mul(time));

        let raw = self.model.forward_continuous(noised.clone(), &scheduled);
        let derived = self.derive(raw.clone(), noised.clone(), &scheduled);
        let target = data.clone().sub(noise.clone());

        // The reference's `predict` switch: flow and clean regress the velocity,
        // noise regresses the noise itself.
        let (prediction, target) = match self.config.predict {
            Predict::Flow | Predict::Clean => (derived.flow.clone(), target),
            Predict::Noise => (raw, noise.clone()),
        };

        let squared = prediction.sub(target).powf_scalar(2.0);
        let per_sample = squared
            .reshape([batch, channels * height * width])
            .sum_dim(1)
            .div_scalar(elements)
            .reshape([batch]);
        let main = match self.config.loss {
            FlowLoss::Mse => per_sample.mean(),
            FlowLoss::PseudoHuber => {
                let c = 0.000_54 * (channels * height * width) as f32;
                per_sample.add_scalar(c * c).sqrt().sub_scalar(c).mean()
            }
        };

        if !self.config.consistency {
            return main;
        }
        let Some(ema) = ema else {
            panic!("consistency flow matching needs the EMA model");
        };

        // Targets from the shadow model a step further along the path.
        let delta = self.config.consistency_delta_time;
        let shifted: Vec<f32> = scheduled
            .iter()
            .map(|time| {
                let shift = match self.config.delta_schedule {
                    DeltaSchedule::Constant => delta,
                    DeltaSchedule::Linear => delta * (1.0 - time),
                    DeltaSchedule::Cosine => {
                        delta * 0.5 * (1.0 + (std::f32::consts::PI * time).cos())
                    }
                };
                (time + shift).min(1.0 - 1e-3)
            })
            .collect();
        let shifted_t = Tensor::<1>::from_data(TensorData::new(shifted.clone(), [batch]), device)
            .reshape([batch, 1, 1, 1]);
        let shifted_noised = noise
            .mul(shifted_t.clone().neg().add_scalar(1.0))
            .add(data.mul(shifted_t));
        let ema_output = ema
            .forward_continuous(shifted_noised.clone(), &shifted)
            .detach();
        let ema_derived = self.derive(ema_output, shifted_noised, &shifted);

        let data_match = derived
            .pred_data
            .sub(ema_derived.pred_data.detach())
            .powf_scalar(2.0)
            .mean();
        let velocity_match = derived
            .flow
            .sub(ema_derived.flow.detach())
            .powf_scalar(2.0)
            .mean();
        main.add(
            data_match
                .add(velocity_match.mul_scalar(self.config.consistency_velocity_alpha))
                .mul_scalar(self.config.consistency_weight),
        )
    }

    /// Integrate from noise at `t = 0` to data at `t = 1` with the default
    /// (midpoint) solver.
    pub fn sample(&self, batch: usize, steps: usize, device: &Device) -> Tensor<4> {
        self.sample_with_solver(batch, steps, FlowSolver::Midpoint, device)
    }

    /// Sample with an explicit choice of integrator.
    pub fn sample_with_solver(
        &self,
        batch: usize,
        steps: usize,
        solver: FlowSolver,
        device: &Device,
    ) -> Tensor<4> {
        self.sample_from(
            Tensor::<4>::random(
                [batch, 3, self.image_size, self.image_size],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
            steps,
            solver,
            device,
        )
    }

    /// The ODE from an explicit starting noise, which reflow needs to keep the
    /// coupling it trains on.
    pub fn sample_from(
        &self,
        noise: Tensor<4>,
        steps: usize,
        solver: FlowSolver,
        _device: &Device,
    ) -> Tensor<4> {
        let steps = steps.max(1);
        let batch = noise.dims()[0];
        let mut x = noise;
        let step = 1.0 / steps as f32;

        for index in 0..steps {
            let time = index as f32 * step;
            match solver {
                FlowSolver::Euler => {
                    let first = self.velocity(x.clone(), time, batch);
                    x = x.add(first.mul_scalar(step));
                }
                FlowSolver::Midpoint => {
                    let first = self.velocity(x.clone(), time, batch);
                    let midpoint = x.clone().add(first.mul_scalar(step * 0.5));
                    let second = self.velocity(midpoint, time + step * 0.5, batch);
                    x = x.add(second.mul_scalar(step));
                }
                FlowSolver::Heun => {
                    let first = self.velocity(x.clone(), time, batch);
                    let endpoint = x.clone().add(first.clone().mul_scalar(step));
                    let second = self.velocity(endpoint, (time + step).min(1.0), batch);
                    x = x.add(first.add(second).mul_scalar(step * 0.5));
                }
            }
        }
        x
    }

    fn velocity(&self, x: Tensor<4>, time: f32, batch: usize) -> Tensor<4> {
        let times = vec![time; batch];
        let output = self.model.forward_continuous(x.clone(), &times);
        self.derive(output, x, &times).flow
    }
}
