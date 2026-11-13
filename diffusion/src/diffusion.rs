//! `GaussianDiffusion` from the reference, with the same buffers and the same
//! sampling loops.

use burn::prelude::*;

use crate::unet::Unet;

/// What the network predicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    PredNoise,
    PredX0,
    PredV,
}

/// How the beta schedule is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaSchedule {
    Linear,
    Cosine,
    Sigmoid,
}

fn linear_beta_schedule(timesteps: usize) -> Vec<f64> {
    let scale = 1000.0 / timesteps as f64;
    let (start, end) = (scale * 0.0001, scale * 0.02);
    (0..timesteps)
        .map(|index| start + (end - start) * index as f64 / (timesteps - 1) as f64)
        .collect()
}

fn cosine_beta_schedule(timesteps: usize, s: f64) -> Vec<f64> {
    let steps = timesteps + 1;
    let mut alphas_cumprod = Vec::with_capacity(steps);
    for index in 0..steps {
        let t = index as f64 / timesteps as f64;
        let value = ((t + s) / (1.0 + s) * std::f64::consts::PI * 0.5)
            .cos()
            .powi(2);
        alphas_cumprod.push(value);
    }
    let first = alphas_cumprod[0];
    for value in &mut alphas_cumprod {
        *value /= first;
    }
    (1..steps)
        .map(|index| (1.0 - alphas_cumprod[index] / alphas_cumprod[index - 1]).clamp(0.0, 0.999))
        .collect()
}

/// Sigmoid schedule from <https://arxiv.org/abs/2212.11972>, the reference default.
fn sigmoid_beta_schedule(timesteps: usize, start: f64, end: f64, tau: f64) -> Vec<f64> {
    let steps = timesteps + 1;
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    // The reference builds these with `torch.tensor(...)`, which is f32, and
    // then divides an f64 tensor by them. Matching that keeps the buffers equal
    // to the reference's to rounding.
    let v_start = sigmoid(start / tau) as f32 as f64;
    let v_end = sigmoid(end / tau) as f32 as f64;

    let mut alphas_cumprod = Vec::with_capacity(steps);
    for index in 0..steps {
        let t = index as f64 / timesteps as f64;
        // `(-sigmoid((t * (end - start) + start) / tau) + v_end) / (v_end - v_start)`
        let raw = -sigmoid((t * (end - start) + start) / tau) + v_end;
        alphas_cumprod.push(raw / (v_end - v_start));
    }
    let first = alphas_cumprod[0];
    for value in &mut alphas_cumprod {
        *value /= first;
    }
    (1..steps)
        .map(|index| (1.0 - alphas_cumprod[index] / alphas_cumprod[index - 1]).clamp(0.0, 0.999))
        .collect()
}

/// The reference's diffusion wrapper.
#[derive(Clone, Debug)]
pub struct GaussianDiffusion {
    pub model: Unet,
    pub objective: Objective,
    pub num_timesteps: usize,
    pub sampling_timesteps: usize,
    pub ddim_eta: f64,
    pub image_size: usize,
    pub channels: usize,

    // Host-side buffers, exactly as the reference registers them.
    pub betas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
    pub alphas_cumprod_prev: Vec<f64>,
    pub sqrt_alphas_cumprod: Vec<f64>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    pub sqrt_recip_alphas_cumprod: Vec<f64>,
    pub sqrt_recipm1_alphas_cumprod: Vec<f64>,
    pub posterior_variance: Vec<f64>,
    pub posterior_log_variance_clipped: Vec<f64>,
    pub posterior_mean_coef1: Vec<f64>,
    pub posterior_mean_coef2: Vec<f64>,
    pub loss_weight: Vec<f64>,
}

impl GaussianDiffusion {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Unet,
        image_size: usize,
        timesteps: usize,
        sampling_timesteps: Option<usize>,
        objective: Objective,
        schedule: BetaSchedule,
        min_snr_loss_weight: bool,
        min_snr_gamma: f64,
    ) -> Self {
        let betas = match schedule {
            BetaSchedule::Linear => linear_beta_schedule(timesteps),
            BetaSchedule::Cosine => cosine_beta_schedule(timesteps, 0.008),
            BetaSchedule::Sigmoid => sigmoid_beta_schedule(timesteps, -3.0, 3.0, 1.0),
        };

        let alphas: Vec<f64> = betas.iter().map(|beta| 1.0 - beta).collect();
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut running = 1.0;
        for alpha in &alphas {
            running *= alpha;
            alphas_cumprod.push(running);
        }
        let mut alphas_cumprod_prev = vec![1.0];
        alphas_cumprod_prev.extend_from_slice(&alphas_cumprod[..timesteps - 1]);

        let sqrt_alphas_cumprod = alphas_cumprod.iter().map(|v| v.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod =
            alphas_cumprod.iter().map(|v| (1.0 - v).sqrt()).collect();
        let sqrt_recip_alphas_cumprod = alphas_cumprod.iter().map(|v| (1.0 / v).sqrt()).collect();
        let sqrt_recipm1_alphas_cumprod = alphas_cumprod
            .iter()
            .map(|v| (1.0 / v - 1.0).sqrt())
            .collect();

        let mut posterior_variance = Vec::with_capacity(timesteps);
        let mut posterior_mean_coef1 = Vec::with_capacity(timesteps);
        let mut posterior_mean_coef2 = Vec::with_capacity(timesteps);
        for index in 0..timesteps {
            let beta = betas[index];
            let alpha_bar = alphas_cumprod[index];
            let prev = alphas_cumprod_prev[index];
            posterior_variance.push(beta * (1.0 - prev) / (1.0 - alpha_bar));
            posterior_mean_coef1.push(beta * prev.sqrt() / (1.0 - alpha_bar));
            posterior_mean_coef2.push((1.0 - prev) * alphas[index].sqrt() / (1.0 - alpha_bar));
        }
        let posterior_log_variance_clipped: Vec<f64> = posterior_variance
            .iter()
            .map(|v| v.max(1e-20).ln())
            .collect();

        let mut loss_weight = vec![1.0; timesteps];
        if min_snr_loss_weight {
            for index in 0..timesteps {
                let alpha_bar = alphas_cumprod[index];
                let snr = alpha_bar / (1.0 - alpha_bar);
                let clipped = snr.min(min_snr_gamma);
                loss_weight[index] = match objective {
                    Objective::PredNoise => clipped / snr,
                    Objective::PredX0 => clipped,
                    Objective::PredV => clipped / (snr + 1.0),
                };
            }
        }

        let sampling_timesteps = sampling_timesteps.unwrap_or(timesteps);
        let channels = model.channels();
        Self {
            model,
            objective,
            num_timesteps: timesteps,
            sampling_timesteps,
            ddim_eta: 0.0,
            image_size,
            channels,
            betas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
            loss_weight,
        }
    }

    /// Per-sample coefficient tensor shaped `[batch, 1, 1, 1]`.
    fn extract(values: &[f64], times: &[i64], device: &Device) -> Tensor<4> {
        let batch = times.len();
        let data: Vec<f32> = times.iter().map(|t| values[*t as usize] as f32).collect();
        Tensor::<4>::from_data(TensorData::new(data, [batch, 1, 1, 1]), device)
    }

    pub fn q_sample(
        &self,
        x_start: Tensor<4>,
        times: &[i64],
        noise: Tensor<4>,
        device: &Device,
    ) -> Tensor<4> {
        let sqrt_ab = Self::extract(&self.sqrt_alphas_cumprod, times, device);
        let sqrt_1mab = Self::extract(&self.sqrt_one_minus_alphas_cumprod, times, device);
        sqrt_ab.mul(x_start).add(sqrt_1mab.mul(noise))
    }

    pub fn predict_start_from_noise(
        &self,
        x_t: Tensor<4>,
        times: &[i64],
        noise: Tensor<4>,
        device: &Device,
    ) -> Tensor<4> {
        let recip = Self::extract(&self.sqrt_recip_alphas_cumprod, times, device);
        let recipm1 = Self::extract(&self.sqrt_recipm1_alphas_cumprod, times, device);
        recip.mul(x_t).sub(recipm1.mul(noise))
    }

    pub fn predict_noise_from_start(
        &self,
        x_t: Tensor<4>,
        times: &[i64],
        x_start: Tensor<4>,
        device: &Device,
    ) -> Tensor<4> {
        let recip = Self::extract(&self.sqrt_recip_alphas_cumprod, times, device);
        let recipm1 = Self::extract(&self.sqrt_recipm1_alphas_cumprod, times, device);
        recip.mul(x_t).sub(x_start).div(recipm1)
    }

    pub fn predict_v(
        &self,
        x_start: Tensor<4>,
        times: &[i64],
        noise: Tensor<4>,
        device: &Device,
    ) -> Tensor<4> {
        let sqrt_ab = Self::extract(&self.sqrt_alphas_cumprod, times, device);
        let sqrt_1mab = Self::extract(&self.sqrt_one_minus_alphas_cumprod, times, device);
        sqrt_ab.mul(noise).sub(sqrt_1mab.mul(x_start))
    }

    pub fn predict_start_from_v(
        &self,
        x_t: Tensor<4>,
        times: &[i64],
        v: Tensor<4>,
        device: &Device,
    ) -> Tensor<4> {
        let sqrt_ab = Self::extract(&self.sqrt_alphas_cumprod, times, device);
        let sqrt_1mab = Self::extract(&self.sqrt_one_minus_alphas_cumprod, times, device);
        sqrt_ab.mul(x_t).sub(sqrt_1mab.mul(v))
    }

    /// `(x_start, pred_noise)` for a batch of noisy images.
    pub fn model_predictions(
        &self,
        x: Tensor<4>,
        times: &[i64],
        clip_x_start: bool,
        device: &Device,
    ) -> (Tensor<4>, Tensor<4>) {
        let model_out = self.model.forward(x.clone(), times);
        let (x_start, pred_noise) = match self.objective {
            Objective::PredNoise => {
                let noise = model_out;
                let start = self.predict_start_from_noise(x.clone(), times, noise.clone(), device);
                (start, noise)
            }
            Objective::PredX0 => {
                let start = model_out;
                let noise = self.predict_noise_from_start(x.clone(), times, start.clone(), device);
                (start, noise)
            }
            Objective::PredV => {
                let start = self.predict_start_from_v(x.clone(), times, model_out, device);
                let noise = self.predict_noise_from_start(x.clone(), times, start.clone(), device);
                (start, noise)
            }
        };
        if clip_x_start {
            (x_start.clamp(-1.0, 1.0), pred_noise)
        } else {
            (x_start, pred_noise)
        }
    }

    /// The reference's `p_losses`, with v-prediction as the default objective.
    pub fn p_losses(
        &self,
        x_start: Tensor<4>,
        times: &[i64],
        noise: Tensor<4>,
        device: &Device,
    ) -> Tensor<1> {
        let x_t = self.q_sample(x_start.clone(), times, noise.clone(), device);
        let target = match self.objective {
            Objective::PredNoise => noise.clone(),
            Objective::PredX0 => x_start.clone(),
            Objective::PredV => self.predict_v(x_start.clone(), times, noise.clone(), device),
        };

        let predicted = self.model.forward(x_t, times);
        let batch = times.len();
        let [_, channels, height, width] = predicted.dims();
        let elements = (channels * height * width) as f32;
        let squared = predicted.sub(target).powf_scalar(2.0);
        let per_sample = squared
            .reshape([batch, channels * height * width])
            .sum_dim(1)
            .div_scalar(elements)
            .reshape([batch]);
        let weight = Self::extract(&self.loss_weight, times, device).reshape([batch]);
        per_sample.mul(weight).mean()
    }

    /// DDPM reverse step from `t` to `t - 1`.
    fn p_sample(&self, x: Tensor<4>, times: &[i64], device: &Device, add_noise: bool) -> Tensor<4> {
        let (x_start, _) = self.model_predictions(x.clone(), times, true, device);
        let coef1 = Self::extract(&self.posterior_mean_coef1, times, device);
        let coef2 = Self::extract(&self.posterior_mean_coef2, times, device);
        let mean = coef1.mul(x_start).add(coef2.mul(x));

        if !add_noise {
            return mean;
        }
        let variance = Self::extract(&self.posterior_variance, times, device);
        let noise = Tensor::<4>::random(
            mean.dims(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        mean.add(variance.sqrt().mul(noise))
    }

    /// Sample with DDIM when `sampling_timesteps < num_timesteps`, otherwise the
    /// full DDPM chain. Mirrors the reference's `sample`.
    pub fn sample(&self, batch: usize, device: &Device) -> Tensor<4> {
        let shape = [batch, self.channels, self.image_size, self.image_size];
        let mut x =
            Tensor::<4>::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device);

        if self.sampling_timesteps < self.num_timesteps {
            let steps = self.sampling_timesteps;
            let total = self.num_timesteps as f64;
            let times: Vec<i64> = (0..=steps)
                .map(|index| {
                    let value = -1.0 + (total) * index as f64 / steps as f64;
                    value.round() as i64
                })
                .rev()
                .collect();

            for pair in times.windows(2) {
                let (time, time_next) = (pair[0], pair[1]);
                if time < 0 {
                    continue;
                }
                let current = vec![time; batch];
                let (x_start, pred_noise) =
                    self.model_predictions(x.clone(), &current, true, device);

                if time_next < 0 {
                    x = x_start;
                    continue;
                }

                let alpha = self.alphas_cumprod[time as usize];
                let alpha_next = self.alphas_cumprod[time_next as usize];
                let sigma = self.ddim_eta
                    * ((1.0 - alpha_next) / (1.0 - alpha)).sqrt()
                    * (1.0 - alpha / alpha_next).sqrt();

                let c1 = alpha_next.sqrt();
                let c2 = (1.0 - alpha_next - sigma * sigma).max(0.0).sqrt();
                let next = x_start
                    .mul_scalar(c1 as f32)
                    .add(pred_noise.mul_scalar(c2 as f32));
                let next = if sigma > 0.0 {
                    let noise = Tensor::<4>::random(
                        next.dims(),
                        burn::tensor::Distribution::Normal(0.0, 1.0),
                        device,
                    );
                    next.add(noise.mul_scalar(sigma))
                } else {
                    next
                };
                x = next;
            }
            return x;
        }

        for time in (0..self.num_timesteps).rev() {
            let current = vec![time as i64; batch];
            x = self.p_sample(x, &current, device, time > 0);
        }
        x
    }
}
