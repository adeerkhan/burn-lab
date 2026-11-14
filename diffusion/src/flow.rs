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
//! and the network is trained with a plain MSE against `v`. Sampling integrates
//! `dx/dt = v(x, t)` from `t = 0` (noise) to `t = 1` (data). The reference uses
//! `torchdiffeq.odeint` with the midpoint method; this port implements the same
//! fixed-step midpoint rule, so `steps` costs two network evaluations each.
//!
//! What is not ported: reflow (the second rectification pass), consistency flow
//! matching, LPIPS / pseudo-Huber losses, the mean-variance head, and the
//! immiscible pairing. They are optional research branches in the reference,
//! not part of the base algorithm.
//!
//! [`lucidrains/rectified-flow-pytorch`]: https://github.com/lucidrains/rectified-flow-pytorch

use burn::prelude::*;
use burn::tensor::Distribution;

use crate::unet::Unet;

/// The reference's `RectifiedFlow` in its default configuration: predict the
/// flow, MSE loss, identity noise schedule.
#[derive(Clone, Debug)]
pub struct RectifiedFlow {
    pub model: Unet,
    pub image_size: usize,
}

impl RectifiedFlow {
    pub fn new(model: Unet, image_size: usize) -> Self {
        Self { model, image_size }
    }

    /// `x_t = (1 - t) * noise + t * data`, the reference's `noise.lerp(data, t)`.
    pub fn interpolate(data: Tensor<4>, noise: Tensor<4>, time: f32) -> Tensor<4> {
        noise.mul_scalar(1.0 - time).add(data.mul_scalar(time))
    }

    /// The flow-matching loss: MSE between the predicted velocity and
    /// `data - noise`, at uniformly sampled times.
    pub fn p_losses(
        &self,
        data: Tensor<4>,
        times: &[f32],
        noise: Tensor<4>,
        device: &Device,
    ) -> Tensor<1> {
        let batch = times.len();
        let [_, channels, height, width] = data.dims();

        // [b, 1, 1, 1] so the interpolation broadcasts over channels and pixels.
        let time = Tensor::<1>::from_data(TensorData::new(times.to_vec(), [batch]), device)
            .reshape([batch, 1, 1, 1]);
        let one_minus_time = time.clone().mul_scalar(-1.0).add_scalar(1.0);
        let x_t = noise
            .clone()
            .mul(one_minus_time)
            .add(data.clone().mul(time));

        let target = data.sub(noise);
        let predicted = self.model.forward_continuous(x_t, times);

        let elements = (channels * height * width) as f32;
        predicted
            .sub(target)
            .powf_scalar(2.0)
            .reshape([batch, channels * height * width])
            .sum_dim(1)
            .div_scalar(elements)
            .reshape([batch])
            .mean()
    }

    /// Integrate from noise at `t = 0` to data at `t = 1` with the midpoint rule.
    pub fn sample(&self, batch: usize, steps: usize, device: &Device) -> Tensor<4> {
        let steps = steps.max(2);
        let mut x = Tensor::<4>::random(
            [batch, 3, self.image_size, self.image_size],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let step = 1.0 / steps as f32;

        for index in 0..steps {
            let time = index as f32 * step;
            let first = self.model.forward_continuous(x.clone(), &vec![time; batch]);
            let midpoint = x.clone().add(first.mul_scalar(step * 0.5));
            let second = self
                .model
                .forward_continuous(midpoint, &vec![time + step * 0.5; batch]);
            x = x.add(second.mul_scalar(step));
        }
        x
    }
}
