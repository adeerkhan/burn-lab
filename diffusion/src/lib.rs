//! Ports of `lucidrains/denoising-diffusion-pytorch` (DDPM/DDIM) and
//! `lucidrains/rectified-flow-pytorch` (flow matching) to Burn.

pub mod data;
pub mod diffusion;
pub mod ema;
pub mod flow;
pub mod inspect;
pub mod test_support;
pub mod unet;
