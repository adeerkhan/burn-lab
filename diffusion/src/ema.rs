//! Exponential moving average of a module's float parameters.
//!
//! `ema_pytorch` is a dependency of the reference trainer; Burn has no
//! equivalent, so this is the minimum needed: a shadow module and a visit-order
//! blend.

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;

/// A shadow copy of a module, blended towards the live weights every update.
#[derive(Clone, Debug)]
pub struct Ema<M: Module> {
    pub model: M,
    pub decay: f64,
    updates: usize,
}

impl<M: Module + Clone> Ema<M> {
    pub fn new(model: &M, decay: f64) -> Self {
        Self {
            model: model.clone(),
            decay,
            updates: 0,
        }
    }

    /// `shadow = decay * shadow + (1 - decay) * live`, parameter by parameter.
    pub fn update(&mut self, live: &M) {
        let mut collector = Collector::default();
        live.visit(&mut collector);
        let mut blender = Blend {
            values: collector.flat,
            cursor: 0,
            decay: self.decay as f32,
        };
        self.model = self.model.clone().map(&mut blender);
        self.updates += 1;
    }

    pub fn updates(&self) -> usize {
        self.updates
    }
}

#[derive(Default)]
struct Collector {
    flat: Vec<f32>,
}

impl burn::module::ModuleVisitor for Collector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let data = param
            .val()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        self.flat.extend(data);
    }
}

struct Blend {
    values: Vec<f32>,
    cursor: usize,
    decay: f32,
}

impl ModuleMapper for Blend {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let current = param.val();
        let dims = current.dims();
        let count: usize = dims.iter().product();
        let live = &self.values[self.cursor..self.cursor + count];
        self.cursor += count;

        let shadow = current
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let blended: Vec<f32> = shadow
            .iter()
            .zip(live)
            .map(|(old, new)| self.decay * old + (1.0 - self.decay) * new)
            .collect();
        let device = param.val().device();
        let tensor = Tensor::<D>::from_data(TensorData::new(blended, dims), &device);
        param.map(|_| tensor)
    }
}
