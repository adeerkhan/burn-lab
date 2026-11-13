//! Dump weights and gradients as compact summaries for heatmaps.
//!
//! Burn's parameter visitor does not name what it visits, so a layer is
//! identified by its index in visit order and its shape.

use burn::module::{ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;
use serde::Serialize;

use crate::unet::Unet;

const CELLS: usize = 32;

#[derive(Clone, Debug, Serialize)]
pub struct LayerDump {
    pub index: usize,
    pub dims: Vec<usize>,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub zero_fraction: f32,
    /// 32x32 block means of the flattened tensor, for the heatmap.
    pub grid: Vec<f32>,
}

fn summarise(index: usize, dims: Vec<usize>, values: &[f32]) -> LayerDump {
    let count = values.len().max(1) as f32;
    let mean = values.iter().sum::<f32>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / count;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let zeros = values.iter().filter(|v| v.abs() < 1e-8).count();

    let mut grid = vec![0.0f32; CELLS * CELLS];
    let mut counts = vec![0usize; CELLS * CELLS];
    for (position, value) in values.iter().enumerate() {
        let cell = (position as u64 * (CELLS * CELLS) as u64 / values.len().max(1) as u64) as usize;
        grid[cell.min(CELLS * CELLS - 1)] += value;
        counts[cell.min(CELLS * CELLS - 1)] += 1;
    }
    for (value, count) in grid.iter_mut().zip(&counts) {
        if *count > 0 {
            *value /= *count as f32;
        }
    }

    LayerDump {
        index,
        dims,
        mean,
        std: variance.sqrt(),
        min,
        max,
        zero_fraction: zeros as f32 / count,
        grid,
    }
}

struct WeightCollector {
    layers: Vec<LayerDump>,
    next: usize,
}

impl ModuleVisitor for WeightCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let value = param.val();
        let dims = value.dims().to_vec();
        let data = value
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let index = self.next;
        self.next += 1;
        self.layers.push(summarise(index, dims, &data));
    }
}

struct GradientCollector<'a> {
    grads: &'a GradientsParams,
    layers: Vec<LayerDump>,
    next: usize,
}

impl ModuleVisitor for GradientCollector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let index = self.next;
        self.next += 1;
        let Some(gradient) = self.grads.get::<D>(param.id) else {
            return;
        };
        let dims = gradient.dims().to_vec();
        let data = gradient
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        self.layers.push(summarise(index, dims, &data));
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Dump {
    pub weights: Vec<LayerDump>,
    pub gradients: Vec<LayerDump>,
}

/// Summarise every parameter, and optionally its gradient, of `model`.
pub fn dump(model: &Unet, grads: Option<&GradientsParams>) -> Dump {
    let mut weights = WeightCollector {
        layers: Vec::new(),
        next: 0,
    };
    model.visit(&mut weights);

    let gradients = match grads {
        Some(grads) => {
            let mut collector = GradientCollector {
                grads,
                layers: Vec::new(),
                next: 0,
            };
            model.visit(&mut collector);
            collector.layers
        }
        None => Vec::new(),
    };

    Dump {
        weights: weights.layers,
        gradients,
    }
}
