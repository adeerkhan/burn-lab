//! Small helpers the tests need to manipulate and inspect parameters.

use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;

use crate::unet::Unet;

/// Set every float parameter to `value`.
pub fn fill(model: &Unet, value: f32, device: &Device) -> Unet {
    struct Fill {
        value: f32,
        device: Device,
    }

    impl ModuleMapper for Fill {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let dims = param.val().dims();
            let count: usize = dims.iter().product();
            let tensor = Tensor::<D>::from_data(
                TensorData::new(vec![self.value; count], dims),
                &self.device,
            );
            param.map(|_| tensor)
        }
    }

    model.clone().map(&mut Fill {
        value,
        device: device.clone(),
    })
}

/// The first scalar of the first float parameter, in visit order.
pub fn first_param(model: &Unet) -> f32 {
    #[derive(Default)]
    struct First(Option<f32>);

    impl ModuleVisitor for First {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if self.0.is_none() {
                let data = param
                    .val()
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .unwrap();
                self.0 = Some(data[0]);
            }
        }
    }

    let mut first = First::default();
    model.visit(&mut first);
    first.0.expect("the model has parameters")
}
