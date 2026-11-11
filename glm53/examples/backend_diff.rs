use burn::{prelude::*, tensor::Device};
use glm53::{GLM53Flash, ModelConfig};

fn run(device: &Device, record: &burn::store::ModuleRecord) -> Vec<f32> {
    let config = ModelConfig::new();
    let model = GLM53Flash::new(&config, device).load_record(record.clone());
    let tokens: Vec<i64> = vec![1, 14, 40, 20, 30, 50, 60, 70, 80, 90];
    let width = tokens.len();
    let input = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, width]), device);
    let (logits, _) = model.forward(input);
    logits
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn main() {
    let record = burn::store::ModuleRecord::load("artifacts/glm53-pretrain/model").expect("load");

    let flex = run(&Device::flex(), &record);
    let wgpu = run(&Device::wgpu(Default::default()), &record);

    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    let mut mismatches = 0usize;
    for (index, (a, b)) in flex.iter().zip(wgpu.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > 1e-3 {
            mismatches += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            max_at = index;
        }
    }

    println!("elements      : {}", flex.len());
    println!("mismatches    : {mismatches} (>{:e})", 1e-3);
    println!("max abs diff  : {max_diff:.6} at {max_at}");
    println!("flex  [0..6]  : {:?}", &flex[..6]);
    println!("wgpu  [0..6]  : {:?}", &wgpu[..6]);
}
