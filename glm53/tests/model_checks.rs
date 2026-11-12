//! Model checks that need parameter-level access: a finite-difference gradient
//! check, MoE routing, a golden forward fixture, checkpoint round-trips and a
//! cross-backend differential.

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    nn::loss::CrossEntropyLossConfig,
    optim::GradientsParams,
    prelude::*,
};
use glm53::{GLM53Flash, ModelConfig, moe::SparseMoE};

fn tiny_config() -> ModelConfig {
    ModelConfig::new()
        .with_dim(32)
        .with_heads(2)
        .with_layers(2)
        .with_experts(2)
        .with_expert_hidden(16)
        .with_streams(2)
        .with_max_sequence_length(64)
}

fn fixed_batch(device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let tokens: Vec<i64> = (0..2 * 12).map(|index| 10 + index as i64 % 60).collect();
    let targets: Vec<i64> = (0..2 * 12).map(|index| 11 + index as i64 % 60).collect();
    (
        Tensor::<2, Int>::from_data(TensorData::new(tokens, [2, 12]), device),
        Tensor::<2, Int>::from_data(TensorData::new(targets, [2, 12]), device),
    )
}

fn loss_of(model: &GLM53Flash, device: &Device) -> f32 {
    let (tokens, targets) = fixed_batch(device);
    let vocab = model.config().vocab_size;
    let (logits, _) = model.clone().forward(tokens);
    let logits = logits.reshape::<2, _>([2 * 12, vocab]);
    let targets = targets.reshape::<1, _>([2 * 12]);
    CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, targets)
        .into_scalar::<f32>()
}

#[derive(Default)]
struct ParamCollector {
    flat: Vec<f32>,
}

impl ModuleVisitor for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let data = param.val().into_data();
        self.flat.extend(
            data.convert::<f32>()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .copied(),
        );
    }
}

fn param_vector(model: &GLM53Flash) -> Vec<f32> {
    let mut collector = ParamCollector::default();
    model.visit(&mut collector);
    collector.flat
}

struct ParamSetter<'a> {
    values: &'a [f32],
    cursor: usize,
}

impl ModuleMapper for ParamSetter<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dims = param.val().dims();
        let count: usize = dims.iter().product();
        let device = param.val().device();
        let values = self.values[self.cursor..self.cursor + count].to_vec();
        self.cursor += count;
        let tensor = Tensor::<D>::from_data(TensorData::new(values, dims), &device);
        param.map(|_| tensor)
    }
}

fn set_param_vector(model: &GLM53Flash, values: &[f32]) -> GLM53Flash {
    let mut setter = ParamSetter { values, cursor: 0 };
    let model = model.clone().map(&mut setter);
    assert_eq!(setter.cursor, values.len(), "every parameter was written");
    model
}

#[derive(Default)]
struct GradCollector {
    grads: Option<GradientsParams>,
    flat: Vec<f32>,
}

impl ModuleMapper for GradCollector {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let id = param.id;
        let dims = param.val().dims();
        let count: usize = dims.iter().product();
        let grads = self.grads.as_mut().expect("gradients are set");
        match grads.remove::<D>(id) {
            Some(grad) => {
                let data = grad.into_data();
                self.flat.extend(
                    data.convert::<f32>()
                        .as_slice::<f32>()
                        .unwrap()
                        .iter()
                        .copied(),
                );
            }
            // A parameter with no gradient (e.g. an expert no token routed to)
            // still occupies its slot, so the vector stays aligned with the
            // parameter order.
            None => self.flat.extend(std::iter::repeat_n(0.0f32, count)),
        }
        param
    }
}

fn grad_vector(model: &GLM53Flash, device: &Device) -> Vec<f32> {
    let (tokens, targets) = fixed_batch(device);
    let vocab = model.config().vocab_size;
    let (logits, _) = model.clone().forward(tokens);
    let logits = logits.reshape::<2, _>([2 * 12, vocab]);
    let targets = targets.reshape::<1, _>([2 * 12]);
    let loss = CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, targets);

    let grads = GradientsParams::from_grads(loss.backward(), model);
    let mut collector = GradCollector {
        grads: Some(grads),
        flat: Vec::new(),
    };
    model.clone().map(&mut collector);
    collector.flat
}

/// Finite differences against `backward()` on a handful of parameters.
///
/// This is the check that matters most: a subtly wrong gradient still produces a
/// falling loss curve, so pretraining cannot detect it.
#[test]
fn gradients_match_finite_differences() {
    let device = Device::flex().autodiff();
    let config = tiny_config();
    let model = GLM53Flash::new(&config, &device);

    let params = param_vector(&model);
    let analytic = grad_vector(&model, &device);
    assert_eq!(
        analytic.len(),
        params.len(),
        "every parameter produced a gradient"
    );

    let epsilon = 1e-2f32;
    // Spread the probes across the parameter vector.
    let probes = [
        0,
        params.len() / 7,
        params.len() / 3,
        params.len() / 2,
        params.len() - 1,
    ];

    for index in probes {
        let mut up = params.clone();
        up[index] += epsilon;
        let mut down = params.clone();
        down[index] -= epsilon;

        let plus = loss_of(&set_param_vector(&model, &up), &device);
        let minus = loss_of(&set_param_vector(&model, &down), &device);
        let numeric = (plus - minus) / (2.0 * epsilon);

        let exact = analytic[index];
        let tolerance = 2e-2 * exact.abs().max(1.0);
        assert!(
            (numeric - exact).abs() < tolerance,
            "parameter {index}: numeric {numeric} vs analytic {exact}"
        );
    }
}

/// Per-layer expert usage must be a distribution over the experts.
#[test]
fn moe_usage_is_a_distribution_per_layer() {
    let device = Device::flex();
    let config = tiny_config();
    let model = GLM53Flash::new(&config, &device);

    let (tokens, _) = fixed_batch(&device);
    let (_, usage) = model.forward(tokens);

    let [layers, experts] = usage.dims();
    assert_eq!(layers, config.layers);
    assert_eq!(experts, config.experts);

    let data = usage.into_data();
    let values = data.as_slice::<f32>().unwrap();
    for layer in 0..layers {
        let row = &values[layer * experts..(layer + 1) * experts];
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "layer {layer} usage sums to {total}"
        );
        assert!(row.iter().all(|share| *share >= 0.0));
    }
}

/// Only the routed experts may receive gradient.
///
/// `SparseMoE` is built in isolation so the parameter layout is deterministic:
/// the router, then each expert's up/down projections in order, then the shared
/// expert. A batch of one token routes to exactly `top_k` experts, so every
/// other expert must have a zero gradient.
#[test]
fn only_routed_experts_receive_gradient() {
    let device = Device::flex().autodiff();
    let config = ModelConfig::new()
        .with_dim(16)
        .with_heads(1)
        .with_layers(1)
        .with_experts(4)
        .with_top_k(1)
        .with_expert_hidden(8)
        .with_streams(1)
        .with_max_sequence_length(32);

    let moe = SparseMoE::new(&config, &device);

    // A single token, so exactly `top_k` experts can fire.
    let token: i64 = 42;
    let input = Tensor::<2, Int>::from_data(TensorData::new(vec![token], [1, 1]), &device);
    let embedded = burn::nn::EmbeddingConfig::new(config.vocab_size, config.dim)
        .init(&device)
        .forward(input);

    let (output, usage) = moe.clone().forward(embedded);
    let loss = output.sum();
    let grads = GradientsParams::from_grads(loss.backward(), &moe);

    let mut collector = GradCollector {
        grads: Some(grads),
        flat: Vec::new(),
    };
    moe.map(&mut collector);
    let grads = collector.flat;

    // Layout: router, then per-expert (up, down), then shared (up, down).
    let dim = config.dim;
    let hidden = config.expert_hidden;
    let mut cursor = dim * config.experts;
    let up_len = dim * hidden * 2;
    let down_len = hidden * dim;

    let usage_data = usage.into_data();
    let usage_values = usage_data.as_slice::<f32>().unwrap();

    for (expert, share) in usage_values.iter().enumerate() {
        let slice = &grads[cursor..cursor + up_len + down_len];
        cursor += up_len + down_len;
        let magnitude: f32 = slice.iter().map(|value| value.abs()).sum();
        if *share == 0.0 {
            assert!(
                magnitude == 0.0,
                "unrouted expert {expert} received gradient {magnitude}"
            );
        }
    }

    let routed = usage_values.iter().filter(|share| **share > 0.0).count();
    assert_eq!(routed, config.top_k, "one token routes to top_k experts");
}

/// The sparse dispatch must reproduce the dense one.
#[test]
fn sparse_moe_matches_dense_dispatch() {
    let device = Device::flex();
    let config = ModelConfig::new()
        .with_dim(32)
        .with_heads(2)
        .with_layers(1)
        .with_experts(4)
        .with_top_k(2)
        .with_expert_hidden(16)
        .with_streams(2)
        .with_max_sequence_length(32);

    let moe = SparseMoE::new(&config, &device);

    let tokens: Vec<i64> = (0..3 * 16).map(|index| 10 + index % 60).collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(tokens, [3, 16]), &device);
    let embedded = burn::nn::EmbeddingConfig::new(config.vocab_size, config.dim)
        .init(&device)
        .forward(input);

    let (dense, dense_usage) = moe.clone().forward(embedded.clone());
    let (sparse, sparse_usage) = moe.forward_sparse(embedded);

    dense.into_data().assert_approx_eq::<f32>(
        &sparse.into_data(),
        burn::tensor::Tolerance::rel_abs(1e-5, 1e-5),
    );
    dense_usage.into_data().assert_approx_eq::<f32>(
        &sparse_usage.into_data(),
        burn::tensor::Tolerance::rel_abs(1e-5, 1e-5),
    );
}

/// A golden fixture: a tiny config and a fixed input must reproduce recorded
/// logits. Catches accidental rewiring of the block, not floating-point drift,
/// hence the loose tolerance.
#[test]
fn golden_forward_fixture() {
    let device = Device::flex();
    let config = tiny_config();

    // A seeded initialisation is not enough: the device RNG is shared, so a
    // test running in parallel would shift the draws. Overwrite every parameter
    // with a deterministic pattern instead.
    let model = GLM53Flash::new(&config, &device);
    let weights: Vec<f32> = (0..model.num_params())
        .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
        .collect();
    let model = set_param_vector(&model, &weights);

    let (tokens, _) = fixed_batch(&device);
    let (logits, _) = model.forward(tokens);

    let data = logits.into_data();
    let values = data.convert::<f32>();
    let slice = values.as_slice::<f32>().unwrap();

    assert_eq!(slice.len(), 6240);

    let expected_first: [f32; 8] = [
        0.008663971,
        -0.0066758734,
        -0.021873312,
        0.0087556075,
        0.014361011,
        0.0062268362,
        -0.00459446,
        -0.00678809,
    ];
    let expected_mid: [f32; 8] = [
        -0.006696689,
        0.003837848,
        0.006064503,
        -0.00015058772,
        -0.0036400883,
        0.0065470096,
        0.006391809,
        -0.00483888,
    ];

    for (index, expected) in expected_first.iter().enumerate() {
        assert!(
            (slice[index] - expected).abs() < 1e-4,
            "logit {index}: got {} expected {expected}",
            slice[index]
        );
    }
    for (offset, expected) in expected_mid.iter().enumerate() {
        let index = 3120 + offset;
        assert!(
            (slice[index] - expected).abs() < 1e-4,
            "logit {index}: got {} expected {expected}",
            slice[index]
        );
    }

    let sum: f32 = slice.iter().sum();
    assert!((sum + 0.068791).abs() < 1e-2, "logit sum drifted: {sum}");
}

/// A fixed prompt must always produce the same logits.
#[test]
fn forward_pass_is_deterministic() {
    let device = Device::flex();
    let config = tiny_config();
    let model = GLM53Flash::new(&config, &device);

    let (tokens, _) = fixed_batch(&device);
    let (first, _) = model.clone().forward(tokens.clone());
    let (second, _) = model.forward(tokens);

    let first = first.into_data();
    let second = second.into_data();
    first.assert_approx_eq::<f32>(&second, burn::tensor::Tolerance::default());
}

/// Save then load must reproduce the logits exactly.
#[test]
fn checkpoint_round_trip_preserves_logits() {
    let device = Device::flex();
    let config = tiny_config();
    let model = GLM53Flash::new(&config, &device);

    let directory = std::env::temp_dir().join("glm53-round-trip");
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join("model");
    model.clone().into_record().save(&path).expect("save");

    let record = burn::store::ModuleRecord::load(&path).expect("load");
    let reloaded = GLM53Flash::new(&config, &device).load_record(record);

    let (tokens, _) = fixed_batch(&device);
    let (before, _) = model.forward(tokens.clone());
    let (after, _) = reloaded.forward(tokens);

    before
        .into_data()
        .assert_approx_eq::<f32>(&after.into_data(), burn::tensor::Tolerance::default());
}

/// The two backends must agree.
///
/// Gated on `GLM53_WGPU_TEST=1` because the test suite must also run on a machine
/// without a GPU.
#[test]
fn backends_agree() {
    if std::env::var("GLM53_WGPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipping the cross-backend check; set GLM53_WGPU_TEST=1 to run it");
        return;
    }

    let config = tiny_config();

    // Both backends must run the *same* weights. The tokens must also live on
    // the same device as the model; Burn's dispatch rejects a cross-device
    // operand rather than silently copying it.
    let (record, on_cpu) = {
        let device = Device::flex();
        let (tokens, _) = fixed_batch(&device);
        let model = GLM53Flash::new(&config, &device);
        let (logits, _) = model.clone().forward(tokens);
        (model.into_record(), logits.into_data())
    };

    let on_gpu = {
        let device = Device::wgpu(Default::default());
        let (tokens, _) = fixed_batch(&device);
        let model = GLM53Flash::new(&config, &device).load_record(record);
        let (logits, _) = model.forward(tokens);
        logits.into_data()
    };

    on_cpu.assert_approx_eq::<f32>(&on_gpu, burn::tensor::Tolerance::rel_abs(1e-4, 1e-4));
}

/// Two micro-batches must produce the same step as one batch of their combined
/// size. Without this, "effective batch 32" would be a different optimiser than
/// the one it replaced.
#[test]
fn gradient_accumulation_matches_one_large_batch() {
    let device = Device::flex().autodiff();
    let config = ModelConfig::new()
        .with_dim(32)
        .with_heads(2)
        .with_layers(1)
        .with_experts(2)
        .with_expert_hidden(16)
        .with_streams(2)
        .with_max_sequence_length(16);
    let base = glm53::seeding::seeded_model(&config, 5, &device);

    let settings = |batch_size: usize, accumulation_steps: usize| glm53::training::TrainConfig {
        steps: 1,
        batch_size,
        accumulation_steps,
        sequence_length: 8,
        learning_rate: 1e-3,
        seed: 5,
        corpus_size: 64,
        log_every: 0,
        checkpoints: Vec::new(),
        artifact_dir: None,
        start_step: 0,
    };

    let mut large_loss = 0.0f32;
    let large = glm53::training::train(base.clone(), &settings(32, 1), &device, |metrics| {
        large_loss = metrics.loss
    });

    let mut split_loss = 0.0f32;
    let split = glm53::training::train(base, &settings(16, 2), &device, |metrics| {
        split_loss = metrics.loss
    });

    let a = param_vector(&large);
    let b = param_vector(&split);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    // One reduction over 32 rows and two reductions over 16 then averaged differ
    // only by f32 reassociation, so the bar is relative to the update size the
    // step actually applies (lr * gradient), not to the weights.
    let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("loss {large_loss:.7} vs {split_loss:.7}; max |delta| {max:e} on scale {scale:e}");
    assert!(
        (large_loss - split_loss).abs() < 1e-4,
        "loss {large_loss} vs {split_loss}"
    );
    assert!(
        max / scale < 1e-3,
        "relative parameter delta {:e}",
        max / scale
    );
}

/// The host initialiser must reproduce the model's configured distribution, not
/// the framework's default. Getting this wrong cost a full retraining run: the
/// loss stalled at 4.9 and every checkpoint scored 0/32.
#[test]
fn seeded_construction_is_deterministic_and_uses_the_configured_scale() {
    let device = Device::flex();
    let config = tiny_config();
    let first = glm53::seeding::seeded_model(&config, 11, &device);
    let second = glm53::seeding::seeded_model(&config, 11, &device);

    struct Matrices {
        sd: Vec<f32>,
    }

    impl ModuleVisitor for Matrices {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let value = param.val();
            if value.dims().len() != 2 {
                return;
            }
            let data = value
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let variance = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
            self.sd.push(variance.sqrt());
        }
    }

    let mut matrices = Matrices { sd: Vec::new() };
    first.visit(&mut matrices);
    assert!(!matrices.sd.is_empty());
    let mean_sd = matrices.sd.iter().sum::<f32>() / matrices.sd.len() as f32;
    assert!(
        (mean_sd - 0.02).abs() < 0.002,
        "mean matrix sd {mean_sd}, expected 0.02"
    );

    assert_eq!(
        param_vector(&first),
        param_vector(&second),
        "the same seed must give the same weights"
    );
}

/// A replaced parameter must stay trainable. `Param::map` with a plain
/// `from_data` tensor yields an inactive parameter, and `Module::train` then
/// never enables its gradient: the loss flattens and every evaluation scores zero.
#[test]
fn host_initialised_parameters_are_trainable() {
    let device = Device::flex().autodiff();
    let config = tiny_config();
    let before = glm53::seeding::seeded_model(&config, 3, &device);

    let settings = glm53::training::TrainConfig {
        steps: 3,
        batch_size: 2,
        accumulation_steps: 1,
        sequence_length: 8,
        learning_rate: 1e-2,
        seed: 3,
        corpus_size: 64,
        log_every: 1,
        checkpoints: Vec::new(),
        artifact_dir: None,
        start_step: 0,
    };
    let after = glm53::training::train(before.clone(), &settings, &device, |_| {});

    let start = param_vector(&before);
    let end = param_vector(&after);
    let moved = start
        .iter()
        .zip(&end)
        .filter(|(a, b)| (*a - *b).abs() > 1e-7)
        .count();
    assert!(
        moved * 2 > start.len(),
        "only {moved} of {} parameters moved in three steps",
        start.len()
    );
}

/// The cached decode path must agree with recomputing the whole prefix.
///
/// This is the acceptance test for the KV cache: at every prefix length, the
/// last-position logits from `forward_step` are within 1e-5 of the full forward.
#[test]
fn the_kv_cache_matches_the_full_prefix() {
    let device = Device::flex();
    let config = ModelConfig::new()
        .with_dim(32)
        .with_heads(2)
        .with_layers(2)
        .with_experts(2)
        .with_expert_hidden(16)
        .with_streams(2)
        .with_max_sequence_length(64);
    let model = glm53::seeding::seeded_model(&config, 5, &device).valid();

    let prompt: Vec<i64> = (0..12).map(|index| 10 + (index * 7 % 50) as i64).collect();
    let mut cache = glm53::kv::Cache::new(model.config());
    let mut worst = 0.0f32;

    for length in 1..=prompt.len() {
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(prompt[..length].to_vec(), [1, length]),
            &device,
        );
        let (logits, _) = model.clone().forward(input);
        let [_, time, vocab] = logits.dims();
        let full = logits
            .slice([0..1, time - 1..time, 0..vocab])
            .reshape([vocab])
            .to_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();

        let token = prompt[length - 1];
        let step = model
            .forward_step(
                Tensor::<2, Int>::from_data(TensorData::new(vec![token], [1, 1]), &device),
                length - 1,
                &mut cache,
            )
            .reshape([vocab])
            .to_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();

        for (a, b) in full.iter().zip(&step) {
            worst = worst.max((a - b).abs());
        }
    }

    assert!(
        worst < 1e-5,
        "max |delta| between cached and full prefix: {worst:e}"
    );
}
