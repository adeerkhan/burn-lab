# glm53 — GLM-5.3-Flash-from-scratch, ported to Burn

A Burn reimplementation of
[`vukrosic/glm-5.3-flash-from-scratch`](https://github.com/vukrosic/glm-5.3-flash-from-scratch):
a 25.7M-parameter, GLM-5.3-Flash-inspired language model pretrained from random
initialization on a synthetic Python corpus, then post-trained with RLOO against
executable unit-test rewards.

## What is ported

| Component | Reference | Here | Notes |
| --- | --- | --- | --- |
| Parameters | 25,730,592 | **25,730,592** | Exact match |
| Active per token | 9,805,344 | **9,805,344** | Exact match |
| Layers | 12 | 12 | 9 linear + 3 sparse attention |
| Experts | 8, top-2, +1 shared | same | |
| Residual streams | 4 | 4 | |
| Tokenizer | byte-level, 260 tokens | same | |
| Evaluator | AST allowlist + executable tests | same | via a `python3` harness |
| Training | pretrain + RLOO | same | |
| Vision path | patch encoder + projector | not ported | see below |

```text
bytes -> embedding -> four residual streams
                       |
        [linear attention + MoE] x 3
                       |
        [sparse attention + MoE] x 1
                       |
                  repeat x 3
                       |
              RMSNorm -> tied LM head
```

## Verified

- **Parameter counts match the reference to the digit**, total and active. That
  only works if the architecture and the tied LM head are both correct.
- **The linear-attention reformulation is numerically checked** against a direct
  transcription of the reference's cumulative-sum definition, including the
  `elu(x) + 1` feature map.
- **Sparse and dense expert dispatch agree to 1e-5**, and only the routed experts
  receive gradient.
- **Gradients match finite differences** on sampled parameters. This is the check
  that matters most: a subtly wrong gradient still produces a falling loss curve.
- **Checkpoints round-trip**, both backends agree to 1e-4, and a fixed prompt is
  deterministic.
- **44 tests** across the library and the integration suite.

## Results

All figures on this port's own corpus and splits, so they are not the
reference's numbers. Dev is the split used for checkpoint selection; confirm was
opened once, for the final report only.

### Pretraining

400 steps, batch 16 x 128 tokens, AdamW, lr 3e-4, 155 s on an RTX 5060 via
wgpu/Vulkan. Loss 5.09 → 0.211.

### Checkpoint selection (dev, greedy pass@1)

| Checkpoint | dev pass@1 |
| --- | --- |
| 100 | 0.094 (3/32) |
| 200 | 0.906 (29/32) |
| 400 | **1.000 (32/32)** |

Dev saturates, exactly as the reference reports for its own benchmark.

### RL (RLOO, `last-block-head`, 32 groups x 8 rollouts, T=0.4, lr 5e-5)

RL starts from **checkpoint 200**, not 400. Two reasons, both measured:

- From checkpoint 100 every sampled rollout was invalid, so every group had zero
  reward spread and RLOO had nothing to learn from. Two runs (T=0.35 group 4,
  T=0.2 group 8) produced 30 and 32 no-signal groups respectively, and dev
  pass@1 stayed at 3/32.
- From checkpoint 400 dev is already 1.0, leaving no headroom to measure.

At checkpoint 200 the policy produces valid code with variation, which is what
RLOO needs.

| Metric | pre-RL | post-RL |
| --- | --- | --- |
| dev pass@1 | 0.906 (29/32) | **0.938 (30/32)** |
| dev pass@8 | 0.875 (28/32) | 0.875 (28/32) |
| dev exact rollouts | 0.812 | 0.781 |
| **confirm pass@1** | **0.906 (29/32)** | 0.844 (27/32) |
| **confirm pass@8** | **0.875 (28/32)** | 0.813 (26/32) |
| confirm exact rollouts | 0.816 | 0.785 |

**RL helped on the selection split and hurt on the held-out one.** One task
better on dev, two worse on confirm, at both pass@1 and pass@8. That is
overfitting to the tasks RLOO trained on, and it is the honest result: 32 groups
of binary reward on a 25M model is not enough to generalise, and the reference
reports the same pattern on its untrained families.

## Usage

```sh
# Parameter counts and the layer mix
cargo run -p glm53 --release -- params

# Pretrain, checkpointing along the way
cargo run -p glm53 --release -- train --steps 400 --batch-size 16 \
  --checkpoints 100,200,400 --out artifacts/glm53-run

# Pick the best checkpoint on dev
cargo run -p glm53 --release -- select --out artifacts/glm53-run --per-family 4

# Greedy pass@1 and sampled pass@k
cargo run -p glm53 --release -- evaluate \
  --ckpt artifacts/glm53-run/checkpoint-0200 --split dev --per-family 4
cargo run -p glm53 --release -- evaluate-passk \
  --ckpt artifacts/glm53-run/checkpoint-0200 --split dev --per-family 4 \
  --samples 8 --temperature 0.4

# RLOO
cargo run -p glm53 --release -- rl \
  --ckpt artifacts/glm53-run/checkpoint-0200 --out artifacts/glm53-rl \
  --groups 32 --group-size 8 --temperature 0.4 --learning-rate 5e-5 \
  --train-scope last-block-head --reward-mode binary
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--backend` | `wgpu` | `wgpu` (Vulkan) or `flex` (pure Rust CPU) |
| `--steps` | `400` | Pretraining steps |
| `--batch-size` | `16` | Dense expert dispatch caps this at 16 on a 6 GiB GPU |
| `--seq-len` | `128` | Sequence length |
| `--checkpoints` | — | Steps to checkpoint, e.g. `100,200,400` |
| `--split` | `dev` | `dev`, `final`, `confirm`, `rl` |
| `--per-family` | `4` | Tasks per family |
| `--samples` | `8` | Samples per task for pass@k |
| `--groups` / `--group-size` | `32` / `4` | RL budget |
| `--train-scope` | `last-block-head` | `all`, `last-block`, `last-block-head` |
| `--reward-mode` | `binary` | `binary` or `case-fraction` |

`--train-scope last-block-head` freezes everything except the last block, the
final norm and the embedding (which is also the tied head): **2,190,152 of
25,730,592 parameters** are trainable under RLOO.

## Deviations from the reference

- **The evaluator shells out to `python3`.** The reference validates with a
  Python AST allowlist and then *executes* the candidate against unit tests. The
  port runs an embedded harness (`python3/evaluator_harness.py`) that mirrors
  `evaluator.py` exactly — same DENIED nodes, `SAFE_CALLS`, restricted builtins,
  2048-byte cap and strict `type(actual) is type(expected)`. Requests are batched
  so one interpreter start serves many candidates. The alternative was embedding
  a second Python runtime.
- **Sparse attention masks instead of gathering.** The reference gathers
  `k[b, h, indices[b, h, t, w], :]`, a rank-5 result that Burn's `gather` cannot
  produce. A gathered softmax over a selected set equals a masked softmax over
  all keys, so the port adds a `[1, 1, t, t]` additive mask. Same numbers,
  smaller intermediate.
- **The linear attention contracts instead of accumulating.** The reference
  materialises `kv[t] = sum_{s<=t} k_s (x) v_s` (rank 5 in Burn terms); the port
  masks a full contraction, which is the same quantity. Checked numerically.
- **Dense expert dispatch is the default.** The reference dispatches only the
  selected experts. Both are implemented and agree to 1e-5, but the sparse path
  is ~30x slower here because every expert sees a different row count and the
  JIT backend recompiles per shape. Sparse is what lets batch 32 fit; dense is
  what makes it fast.
- **The LM head reads the embedding weight directly.** Burn's module tree has no
  weight tying, so the head is `hidden @ embedding.weight.T` rather than a module.
  That is what makes the parameter count match.
- **The corpus generator is deterministic but not byte-identical.** Same eight
  families and prompt shape, a different PRNG.
- **No vision path.** `vision.py` and `train_vision.py` are not ported; that is a
  separate model path, not part of the language-model work. The reference's
  pretraining ablations (`experiments/`) are likewise out of scope.

## Performance notes

| Stage | Wall clock | Peak GPU memory added |
| --- | --- | --- |
| Pretrain 400 steps (wgpu) | 155 s | fits in ~4 GiB free at batch 16 |
| Pretrain 400 steps (flex) | ~32 min | n/a |
| Pretrain 5 steps, batch 4 x 64 (wgpu) | 4 s | +724 MiB |
| Greedy pass@1, 32 tasks (wgpu) | 16 s | not measured |
| pass@8, 32 tasks (wgpu) | 46 s | not measured |
| RLOO 32 groups x 8 (wgpu) | ~4 min | +241 MiB (measured on a 2-group run) |

Burn does not expose allocator statistics, so `metrics::PeakMemory` polls
`nvidia-smi` every 250 ms and reports the peak alongside the baseline it saw at
start-up. That baseline includes whatever else is using the device, so the
*delta* is the meaningful figure.

The batch-16 ceiling is real: batch 32 exhausts the device with dense dispatch
and fits with sparse dispatch. Measurements taken while another process held
~3 GB of the GPU are therefore pessimistic, and one OOM cascade during
development was caused by an unrelated tenant rather than by this port.

## Tests

```sh
cargo test -p glm53 --release
GLM53_WGPU_TEST=1 cargo test -p glm53 --release --test model_checks backends_agree
```

Covers the tokenizer, the corpus generator, both attention masks, the evaluator's
rejection rules, the linear-attention equivalence, generation determinism, the
MoE usage distribution, routed-only gradients, sparse-vs-dense dispatch, the
finite-difference gradient check, checkpoint round-trips and the cross-backend
differential.

## License

The port follows the reference's MIT license.
