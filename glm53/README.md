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
| Vision path | patch encoder + projector | same | 263-token vocabulary; see below |

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
- **Initialization is seeded** at every model construction. Before that, the same
  arm and seed started from different weights on every run — which moved a
  10-seed study's p-value from 0.008 to 0.43. A test now fails if two identical
  runs diverge. That test is marked ignored because `Device::seed` is
  process-global and the harness runs tests in parallel, so another test can draw
  from the same device between the seed and the construction; run
  `cargo test -- --ignored --test-threads=1` to exercise it. The study CLI is
  single-threaded, so the studies themselves are deterministic.
- **97 tests** across the library and the integration suite, one of them ignored
  (see the checkpoint-resume gap below).

## Results

All figures on this port's own corpus and splits, so they are not the
reference's numbers. Dev is the split used for checkpoint selection; confirm was
opened once, for the final report only.

### Pretraining

400 steps, batch 16 x 128 tokens, AdamW, lr 3e-4, on an RTX 5060 via wgpu/Vulkan.
Loss 5.09 → 0.211.

### Effective batch size

The reference trained with batch 32, so its step 100 saw 345,995 tokens; ours saw
204,800. Two micro-batches of 16 accumulated per step doubles the tokens without
doubling the activation memory:

| Run | Tokens at 400 steps | Dev pass@1 at 200 | at 400 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| 1 x 16 | 819,200 | **29/32** | 32/32 | 4.2 GiB |
| 2 x 16 (effective 32) | 1,638,400 | 17/32 | 32/32 | 7.8 GiB |

**Matching the reference's token budget did not help.** The larger effective batch has
the same number of optimizer steps but a less noisy gradient, so at a fixed step count
it is further behind, and both runs converge by step 400. Accumulation also cost memory
rather than saving it (7.8 against 4.2 GiB): the first micro-batch's gradients stay
resident while the second graph is built. Accumulation buys a larger batch, not a
cheaper one.

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

## Vision

The image path is the reference's topology at teaching scale: RGB patches into two
non-causal vision blocks, a 2x2 spatial merge, and a gated projector that emits
language-model-width tokens. It appends ids 260/261/262, so it needs a 263-token
vocabulary; the text model is untouched, and a test proves the shared logits are
unchanged to 1e-5 after the vocabulary is widened.

![Both image paths and the 25M pilot](../docs/vision.webp)

At this scale the **direct-patch baseline wins**, and it wins in every seed we ran:
87.5% against 56-76.5% for the faithful tower on held-out digits.

The reference's own run reproduces closely. Using the reference implementation itself
(PyTorch, CPU, same config), the faithful tower scored **112/200 (56.0%)** and the
baseline **187/200 (93.5%)**; our faithful tower scored **112/200 (56.0%)** and the
baseline **175/200 (87.5%)**. The faithful towers agree to the example, which is the
strongest evidence that the port is faithful.

**Five seeds, because one is not enough to rank these architectures.** Seeds 42, 7,
13, 21 and 34, 120 steps, 200 held-out examples:

| Configuration | Mean | sd | Per-seed |
| --- | ---: | ---: | --- |
| Direct-patch baseline (42,500 params) | **92.7%** | 5.6 | 95.5, 85.0, 100, 91.5, 91.5 |
| Faithful, width 24 (60,924 params) | 55.6% | 17.6 | 31.0, 72.0, 58.0, 71.5, 45.5 |
| Faithful, width 48 (101,844 params) | 48.7% | 16.1 | 22.0, 60.0, 45.5, 61.0, 55.0 |

The direction is unambiguous — added structure does not pay for itself at this budget,
and widening the tower makes it worse, not better. But the faithful tower's seed-to-seed
sd is three times the baseline's, so any single-seed difference, including the
reference's 119/200 against 189/200, is one draw from a wide distribution.

The full-model pilot attaches the tower to the 25.7M language model and freezes
everything except the vision tower, the last block, the final norm and the tied
embedding — 2.3M of 25.7M parameters. It learns the digit task through a saturated
coding model, but 40 steps plateaued at chance (10%) and 200 steps reached 32%.

## The miniature studies

Two matched studies over an arithmetic corpus (add/sub/mul/neg/abs), each with a
~248k-parameter model, identical optimiser and step budget, and only the data
presentation changing.

![Data diversity and curriculum ordering](../docs/experiments.webp)

Ten paired seeds, 200 steps, batch 24, with initialization seeded per arm so all
arms start from the same weights for a given seed:

| Study | Result |
| --- | --- |
| Data diversity, 16 vs 8 structures | +2.20 points (95% CI +0.3 to +4.5), permutation p = 0.078, Holm p = 0.31 |
| Data diversity, 32 vs 8 structures | +2.35 points (95% CI +0.3 to +4.9), p = 0.0625, Holm p = 0.31 |
| Data diversity, 48 vs 8 structures | +2.84 points (95% CI +0.02 to +6.4), p = 0.14, Holm p = 0.43 |
| Data diversity, 64 vs 8 structures | +1.65 points (95% CI -1.0 to +4.4), Holm p = 0.59 |
| Data diversity, 88 vs 8 structures | +1.81 points (95% CI -1.1 to +5.3), Holm p = 0.59 |

**No diversity effect survives Holm correction.** The point estimates are consistently
positive (+1.6 to +2.8 points) and peak around 32-48 structures, but ten seeds cannot
separate any arm from 8. Exact-expression accuracy was 0% in every arm.

**Correction.** An earlier run reported +3.8 points with Holm p = 0.0078 for 32 vs 8.
That number was not reproducible: model initialization was never seeded, so each arm
and seed started from different weights and the pairing was only in the data. With the
fix, the same comparison is +2.35 points at Holm p = 0.31. The initialization variance
was larger than the effect being measured.

### Curriculum and ordering

The same examples, the same budget, only the presentation order changes:

| Arm | Mean byte accuracy | sd |
| --- | --- | --- |
| curriculum 8 to 88 | 58.9% | 2.1 |
| diverse-interleaved | 58.5% | 2.0 |
| diverse-blocked | 50.2% | 2.5 |

| Comparison | Result |
| --- | --- |
| Blocked vs interleaved | -8.77 points (95% CI -10.4 to -7.3), p = 0.0020, Holm p = 0.0039 |
| Interleaved vs curriculum | -0.39 points (95% CI -1.9 to +1.1), Holm p = 0.65 |

The two factors do not interact. Crossing them at 8 and 48 structures, ten seeds:

| | 8 structures | 48 structures |
| --- | ---: | ---: |
| Interleaved | 57.6% | 60.7% |
| Blocked | 39.4% | 44.3% |
| Cost of blocking | **-18.2 pts** | **-16.4 pts** |

The difference in differences is **+1.8 points (95% CI -0.7 to +4.3, p = 0.23)**: the
cost of blocked ordering is the same whether the corpus holds 8 structures or 48, so
this is one mechanism, not two findings wearing the same name.

**Blocked ordering is the one robust result in the study set.** Presenting the same
examples in blocks costs 8.8 points against interleaving them, and it survives Holm
correction in every run we have made. Curriculum ordering and interleaving are
indistinguishable — the earlier -1.6 point gap was initialization noise.

## RL: where is the reward signal, and does any arm move it?

The earlier screen ran at the 200-step checkpoint and every arm tied. That was not a
fact about RLOO: dev was already 29/32, so nearly every rollout passed and the
advantage was noise. So measure the signal first. Sampled pass@16 on the RL split, 512
rollouts per checkpoint, all from one seeded run:

| Step | Exact rollouts | Step | Exact rollouts |
| ---: | ---: | ---: | ---: |
| 100 | 3.3% | 150 | 38.9% |
| 110 | 2.1% | 160 | 67.6% |
| 120 | 10.7% | 170 | 74.6% |
| 130 | 16.8% | 180 | 84.6% |
| 140 | **33.6%** | 200 | 86.7% |

The band is early and narrow. Step 140 gives 33.6% exact rollouts (pass@16 47%), the
closest to a balanced reward, and the screen runs there.

### Eight arms inside the band

16 groups x 8 rollouts, 32 dev tasks, from the step-140 checkpoint. The baseline arm
saw signal in 7 of 16 groups and 64 of 128 rollouts were exactly correct, so the arms
had something to learn from.

| Arm | Signal | Exact | Dev pass@1 |
| --- | --- | --- | --- |
| baseline (binary, -0.1, T=0.35, group 8) | 7/16 | 64/128 | 0.688 |
| partial-test reward | 6/16 | 65/128 | 0.688 |
| invalid penalty 0 | 6/16 | 63/128 | 0.688 |
| invalid penalty -1 | 7/16 | 64/128 | 0.656 |
| temperature 0.20 | 2/16 | 72/128 | 0.656 |
| temperature 0.80 | 3/16 | 0/128 | 0.656 |
| group size 4 | 6/16 | 32/64 | 0.688 |
| group size 16 | 11/16 | 130/256 | 0.625 |

| Arm | only arm | only reference | mean difference | Holm p |
| --- | ---: | ---: | ---: | ---: |
| partial-reward | 0 | 0 | +0.000 | 1.000 |
| no-penalty | 0 | 0 | +0.000 | 1.000 |
| harsh-penalty | 0 | 1 | -0.031 | 1.000 |
| low-temp | 0 | 1 | -0.031 | 1.000 |
| high-temp | 0 | 1 | -0.031 | 1.000 |
| group-4 | 0 | 0 | +0.000 | 1.000 |
| group-16 | 0 | 2 | -0.062 | 1.000 |

**No arm separates: every comparison is Holm p = 1.00 and the largest difference is two
tasks out of 32.** This null is not a placement artefact — the policy solves half its
rollouts from the same prompt and still none of RLOO's design choices move greedy dev
accuracy. At 25.7M parameters on eight operation families, the design surface of the
algorithm is flat. With 32 dev tasks, exact McNemar needs six net discordant tasks for
p < 0.05; two is noise by construction.

## Receipts

Every checkpoint directory carries `config.json`, `metadata.json` and a sha256 of
its weights, and loading verifies the hash: a tampered file is a hard error.
`system-receipt` captures the OS, CPU, GPU, driver, toolchain and model config that
produced a result.

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
- **Masked language-model loss is hand-rolled.** Burn's cross-entropy has no
  `ignore_index`, so the mask is applied to a gathered log-probability and the mean
  is taken over valid positions only.
- **Placeholder replacement is a splice, not a scatter.** The reference scatters
  visual embeddings into placeholder positions with `masked_scatter`; Burn has no
  equivalent, and the image span is always contiguous, so the port concatenates
  around it. Same result, no host round-trip.
- **The vision and experiment runs use `flex`, not `wgpu`.** wgpu took 58 s for two
  vision training steps where `flex` took 0.3 s: the tiny model presents many new
  shapes and the JIT backend recompiles instead of reusing kernels.
- **The arithmetic corpus and digit renderer use this crate's PRNG.** Same
  distributions, different bytes.
- **The language model initialises from our own RNG.** Burn's initialisers draw from a
  process-global RNG, so the same arm and seed started from different weights on every
  run. Every rank-2 parameter is now drawn at `N(0, 0.02)` — the distribution this
  model configures — and marked trainable. Vision still initialises through Burn's
  initialiser under a process lock, which is deterministic for the single-threaded CLI.
- **The RL checkpoint is chosen by measured signal, not by dev pass@1.** The reference
  selected step 100 on dev; here the reward-signal rate is measured per checkpoint
  (sampled pass@16 on the RL split) and step 140 is selected for a ~33% exact-rollout
  rate. Selecting on dev alone put us either side of the usable band twice.

## Performance notes

| Stage | Wall clock | Peak GPU memory |
| --- | --- | --- |
| Pretrain 400 steps (wgpu) | 155 s | fits batch 16 |
| Vision comparison, 120 steps (flex) | 12.3 s | CPU |
| Vision smoke, 40 steps (flex) | 1.0 s | CPU |
| Vision pilot, 200 steps (flex) | 160.6 s | CPU |
| Data diversity, 30 arms (flex) | 123.2 s | CPU |
| Curriculum, 30 arms (flex) | 148.7 s | CPU |
| RL variant screen, 8 arms (wgpu) | 631.6 s | not sampled |
| Group confirmation, 6 runs (wgpu) | ~350 s | +489 MiB |
| RL, 96 groups (wgpu) | 282 s | +2725 MiB |


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

The statistics are pinned against hand-computed values — exact McNemar tails,
exhaustive paired permutation, Holm step-down and the bootstrap interval — so a
silent change in an analyzer fails a test instead of altering a published table.
The sequence-length boundary is tested explicitly: `time == max_sequence_length`
runs, `time == max + 1` panics with a message naming the limit.

**Known gap: no exact resume.** Checkpoints carry weights only. `TrainConfig::start_step`
now replays the sampler so a continued run sees the batches it would have seen, which
moved the divergence after two steps from `max |delta| = 2.786e-3` to `2.766e-3` over
26k parameters — so the data order accounts for under 1% of it and the missing AdamW
moments are essentially the whole gap. Burn's `AdamW` exposes no state and is not a
`Module`, so closing this needs a state accessor in `burn-optim` rather than a change
here. `tests/resume_check.rs` holds the requirement as an ignored test.

## License

The port follows the reference's MIT license.
