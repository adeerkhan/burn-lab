<div align="center">
  <img src="docs/hero.webp" alt="burn-lab: GLM-5.3-Flash from scratch, in Burn" width="100%"/>
  <br/><br/>

  ![Rust](https://img.shields.io/badge/Rust-1.95%2B-f15a24?style=flat-square&logo=rust&logoColor=white)
  ![Burn](https://img.shields.io/badge/Burn-0.23.0--pre.2-f15a24?style=flat-square)
  ![License](https://img.shields.io/badge/License-MIT-f15a24?style=flat-square)
  ![Tests](https://img.shields.io/badge/tests-44%20passing-f15a24?style=flat-square)

  <p><b>A 25.7M-parameter language model — hybrid attention, sparse mixture of experts, four residual
  streams — pretrained from scratch and post-trained with executable-reward RL, all in Burn.</b></p>
</div>

---

## What this is

A port of [`vukrosic/glm-5.3-flash-from-scratch`](https://github.com/vukrosic/glm-5.3-flash-from-scratch)
to [Burn](https://github.com/tracel-ai/burn): a GLM-5.3-Flash-inspired language model trained from
random initialization on a synthetic Python corpus, then fine-tuned with RLOO against executable
unit-test rewards.

The goal is not to beat the reference. It is to take a real language-model workload — the kind that
usually means PyTorch — and find out exactly where Burn holds up and where it does not. The
[port notes](#the-port) record every deviation, and the [findings](#what-we-learned-about-burn)
record every rough edge we hit.

The workspace also carries `burn-lab`, the original MNIST lab from the Burn Book's basic workflow,
which is where this started.

## Results

Every number below is measured from runs in this repository. See [`glm53/README.md`](glm53/README.md)
for the full tables, commands and seeds.

<img src="docs/loss.webp" alt="Pretraining loss, 5.09 to 0.211 over 400 steps" width="100%"/>

Pretraining takes **155 s** on an RTX 5060 through wgpu/Vulkan: 400 steps, batch 16 × 128 tokens,
AdamW at 3e-4. Loss falls from 5.09 — barely above `ln(260) = 5.56`, a uniform guess — to **0.211**.

<img src="docs/checkpoints.webp" alt="Dev pass@1 by checkpoint" width="100%"/>

Dev pass@1 saturates: 3/32 at step 100, 29/32 at 200, 32/32 at 400. The reference reports the same
saturation on its own benchmark, which is why both projects RL from an early checkpoint rather than
the final one.

<img src="docs/rl.webp" alt="RLOO before and after, dev and confirm" width="100%"/>

RLOO moves the policy **one task better on the selection split and two tasks worse on the held-out
split**, at both pass@1 and pass@8:

| Metric | pre-RL | post-RL |
| --- | --- | --- |
| dev pass@1 | 0.906 (29/32) | **0.938 (30/32)** |
| dev pass@8 | 0.875 (28/32) | 0.875 (28/32) |
| confirm pass@1 | **0.906 (29/32)** | 0.844 (27/32) |
| confirm pass@8 | **0.875 (28/32)** | 0.813 (26/32) |

That is overfitting, and it is the honest result. 32 groups of binary reward on a 25M model is not
enough to generalise; the reference reports the same pattern on its untrained families.

## Model

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

| Component | Reference | This port |
| --- | --- | --- |
| Total parameters | 25,730,592 | **25,730,592** |
| Active per token | 9,805,344 | **9,805,344** |
| Layers | 12 (9 linear + 3 sparse attention) | same |
| Experts | 8 routed, top-2, +1 shared | same |
| Residual streams | 4 | same |
| Tokenizer | byte-level, 260 tokens | same |
| Context | 192 | 192 |

The parameter counts match to the digit, which only happens if the architecture *and* the tied LM head
are both right.

## Quick start

```sh
git clone https://github.com/adeerkhan/burn-lab
cd burn-lab
cargo test --workspace --release          # 44 tests
cargo run -p glm53 --release -- params    # parameter counts
```

Train and evaluate the language model:

```sh
# Pretrain, checkpointing along the way
cargo run -p glm53 --release -- train --steps 400 --batch-size 16 \
  --checkpoints 100,200,400 --out artifacts/glm53-run

# Pick a checkpoint on dev, then measure it
cargo run -p glm53 --release -- select --out artifacts/glm53-run
cargo run -p glm53 --release -- evaluate \
  --ckpt artifacts/glm53-run/checkpoint-0200 --split dev
cargo run -p glm53 --release -- evaluate-passk \
  --ckpt artifacts/glm53-run/checkpoint-0200 --split dev --samples 8

# RLOO against executable reward
cargo run -p glm53 --release -- rl \
  --ckpt artifacts/glm53-run/checkpoint-0200 --out artifacts/glm53-rl \
  --groups 32 --group-size 8 --temperature 0.4 --learning-rate 5e-5
```

Sample a completion:

```sh
cargo run -p glm53 --release -- generate \
  --ckpt artifacts/glm53-run/checkpoint-0400 \
  --prompt "# Complete this Python function.
# Return x squared.
def square_abc(x):"
```

```python
def square_abc(x):
    return x * x
```

<details>
<summary><b>The MNIST lab</b></summary>
<br/>

The workspace root is the Burn Book's basic workflow: a CNN trained on MNIST with the `Learner`, a
terminal dashboard and a burnpack checkpoint.

```sh
cargo run --release -- train            # train, with the live TUI dashboard
cargo run --release -- test             # evaluate on the test set
cargo run --release -- infer --index 42 # predict a single digit
```

Burn's dashboard only renders when stdout is a terminal, so
`./scripts/dashboard.sh` runs the command inside `tmux` to get a real pty.

</details>

## Repository layout

```
burn-lab/
├── src/                  # the MNIST lab (Burn Book basic workflow)
├── scripts/              # tmux helper for the TUI dashboard
├── docs/                 # figures + the script that renders them
└── glm53/                # the GLM-5.3-Flash port
    ├── python/           # evaluator harness (AST allowlist + test runner)
    ├── src/
    │   ├── attention.rs      linear + gathered sparse attention
    │   ├── moe.rs            dense and sparse expert dispatch
    │   ├── block.rs          hybrid block, four-stream hyper-connection
    │   ├── model.rs          the language model
    │   ├── evaluator.rs      fail-closed executable verifier
    │   ├── generate.rs       batched sampling with early stop
    │   ├── evaluation.rs     pass@1 and pass@k
    │   ├── rl.rs             RLOO
    │   └── training.rs       pretraining with periodic checkpoints
    └── tests/            # gradient check, routing, golden fixture, round-trip
```

## The port

### What matches the reference

- **The model, exactly.** Same shapes, same parameter counts, same tied head.
- **The evaluator, exactly.** Same AST `DENIED` node list, `SAFE_CALLS` allowlist, restricted
  builtins, 2048-byte cap and strict `type(actual) is type(expected)` comparison.
- **The algorithm.** RLOO with leave-one-out advantages, mean per-token log-probabilities,
  `last-block-head` freezing, binary and case-fraction rewards, gradient clipping at 1.0.

### Where it deviates, and why

| Deviation | Why |
| --- | --- |
| The evaluator shells out to `python3` | The reference *executes* candidates against unit tests. Embedding a second Python runtime was the alternative; a batched subprocess keeps the semantics exact. |
| Sparse attention masks instead of gathering | The reference gathers into a rank-5 tensor that Burn's `gather` cannot produce. A gathered softmax over a selected set equals a masked softmax over all keys. |
| Linear attention contracts instead of accumulating | Same reason: the reference's rank-5 cumulative outer product becomes a masked contraction. Checked numerically against a transcription of the reference. |
| Dense expert dispatch is the default | Both are implemented and agree to 1e-5, but sparse is ~30× slower here — every expert sees a different row count and the JIT backend recompiles per shape. Sparse is what lets batch 32 fit; dense is what makes it fast. |
| The LM head reads the embedding weight | Burn's module tree has no weight tying, so the head is `hidden @ embedding.weight.T`. This is what makes the count match. |
| The corpus generator is its own PRNG | Same eight families and prompt shape, different random stream. |

### Verification

Nothing here is asserted without a check:

- **Finite-difference gradient check.** The highest-value test in the repo: a subtly wrong gradient
  still produces a falling loss curve, so only this catches it.
- **Sparse vs dense expert dispatch** agree to 1e-5, and only routed experts receive gradient.
- **A golden forward fixture** on a deterministic parameter pattern, so a rewiring refactor fails loudly.
- **The linear attention** matches a direct transcription of the reference's cumulative-sum definition.
- **Checkpoint round-trips** and the two backends agree to 1e-4.

## What we learned about Burn

Four things a language-model port actually trips over. Each is written up with symptom, cause and
workaround in the port notes.

| Finding | Severity |
| --- | --- |
| An OOM can return `SIGILL` rather than an `Err`, and can leave the wgpu device lost across processes | major |
| No weight tying in the module tree — a tied-embedding model cannot be a plain module tree | minor |
| `gather` cannot produce a rank-5 result, forcing gathered attention to be reformulated | minor |
| No `no_grad` equivalent — generation on an autodiff device records every forward and exhausts it | minor |

Plus a tail of ergonomics that each cost a compile cycle: `reshape::<D2, _>` needs both generic
arguments, every `*_dim` reduction keeps its dimension (boolean reductions included), `one_hot`
preserves tensor kind, `Int` defaults to I32, and `visit_float` does not name the parameter it visits.

None of it was fatal. The two things that genuinely surprised us in Burn's favour were `RmsNorm`
accepting any rank, and `RotaryEncoding` being numerically identical to a hand-written RoPE.

## Tests

```sh
cargo test -p glm53 --release                                                  # 44 tests
GLM53_WGPU_TEST=1 cargo test -p glm53 --release --test model_checks backends_agree
```

The cross-backend check is gated on an environment variable so the suite still runs on a machine
without a GPU.

## Figures

`docs/make_plots.py` renders every figure from the measured numbers, using Burn's palette. Regenerate
with:

```sh
python3 docs/make_plots.py && magick docs/*.png docs/*.webp
```

## Credits and license

The model, the synthetic corpus and the executable-reward RL method are
[`vukrosic/glm-5.3-flash-from-scratch`](https://github.com/vukrosic/glm-5.3-flash-from-scratch), MIT
licensed.

The framework is [Burn](https://github.com/tracel-ai/burn) by Tracel AI, MIT/Apache-2.0.

This port is MIT, matching the reference it derives from.
