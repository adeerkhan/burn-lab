# flowers-64-dim24-13k

The best 64x64 flower model this workspace produced. On the held-out split it
scores **pixel-FID 4.761 +- 0.346** with the full DDPM chain and **5.131 +- 0.233**
with DDIM 50, against a real-image floor of **2.957 +- 0.234**.

One set of weights. DDIM and DDPM are two ways to run it, not two models.

| | |
| --- | --- |
| Parameters | 5,277,891 |
| Architecture | `Unet`, `dim 24`, `dim_mults (1,2,4,8)`, channels 3 |
| Diffusion | sigmoid schedule, 200 timesteps, v-prediction |
| Training | 13,000 steps, batch 16, AdamW 2e-4, EMA 0.999 updated every step |
| Data | 3,303 of the 3,670 `flower_photos` images, centre-cropped to 64x64 |
| Checkpoint | `model.bpk`, 21 MB, `sha256 2d2f1f4f9ff79e7f8a427b45b1e1e148abf7f34143a51fbfed536289984fd083` |

Held-out scores, 512 samples per draw, three draws per sampler where a spread is shown:

| Sampler | pixel-FID | nearest L1 | gradient ratio | colour distance |
| --- | ---: | ---: | ---: | ---: |
| DDIM 20 | 7.262 | 0.286 | 0.818 | 0.263 |
| DDIM 50 | 5.131 +- 0.233 | 0.313 | 0.939 | 0.175 |
| **DDPM 200** | **4.761 +- 0.346** | 0.315 | 0.966 | 0.145 |
| real images vs held out (floor) | 2.957 +- 0.234 | 0.329 | 1.000 | 0.117 |

The best single DDPM draw scored 4.474; the three draws were 4.474, 5.147 and
4.661, so the mean is what is claimed here, not the best draw.

## Sample from it

```sh
git clone https://github.com/adeerkhan/burn-lab
cd burn-lab

# DDIM, 50 steps
cargo run -p diffusion --release -- sample --backend wgpu \
  --ckpt models/flowers-64-dim24-13k --samples 16 --sampling-steps 50

# DDPM, full 200-step chain
cargo run -p diffusion --release -- sample --backend wgpu \
  --ckpt models/flowers-64-dim24-13k --samples 16 --sampling-steps 200
```

The command writes `sample-ddim-50.png` or `sample-ddpm-200.png` beside the
checkpoint. The `data` field in `config.json` points at the training file used
during the run; if it is absent the sampler still works, but the printed
nearest-training distance is skipped.

## What the samples are, and are not

Painterly flower masses: sunflower heads, poppy fields, rose clusters, skies.
Structure and palette are right; they are not photographs, and 64x64 does not
resolve individual petals. The full study, including the estimator's floor, the
FID-versus-steps curve and the `dim 16` comparison, is in
[`diffusion/README.md`](../../diffusion/README.md).
