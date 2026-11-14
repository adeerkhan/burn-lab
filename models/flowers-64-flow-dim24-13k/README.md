# flowers-64-flow-dim24-13k

The best 64x64 flower model this workspace has produced, and the first one to
beat the **pixel-FID 4.0** target: **3.170 +- 0.074** against a real-image floor
of **2.957 +- 0.234**.

It is the same UNet and the same data as
[`flowers-64-dim24-13k`](../flowers-64-dim24-13k), trained with a rectified flow
objective instead of DDPM noise prediction: the network predicts a velocity field
and sampling integrates it from noise to data.

| | |
| --- | --- |
| Parameters | 5,277,891 |
| Architecture | `Unet`, `dim 24`, `dim_mults (1,2,4,8)`, channels 3 |
| Objective | rectified flow, velocity = data - noise, MSE |
| Training | 13,000 steps, batch 16, AdamW 2e-4, EMA 0.999, 2,064.8 s |
| Data | 3,303 of the 3,670 `flower_photos` images, centre-cropped to 64x64 |
| Checkpoint | `model.bpk`, 21 MB, `sha256 164275f97f231637c1c4af483b5db148fe5d24b8aa99f21f6fd073ffc21a38ee` |

Sampling integrates `dx/dt = v(x, t)` with the midpoint rule, so each step costs
two network evaluations. Held-out scores, 512 samples per draw:

| Steps (NFE) | pixel-FID | nearest L1 | gradient ratio | colour distance |
| ---: | ---: | ---: | ---: | ---: |
| 10 (20) | 3.264 | 0.333 | 1.043 | 0.109 |
| 20 (40) | **3.170 +- 0.074** | 0.334 | 1.063 | 0.101 |
| 50 (100) | 3.224 | 0.335 | 1.073 | 0.094 |
| real images vs held out (floor) | 2.957 +- 0.234 | 0.329 | 1.000 | 0.117 |

Ten steps already land at 3.26; twenty are the best point. The 20-step row is
three draws (3.096, 3.172, 3.243); the others are single draws.

## Sample from it

```sh
git clone https://github.com/adeerkhan/burn-lab
cd burn-lab

cargo run -p diffusion --release -- sample --backend wgpu \
  --ckpt models/flowers-64-flow-dim24-13k --samples 16 --sampling-steps 20
```

`--sampling-steps` here is the number of ODE steps, not a DDPM chain length; any
count from 2 upward is valid, and 20 is the measured sweet spot.

## What the samples are, and are not

Clearer than the DDPM samples at the same configuration: daisies with yellow
centres, sunflowers, poppies, rose clusters, foliage and sky. A few images
degenerate into colour masses, more often at 10 steps than at 20. They are still
painterly, not photographs, and 64x64 does not resolve individual petals. The
full study and the comparison against DDPM and DDIM are in
[`diffusion/README.md`](../../diffusion/README.md).
