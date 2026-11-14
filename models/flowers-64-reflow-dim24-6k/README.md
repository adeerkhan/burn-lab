# flowers-64-reflow-dim24-6k

The reflow pass of [`flowers-64-flow-dim24-13k`](../flowers-64-flow-dim24-13k):
the same architecture, retrained on straight-line pairs from the frozen flow
model. Its point is **one-step sampling**.

| | |
| --- | --- |
| Backbone | UNet, `dim 24`, `dim_mults (1,2,4,8)`, 5,277,891 parameters |
| Objective | rectified flow, reflow pass |
| Training | 6,000 steps, batch 16, 2,048 cached pairs at 20 ODE steps, 1,077.5 s |
| Checkpoint | `model.bpk`, 21 MB, `sha256 433e2a4676c7fc20566ba068c4464bcf15b8464e7151f399b5902325c9c27856` |

Held-out pixel-FID, 512 samples per draw:

| ODE steps | 1 | 2 | 4 | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base flow | 24.967 | 11.237 | — | 3.264 | **3.170** |
| **reflow** | **4.418** | **4.360** | **4.191** | 3.625 | 3.745 |

One step is 5.6x better than the base model's one step; twenty steps are slightly
worse, the trade the paper describes.

## Sample from it

```sh
git clone https://github.com/adeerkhan/burn-lab
cd burn-lab

cargo run -p diffusion --release -- sample --backend wgpu \
  --ckpt models/flowers-64-reflow-dim24-6k --samples 16 --sampling-steps 1
```

`--sampling-steps 1` is one Euler step from the noise: a single forward pass,
about 3 ms per image. Four steps look better; twenty do not buy much.
