# flowers-64-flow-big-dim24-22k

The rectified-flow UNet trained on the enlarged dataset: 3,303 TensorFlow
flowers plus 8,189 Oxford 102 flowers, each contributing a centre crop and two
random-resized crops with flips — **34,476 training images**. 22,000 steps,
`dim 24`, 64×64, with the same 367 held-out photographs the other models use.

| | |
| --- | --- |
| Parameters | 5,277,891 |
| Training images | 34,476 |
| Steps | 22,000 (the first 18k run plus a resumed 4k; resume restarts Adam) |
| Checkpoint | `model.bpk`, 21 MB, `sha256 78cf5d6ad9ffb43af2e88bb106a79ba9999cdf6e6a100143b265b6cc78f43779` |

Held-out scores, 512 samples per draw, 20 ODE steps:

| Metric | This model | Small-data model (3,303 images, 13k steps) |
| --- | ---: | ---: |
| pixel-FID (8×8) | 5.963 | **3.396** |
| Inception-256 FID | **106.54** | 151.04 |
| nearest-training L1 | 0.349 | 0.336 |
| gradient ratio | 0.987 | 1.066 |
| colour distance | 0.107 | 0.105 |
| real-image floor (Inception-256) | 17.79 | 17.79 |

**The two metrics disagree.** In Inception space — which sees texture and flower
parts — this is the best model in the repository, 30% better than the small-data
one. In the 8×8 pixel space it is worse, despite an identical colour histogram
distance. The coarse estimator is measuring statistics that the detailed model
no longer matches, not quality.

This checkpoint is the answer to "what does more data buy": at a fixed step
budget, better structures and worse coarse statistics. Data and steps have to
scale together; the next run is this one continued to 48k steps.

## Sample from it

```sh
git clone https://github.com/adeerkhan/burn-lab
cd burn-lab
cargo run -p diffusion --release -- sample --backend wgpu \
  --ckpt models/flowers-64-flow-big-dim24-22k --samples 16 --sampling-steps 20
```
