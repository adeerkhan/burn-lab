#!/usr/bin/env python3
"""The same forward-pass benchmark as `bench_forward.rs`, on the Python reference.

Measures the reference implementation shipped in
`vukrosic/glm-5.3-flash-from-scratch`, so the two numbers describe the same work
at the same shapes. Random initialisation is fine: latency does not depend on the
weight values.

    <python-with-torch> bench/pytorch_forward.py cpu 16 128 20
    <python-with-torch> bench/pytorch_forward.py cuda 1 192 20
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# `--reference <dir>` points at a clone of the reference repository; without it
# the script looks for a sibling `glm-ref` checkout.
reference = None
for index, argument in enumerate(sys.argv):
    if argument == "--reference" and index + 1 < len(sys.argv):
        reference = Path(sys.argv[index + 1])
if reference is None:
    reference = Path(__file__).resolve().parents[1] / "glm-ref"
assert (reference / "glm53_flash" / "model.py").exists(), (
    f"reference not found at {reference}; clone it or pass --reference <dir>"
)
sys.path.insert(0, str(reference))

from glm53_flash.model import GLM53FlashFromScratch, ModelConfig  # noqa: E402

positional = [value for value in sys.argv[1:] if not value.startswith("--")]
# Drop the value that follows --reference.
if "--reference" in sys.argv:
    at = sys.argv.index("--reference")
    if len(sys.argv) > at + 1:
        positional = [value for value in positional if value != sys.argv[at + 1]]

device_name = positional[0] if positional else "cpu"
batch = int(positional[1]) if len(positional) > 1 else 16
sequence = int(positional[2]) if len(positional) > 2 else 128
iterations = int(positional[3]) if len(positional) > 3 else 20

device = torch.device(device_name)
config = ModelConfig()
assert sequence <= config.max_sequence_length

build_started = time.perf_counter()
model = GLM53FlashFromScratch(config).to(device).eval()
if device.type == "cuda":
    torch.cuda.synchronize()
build_seconds = time.perf_counter() - build_started

tokens = torch.arange(batch * sequence, device=device, dtype=torch.long) % 60 + 10
tokens = tokens.view(batch, sequence)

def synchronize() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

with torch.no_grad():
    for _ in range(3):
        model(tokens)
    synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        model(tokens)
    synchronize()
    seconds = time.perf_counter() - started

per_forward = seconds / iterations
tokens_per_second = (batch * sequence) / per_forward

print(f"backend      pytorch {device.type}")
print(f"torch        {torch.__version__}")
print(f"shape        batch {batch} x seq {sequence}")
print(f"build        {build_seconds * 1e3:.1f} ms (device + model init)")
print(f"forward      {per_forward * 1e3:.1f} ms")
print(f"throughput   {tokens_per_second:.0f} tokens/s")
