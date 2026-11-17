"""Run the Burn VQ-VAE over recorded frames and write a flat i64 token file.

The tokenizer is loaded from a Burn checkpoint through a tiny Rust helper
(`world-encode`), which reads frames.npy-equivalent flat u8 and writes codes.
This script only flattens the recorded .npy files to the layout the helper uses.
"""

import numpy as np
from pathlib import Path


def main() -> None:
    root = Path("/home/adeerkhan/tmp/world/data")
    frames = np.load(root / "frames.npy")
    actions = np.load(root / "actions.npy")
    flat = np.ascontiguousarray(frames).reshape(-1)
    (root / "frames.bin").write_bytes(flat.tobytes())
    (root / "actions.bin").write_bytes(np.ascontiguousarray(actions).tobytes())
    print(f"wrote frames.bin {flat.size * 1 / 1e6:.1f} MB, actions.bin {actions.size} bytes")


if __name__ == "__main__":
    main()
