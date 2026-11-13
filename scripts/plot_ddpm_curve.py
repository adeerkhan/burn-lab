#!/usr/bin/env python3
"""Plot pixel-FID against training steps, with the estimator's floor.

    /tmp/glm-venv/bin/python scripts/plot_ddpm_curve.py \
        --stage /tmp/opencode/fid4-stage24000/summary.json ... \
        --floor /tmp/opencode/floor64.json --out /tmp/opencode/curve.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="append", default=[], type=Path)
    parser.add_argument("--floor", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=float, default=4.0)
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=PATH")
    args = parser.parse_args()

    stages = sorted(
        (json.loads(path.read_text()) for path in args.stage), key=lambda s: s["steps"]
    )
    steps = [stage["steps"] for stage in stages]
    fid = [stage["fid"] for stage in stages]
    error = [stage.get("fid_sd", 0.0) for stage in stages]

    figure, axis = plt.subplots(figsize=(7.5, 3.8))
    if args.floor:
        floor = json.loads(args.floor.read_text())
        mean = floor["floor_matched_mean"]
        spread = floor["floor_matched_sd"]
        axis.axhline(mean, color="#16a34a", linewidth=1.2, linestyle="--",
                     label=f"real-vs-real floor {mean:.2f}")
        axis.fill_between([min(steps) * 0.9, max(steps) * 1.1], mean - spread, mean + spread,
                          color="#16a34a", alpha=0.12)
    axis.axhline(args.target, color="#dc2626", linewidth=1.0, linestyle=":",
                 label=f"target {args.target:.1f}")
    axis.errorbar(steps, fid, yerr=error, marker="o", color="#2563eb", linewidth=1.8,
                  capsize=3, label="pixel-FID, 3 draws")
    for step, value in zip(steps, fid):
        axis.annotate(f"{value:.2f}", (step, value), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8)
    for item in args.compare:
        label, _, path = item.partition("=")
        point = json.loads(Path(path).read_text())
        axis.errorbar([point["steps"]], [point["fid"]], yerr=[point.get("fid_sd", 0.0)],
                      marker="s", color="#7c3aed", linestyle="none", capsize=3,
                      markersize=7, label=label)
        axis.annotate(f"{point['fid']:.2f}", (point["steps"], point["fid"]),
                      textcoords="offset points", xytext=(0, 10), ha="center",
                      fontsize=8, color="#7c3aed")
    axis.set_xlabel("training steps")
    axis.set_ylabel("pixel-FID (8x8x3)")
    axis.set_title("64x64 flowers: pixel-FID against steps")
    axis.set_xlim(min(steps) * 0.9, max(steps) * 1.1)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
