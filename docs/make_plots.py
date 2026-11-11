#!/usr/bin/env python3
"""Render the README figures.

Every number here is measured output from this repository's runs; see
`glm53/README.md` for the tables they come from. Run from the repo root:

    python3 docs/make_plots.py && magick docs/*.png docs/*.webp
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent
FONT = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"

INK = (22, 24, 29)
MUTED = (95, 102, 114)
GRID = (233, 236, 241)
PANEL = (247, 248, 250)
ORANGE = (241, 90, 36)
ORANGE_SOFT = (253, 226, 214)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    return image, ImageDraw.Draw(image)


def centred(draw: ImageDraw.ImageDraw, box, text: str, face, fill=INK) -> None:
    left, top, right, bottom = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=face)
    draw.text(
        (left + (right - left - (x1 - x0)) / 2 - x0, top + (bottom - top - (y1 - y0)) / 2 - y0),
        text,
        font=face,
        fill=fill,
    )


def arrow(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    draw.line((x0, y0, x1, y1), fill=(180, 186, 196), width=3)
    draw.polygon([(x1, y1), (x1 - 9, y1 - 5), (x1 - 9, y1 + 5)], fill=(180, 186, 196))


def hero() -> None:
    width, height = 1600, 700
    image, draw = canvas(width, height)

    draw.text((80, 44), "burn-lab", font=font(28, True), fill=ORANGE)
    draw.text((80, 84), "GLM-5.3-Flash from scratch, in Burn", font=font(44, True), fill=INK)
    draw.text(
        (82, 146),
        "25,730,592 parameters  ·  9,805,344 active per token  ·  12 hybrid blocks  ·  byte-level",
        font=font(21),
        fill=MUTED,
    )

    # One compact row of stages.
    stages = [
        ("bytes", "260 vocab", 140),
        ("Embedding", "260 x 192", 176),
        ("4 streams", "learned mix", 176),
        ("12 hybrid blocks", "9 linear · 3 sparse", 380),
        ("RMSNorm", "pre-head", 150),
        ("tied head", "shares embedding", 168),
    ]
    total = sum(width for _, _, width in stages) + 40 * (len(stages) - 1)
    x = (width - total) // 2
    y, h = 244, 112
    for index, (title, sub, box_width) in enumerate(stages):
        highlight = index == 3
        draw.rounded_rectangle(
            (x, y, x + box_width, y + h),
            radius=16,
            fill=ORANGE_SOFT if highlight else PANEL,
            outline=ORANGE if highlight else GRID,
            width=2,
        )
        centred(draw, (x, y + 30, x + box_width, y + 30 + 26), title, font(23, True))
        centred(draw, (x, y + 62, x + box_width, y + 62 + 22), sub, font(17), MUTED)
        if index < len(stages) - 1:
            arrow(draw, x + box_width + 6, y + h // 2, x + box_width + 34, y + h // 2)
        x += box_width + 40

    # The layer rhythm, as its own band so it cannot overflow the row above.
    draw.text(
        (80, 400),
        "layer rhythm",
        font=font(19, True),
        fill=MUTED,
    )
    draw.text(
        (80 + 150, 402),
        "linear, linear, linear, sparse — repeated three times",
        font=font(18),
        fill=MUTED,
    )

    cell, gap, group_gap = 74, 8, 44
    strip = 12 * cell + 9 * gap + 2 * group_gap
    sx = (width - strip) // 2
    sy, sh = 434, 34
    for index in range(12):
        group = index // 4
        within = index % 4
        left = sx + index * cell + index * gap + group * group_gap
        sparse = within == 3
        draw.rounded_rectangle(
            (left, sy, left + cell, sy + sh),
            radius=7,
            fill=ORANGE if sparse else WHITE,
            outline=ORANGE if sparse else GRID,
            width=2,
        )
        centred(
            draw,
            (left, sy, left + cell, sy + sh),
            "sparse" if sparse else "linear",
            font(14, True),
            WHITE if sparse else MUTED,
        )

    draw.line((80, 520, width - 80, 520), fill=GRID, width=2)

    facts = [
        ("8 experts", "top-2 + 1 shared", 80, ORANGE),
        ("4 residual streams", "learned routing", 450, ORANGE),
        ("192 context", "byte tokenizer", 880, ORANGE),
        ("AdamW + RLOO", "executable reward", 1220, ORANGE),
    ]
    for title, sub, fx, colour in facts:
        draw.text((fx, 566), title, font=font(21, True), fill=colour)
        draw.text((fx, 598), sub, font=font(17), fill=MUTED)

    image.save(ROOT / "hero.png")


def loss_curve() -> None:
    width, height = 1280, 660
    image, draw = canvas(width, height)
    draw.text((70, 46), "Pretraining", font=font(30, True), fill=INK)
    draw.text(
        (70, 88),
        "400 steps, batch 16 x 128, AdamW, lr 3e-4, 155 s on an RTX 5060 (wgpu)",
        font=font(20),
        fill=MUTED,
    )
    draw.text(
        (70, 120),
        "ln(260) = 5.56 at initialisation — the first point is a uniform guess",
        font=font(17),
        fill=MUTED,
    )

    steps = [1, 100, 200, 300, 400]
    losses = [5.0891, 0.7464, 0.2935, 0.3228, 0.2113]

    left, right, top, bottom = 120, width - 70, 182, height - 90
    draw.rectangle((left, top, right, bottom), fill=PANEL)

    # Log-scaled y axis from 0.2 to 6.
    import math

    lo, hi = math.log(0.18), math.log(6.0)

    def y_of(value: float) -> float:
        return bottom - (math.log(value) - lo) / (hi - lo) * (bottom - top)

    def x_of(step: int) -> float:
        return left + step / 400 * (right - left)

    for tick in [0.2, 0.5, 1.0, 2.0, 5.0]:
        yy = y_of(tick)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 62, yy - 11), f"{tick:g}", font=font(17), fill=MUTED)

    for tick in [0, 100, 200, 300, 400]:
        xx = x_of(tick)
        draw.text((xx - 16, bottom + 14), str(tick), font=font(17), fill=MUTED)

    draw.text((left, bottom + 44), "step", font=font(17), fill=MUTED)
    draw.text((left - 78, top - 30), "loss", font=font(17), fill=MUTED)

    points = [(x_of(s), y_of(l)) for s, l in zip(steps, losses)]
    draw.line(points, fill=ORANGE, width=5, joint="curve")
    for (px, py), loss in zip(points, losses):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=WHITE, outline=ORANGE, width=4)
        draw.text((px + 14, py - 30), f"{loss:.3f}", font=font(17), fill=INK)

    image.save(ROOT / "loss.png")


def checkpoint_bars() -> None:
    width, height = 1180, 600
    image, draw = canvas(width, height)
    draw.text((70, 44), "Checkpoint selection on dev", font=font(30, True), fill=INK)
    draw.text(
        (70, 86),
        "greedy pass@1, 32 tasks. Dev saturates at step 400, as the reference reports for its own benchmark.",
        font=font(19),
        fill=MUTED,
    )

    bars = [(100, 0.0938, "3 / 32"), (200, 0.9062, "29 / 32"), (400, 1.0, "32 / 32")]
    left, right, top, bottom = 150, width - 90, 170, height - 110
    base = bottom
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = base - tick * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 62, yy - 11), f"{tick:.2f}", font=font(17), fill=MUTED)

    slot = (right - left) / len(bars)
    width_bar = 150
    for index, (step, value, label) in enumerate(bars):
        cx = left + slot * (index + 0.5)
        top_y = base - value * (bottom - top)
        colour = ORANGE if value > 0.5 else (240, 160, 120)
        draw.rounded_rectangle(
            (cx - width_bar / 2, top_y, cx + width_bar / 2, base), radius=10, fill=colour
        )
        draw.text((cx - 30, top_y - 34), label, font=font(20, True), fill=INK)
        centred(
            draw,
            (cx - slot / 2, base + 12, cx + slot / 2, base + 40),
            f"checkpoint {step}",
            font(19),
            MUTED,
        )
    image.save(ROOT / "checkpoints.png")


def rl_comparison() -> None:
    width, height = 1280, 640
    image, draw = canvas(width, height)
    draw.text((70, 44), "RLOO before and after", font=font(30, True), fill=INK)
    draw.text(
        (70, 86),
        "dev is the selection split; confirm was opened once, for the final report only.",
        font=font(19),
        fill=MUTED,
    )

    groups = [
        ("dev pass@1", 0.9062, 0.9375),
        ("dev pass@8", 0.8750, 0.8750),
        ("confirm pass@1", 0.9062, 0.8438),
        ("confirm pass@8", 0.8750, 0.8125),
    ]

    left, right, top, bottom = 150, width - 90, 180, height - 120
    base = bottom
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = base - tick * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 62, yy - 11), f"{tick:.2f}", font=font(17), fill=MUTED)

    slot = (right - left) / len(groups)
    bar = 66
    for index, (label, before, after) in enumerate(groups):
        cx = left + slot * (index + 0.5)
        for offset, value, colour in ((-bar / 2 - 6, before, (205, 210, 218)), (bar / 2 + 6, after, ORANGE)):
            top_y = base - value * (bottom - top)
            draw.rounded_rectangle(
                (cx + offset - bar / 2, top_y, cx + offset + bar / 2, base), radius=8, fill=colour
            )
            centred(
                draw,
                (cx + offset - bar / 2, top_y - 30, cx + offset + bar / 2, top_y - 6),
                f"{value:.3f}",
                font(16, True),
                INK,
            )
        centred(draw, (cx - slot / 2, base + 14, cx + slot / 2, base + 42), label, font(18), MUTED)

    # Legend.
    lx, ly = width - 360, 96
    draw.rounded_rectangle((lx, ly - 6, lx + 26, ly + 14), radius=5, fill=(205, 210, 218))
    draw.text((lx + 38, ly - 8), "before", font=font(18), fill=MUTED)
    draw.rounded_rectangle((lx + 140, ly - 6, lx + 166, ly + 14), radius=5, fill=ORANGE)
    draw.text((lx + 178, ly - 8), "after", font=font(18), fill=MUTED)

    draw.text(
        (70, height - 46),
        "One task better on dev, two worse on confirm, at both pass@1 and pass@8.",
        font=font(19, True),
        fill=INK,
    )
    image.save(ROOT / "rl.png")


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    hero()
    loss_curve()
    checkpoint_bars()
    rl_comparison()
    for name in ("hero", "loss", "checkpoints", "rl"):
        print("wrote", ROOT / f"{name}.png")
