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
MONO = "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf"

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



def _load(name: str) -> dict:
    import json

    return json.loads((ROOT / "data" / f"{name}.json").read_text())


def comparison() -> None:
    """Three panels, mirroring the reference's main-results chart."""
    import json

    width, height = 2400, 1120
    image, draw = canvas(width, height)
    draw.text(
        (60, 40),
        "burn-lab: the same experiment, re-run in Rust",
        font=font(40, True),
        fill=INK,
    )
    draw.text(
        (62, 96),
        "25.7M parameters · synthetic Python only · RTX 5060 via wgpu · exact unit-test verifier",
        font=font(22),
        fill=MUTED,
    )

    panel_top, panel_bottom = 260, 950
    titles = [
        "1. Pretraining learns code",
        "2. RL raises the on-policy rate",
        "3. Untouched confirmation",
    ]

    # Panel 1: dev pass@1 against pretraining step.
    left, right = 170, 760
    draw.text((left, 200), titles[0], font=font(26, True), fill=INK)
    draw.rectangle((left, panel_top, right, panel_bottom), fill=PANEL)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = panel_bottom - tick * (panel_bottom - panel_top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 76, yy - 11), f"{tick:.0%}", font=font(18), fill=MUTED)
    points = [(100, 0.0938), (200, 0.9062), (400, 1.0)]
    xy = [
        (
            left + (step - 60) / 360 * (right - left),
            panel_bottom - value * (panel_bottom - panel_top),
        )
        for step, value in points
    ]
    draw.line(xy, fill=ORANGE, width=5, joint="curve")
    for (px, py), (step, value) in zip(xy, points):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=WHITE, outline=ORANGE, width=4)
        draw.text((px - 26, py - 38), f"{value:.0%}", font=font(19, True), fill=INK)
        draw.text((px - 16, panel_bottom + 12), str(step), font=font(18), fill=MUTED)
    draw.text((left, panel_bottom + 44), "pretraining steps", font=font(18), fill=MUTED)
    draw.text((left, panel_top - 44), "dev pass@1", font=font(18, True), fill=MUTED)

    # Panel 2: exact rollout rate across RL groups, in windows of eight.
    left, right = 990, 1580
    draw.text((left, 200), titles[1], font=font(26, True), fill=INK)
    draw.rectangle((left, panel_top, right, panel_bottom), fill=PANEL)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = panel_bottom - tick * (panel_bottom - panel_top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 76, yy - 11), f"{tick:.0%}", font=font(18), fill=MUTED)
    windows = [(4, 0.672), (12, 0.859), (20, 0.828), (28, 0.734)]
    xy = [
        (left + (group - 1) / 31 * (right - left), panel_bottom - value * (panel_bottom - panel_top))
        for group, value in windows
    ]
    draw.line(xy, fill=(37, 132, 92), width=5, joint="curve")
    for (px, py), (group, value) in zip(xy, windows):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=WHITE, outline=(37, 132, 92), width=4)
        draw.text((px - 28, py - 38), f"{value:.1%}", font=font(19, True), fill=INK)
    draw.text((left, panel_bottom + 44), "RLOO prompt groups", font=font(18), fill=MUTED)
    draw.text((left, panel_top - 44), "exact rollouts, share of 8", font=font(18, True), fill=MUTED)

    # Panel 3: confirm, before against after.
    left, right = 1810, 2340
    draw.text((left, 200), titles[2], font=font(26, True), fill=INK)
    draw.rectangle((left, panel_top, right, panel_bottom), fill=PANEL)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = panel_bottom - tick * (panel_bottom - panel_top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 76, yy - 11), f"{tick:.0%}", font=font(18), fill=MUTED)

    bars = [("greedy", 0.9062, 0.8438), ("pass@8", 0.8750, 0.8438)]
    slot = (right - left) / len(bars)
    bar = 58
    for index, (label, before, after) in enumerate(bars):
        cx = left + slot * (index + 0.5)
        for offset, value, colour in (
            (-bar / 2 - 5, before, (166, 173, 184)),
            (bar / 2 + 5, after, (37, 132, 92) if after >= before else (196, 80, 60)),
        ):
            top_y = panel_bottom - value * (panel_bottom - panel_top)
            draw.rounded_rectangle(
                (cx + offset - bar / 2, top_y, cx + offset + bar / 2, panel_bottom),
                radius=8,
                fill=colour,
            )
            centred(
                draw,
                (cx + offset - bar / 2, top_y - 32, cx + offset + bar / 2, top_y - 6),
                f"{value:.1%}",
                font(17, True),
                INK,
            )
        centred(draw, (cx - slot / 2, panel_bottom + 12, cx + slot / 2, panel_bottom + 40), label, font(18), MUTED)

    draw.rounded_rectangle((2075, 150, 2095, 168), radius=3, fill=(166, 173, 184))
    draw.text((2104, 148), "before", font=font(18), fill=MUTED)
    draw.rounded_rectangle((2180, 150, 2200, 168), radius=3, fill=(196, 80, 60))
    draw.text((2209, 148), "after", font=font(18), fill=MUTED)

    draw.text(
        (60, height - 60),
        "The port reproduces the shape of the experiment. It does not reproduce the outcome: "
        "RL helps on dev and regresses on confirm.",
        font=font(21, True),
        fill=INK,
    )
    image.save(ROOT / "comparison.png")


def families_chart() -> None:
    """Per-family exact sampled completions on confirm, before and after RL."""
    before = _load("passk-confirm-pre")
    after = _load("passk-confirm-post")

    order = ["increment", "double", "even", "square", "list_sum", "absolute", "nonnegative", "reverse"]
    # RL trained on every family: `frozen_tasks("rl", 4)` was not filtered.
    by_before = {row["family"]: row for row in before["by_family"]}
    by_after = {row["family"]: row for row in after["by_family"]}

    def rate(row: dict) -> float:
        return row["exact_rollouts"] / row["rollouts"] if row["rollouts"] else 0.0

    width, height = 2200, 1120
    image, draw = canvas(width, height)
    draw.text((60, 44), "Where RL helped, and where it regressed", font=font(38, True), fill=INK)
    draw.text(
        (62, 100),
        "exact sampled completions per family on the untouched confirm split, 64 samples each",
        font=font(21),
        fill=MUTED,
    )

    left, right, top, bottom = 180, width - 120, 190, height - 150
    draw.rectangle((left, top, right, bottom), fill=PANEL)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = bottom - tick * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 78, yy - 11), f"{tick:.0%}", font=font(18), fill=MUTED)

    slot = (right - left) / len(order)
    bar = 54
    for index, family in enumerate(order):
        cx = left + slot * (index + 0.5)
        before_rate = rate(by_before[family])
        after_rate = rate(by_after[family])
        for offset, value, colour in (
            (-bar / 2 - 5, before_rate, (166, 173, 184)),
            (bar / 2 + 5, after_rate, ORANGE),
        ):
            top_y = bottom - value * (bottom - top)
            draw.rounded_rectangle(
                (cx + offset - bar / 2, top_y, cx + offset + bar / 2, bottom), radius=7, fill=colour
            )
            if value > 0.02:
                centred(
                    draw,
                    (cx + offset - bar / 2, top_y - 30, cx + offset + bar / 2, top_y - 6),
                    f"{value:.0%}",
                    font(15, True),
                    INK,
                )
            else:
                centred(
                    draw,
                    (cx + offset - bar / 2, bottom - 28, cx + offset + bar / 2, bottom - 6),
                    "0%",
                    font(15, True),
                    MUTED,
                )
        centred(
            draw,
            (cx - slot / 2, bottom + 12, cx + slot / 2, bottom + 40),
            family,
            font(19),
            INK,
        )

    draw.rounded_rectangle((width - 300, 96, width - 276, 118), radius=4, fill=(166, 173, 184))
    draw.text((width - 264, 94), "before", font=font(19), fill=MUTED)
    draw.rounded_rectangle((width - 150, 96, width - 126, 118), radius=4, fill=ORANGE)
    draw.text((width - 114, 94), "after", font=font(19), fill=MUTED)

    draw.text(
        (60, height - 60),
        "RLOO trained on all eight families. It moved two of them backwards and one forwards.",
        font=font(20),
        fill=MUTED,
    )
    image.save(ROOT / "families.png")


def generation_card() -> None:
    """The same prompts before and after RL, one gain and one loss."""
    width, height = 2200, 1180
    image, draw = canvas(width, height)
    draw.text(
        (60, 36),
        "Same prompt. Same verifier. The reward changed the output.",
        font=font(38, True),
        fill=INK,
    )
    draw.text(
        (62, 92),
        "greedy decoding, 48-token cap, both checkpoints on the same device",
        font=font(21),
        fill=MUTED,
    )

    cases = [
        (
            "dev-reverse-003 - RL fixed it",
            "# Return text in reverse order.\n"
            "def reverse_xxx(text):",
            "\n    return text[:::-1]\n",
            "\n    return text[::-1]\n",
            ("FAIL", "0/3"),
            ("PASS", "3/3"),
        ),
        (
            "confirm-reverse-002 - RL broke it",
            "# Return text in reverse order.\n"
            "def reverse_yyy(text):",
            "\n    return text[::-1]\n",
            "\n    return text[:-1]\n",
            ("PASS", "3/3"),
            ("FAIL", "0/3"),
        ),
    ]

    code_face = ImageFont.truetype(MONO, 24)
    for row, (title, prompt, before, after, before_verdict, after_verdict) in enumerate(cases):
        top = 170 + row * 480
        draw.text((70, top), title, font=font(24, True), fill=INK)

        for column, (label, code, verdict, good) in enumerate(
            (("BEFORE RL", prompt + before, before_verdict, before_verdict[0] == "PASS"),
             ("AFTER RL", prompt + after, after_verdict, after_verdict[0] == "PASS"))
        ):
            left = 70 + column * 1050
            box_top = top + 40
            fill = (232, 246, 238) if good else (251, 234, 232)
            edge = (37, 132, 92) if good else (196, 80, 60)
            draw.rounded_rectangle(
                (left, box_top, left + 990, box_top + 380), radius=14, fill=fill, outline=edge, width=3
            )
            draw.text((left + 28, box_top + 22), label, font=font(20, True), fill=edge)
            y = box_top + 78
            for line in code.split("\n"):
                draw.text((left + 28, y), line, font=code_face, fill=INK)
                y += 34
            draw.text(
                (left + 28, box_top + 320),
                f"{verdict[0]}  ·  {verdict[1]} tests",
                font=font(24, True),
                fill=edge,
            )

    draw.text(
        (60, height - 56),
        "On dev RL gained one task and lost none; on the untouched split it gained none and lost two.",
        font=font(21, True),
        fill=INK,
    )
    image.save(ROOT / "generation.png")



def capability_band() -> None:
    """Reward signal and confirm movement across the three runs."""
    width, height = 2000, 820
    image, draw = canvas(width, height)
    draw.text((60, 40), "Executable-reward RL has a capability band", font=font(38, True), fill=INK)
    draw.text(
        (62, 96),
        "below the band no rollout is valid; above it the reward has nothing to correct",
        font=font(21),
        fill=MUTED,
    )

    runs = [
        ("reference\n(their ckpt)", 73, 16),
        ("ours, their config\nfrom step 100", 4, 1),
        ("ours, their budget\nfrom step 200", 26, 0),
    ]

    # Left: groups that produced a reward signal.
    left, right, top, bottom = 150, 960, 250, 640
    draw.text((left, top - 46), "groups with reward signal (of 96)", font=font(21, True), fill=INK)
    draw.rectangle((left, top, right, bottom), fill=PANEL)
    for tick in (0, 24, 48, 72, 96):
        yy = bottom - tick / 96 * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 56, yy - 11), str(tick), font=font(17), fill=MUTED)
    slot = (right - left) / len(runs)
    for index, (label, signal, _) in enumerate(runs):
        cx = left + slot * (index + 0.5)
        height_bar = signal / 96 * (bottom - top)
        draw.rounded_rectangle(
            (cx - 52, bottom - height_bar, cx + 52, bottom), radius=9, fill=ORANGE
        )
        draw.text((cx - 26, bottom - height_bar - 34), str(signal), font=font(21, True), fill=INK)
    for index, (label, _, _) in enumerate(runs):
        cx = left + slot * (index + 0.5)
        for offset, line in enumerate(label.split("\n")):
            centred(
                draw,
                (cx - slot / 2, bottom + 14 + offset * 26, cx + slot / 2, bottom + 40 + offset * 26),
                line,
                font(17),
                MUTED,
            )

    # Right: change in confirm pass@1, in tasks.
    left, right = 1120, 1940
    draw.text((left, top - 46), "change in confirm pass@1 (tasks of 32)", font=font(21, True), fill=INK)
    draw.rectangle((left, top, right, bottom), fill=PANEL)
    zero = bottom - 0 / 20 * (bottom - top) * 0.5
    for tick in (-4, 0, 4, 8, 12, 16):
        yy = bottom - ((tick + 4) / 24) * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 46, yy - 11), f"{tick:+d}" if tick else "0", font=font(17), fill=MUTED)
    baseline = bottom - (4 / 24) * (bottom - top)
    draw.line((left, baseline, right, baseline), fill=(180, 186, 196), width=2)

    slot = (right - left) / len(runs)
    for index, (label, _, delta) in enumerate(runs):
        cx = left + slot * (index + 0.5)
        top_y = bottom - ((delta + 4) / 24) * (bottom - top)
        colour = (37, 132, 92) if delta > 0 else ((196, 80, 60) if delta < 0 else (166, 173, 184))
        draw.rounded_rectangle(
            (cx - 52, min(top_y, baseline), cx + 52, max(top_y, baseline)), radius=9, fill=colour
        )
        if delta == 0:
            draw.text((cx - 8, baseline - 34), "0", font=font(21, True), fill=INK)
        else:
            text = f"{delta:+d}"
            draw.text((cx - 20, min(top_y, baseline) - 34), text, font=font(21, True), fill=INK)

    draw.text(
        (60, height - 56),
        "Their checkpoint sits inside the band: enough valid samples to learn from, enough headroom to move.",
        font=font(21, True),
        fill=INK,
    )
    image.save(ROOT / "band.png")



def vision_chart() -> None:
    """Both image paths, before and after training."""
    vision = _load("vision")
    pilot = _load("vision-pilot")

    width, height = 2000, 780
    image, draw = canvas(width, height)
    draw.text((60, 40), "The direct-patch baseline wins at this scale", font=font(38, True), fill=INK)
    draw.text(
        (62, 96),
        "held-out seven-segment digits, 120 steps, 200 examples; chance is 10%",
        font=font(21),
        fill=MUTED,
    )

    left, right, top, bottom = 150, 1180, 210, 620
    draw.rectangle((left, top, right, bottom), fill=PANEL)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = bottom - tick * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 66, yy - 11), f"{tick:.0%}", font=font(17), fill=MUTED)

    labels = ["faithful\n(patch + 2 blocks + merge)", "direct-patch\n(baseline)"]
    values = [r["after"]["accuracy"] for r in vision["results"]]
    slot = (right - left) / len(values)
    bar = 120
    for index, (label, value) in enumerate(zip(labels, values)):
        cx = left + slot * (index + 0.5)
        top_y = bottom - value * (bottom - top)
        colour = ORANGE if value == max(values) else (166, 173, 184)
        draw.rounded_rectangle((cx - bar / 2, top_y, cx + bar / 2, bottom), radius=9, fill=colour)
        centred(draw, (cx - bar / 2, top_y - 34, cx + bar / 2, top_y - 6), f"{value:.1%}", font(21, True), INK)
        for offset, line in enumerate(label.split("\n")):
            centred(draw, (cx - slot / 2, bottom + 14 + offset * 24, cx + slot / 2, bottom + 38 + offset * 24), line, font(16), MUTED)

    # Pilot panel.
    left, right = 1330, 1940
    draw.text((left, top - 46), "25.7M language model + vision", font=font(20, True), fill=INK)
    draw.rectangle((left, top, right, bottom), fill=PANEL)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = bottom - tick * (bottom - top)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        draw.text((left - 66, yy - 11), f"{tick:.0%}", font=font(17), fill=MUTED)
    pairs = [("before", pilot["before"]["accuracy"], (166, 173, 184)), ("after 200 steps", pilot["after"]["accuracy"], ORANGE)]
    slot = (right - left) / len(pairs)
    for index, (label, value, colour) in enumerate(pairs):
        cx = left + slot * (index + 0.5)
        top_y = bottom - value * (bottom - top)
        draw.rounded_rectangle((cx - 70, top_y, cx + 70, bottom), radius=9, fill=colour)
        centred(draw, (cx - 70, top_y - 34, cx + 70, top_y - 6), f"{value:.1%}", font(21, True), INK)
        centred(draw, (cx - slot / 2, bottom + 14, cx + slot / 2, bottom + 38), label, font(17), MUTED)

    draw.text(
        (60, height - 52),
        "2.3M of 25.7M parameters trainable; the tower learns through a frozen coding model.",
        font=font(20, True),
        fill=INK,
    )
    image.save(ROOT / "vision.png")


def experiments_chart() -> None:
    """The two miniature studies, with seed dots."""
    import json

    diversity = _load("diversity")
    curriculum = _load("curriculum")

    width, height = 2200, 880
    image, draw = canvas(width, height)
    draw.text((60, 36), "Two matched miniature studies", font=font(38, True), fill=INK)
    draw.text(
        (62, 92),
        "same model, optimiser, steps and batch; only the data presentation changes. Ten paired seeds.",
        font=font(21),
        fill=MUTED,
    )

    panels = [
        (70, 1080, "Data diversity", diversity, "structures"),
        (1180, 2130, "Curriculum and ordering", curriculum, "conditions"),
    ]
    top, bottom = 200, 690

    for left, right, title, analysis, unit in panels:
        draw.text((left, top - 44), title, font=font(23, True), fill=INK)
        draw.rectangle((left, top, right, bottom), fill=PANEL)
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = bottom - tick * (bottom - top)
            draw.line((left, yy, right, yy), fill=GRID, width=1)
            draw.text((left - 62, yy - 11), f"{tick:.0%}", font=font(16), fill=MUTED)

        arms = analysis["arms"]
        slot = (right - left) / len(arms)
        for index, arm in enumerate(arms):
            cx = left + slot * (index + 0.5)
            value = arm["mean_byte_accuracy"]
            top_y = bottom - value * (bottom - top)
            colour = ORANGE if value == max(a["mean_byte_accuracy"] for a in arms) else (166, 173, 184)
            draw.rounded_rectangle((cx - 52, top_y, cx + 52, bottom), radius=8, fill=colour)
            centred(draw, (cx - 52, top_y - 30, cx + 52, top_y - 6), f"{value:.1%}", font(18, True), INK)
            if unit == "conditions":
                name = {
                    "curriculum-8-to-88": "curriculum\n8 to 88",
                    "diverse-interleaved": "interleaved",
                    "diverse-blocked": "blocked",
                }.get(arm["arm"], arm["arm"])
            else:
                name = arm["arm"].replace(" structures", "\nstructures")
            for offset, line in enumerate(name.split("\n")):
                centred(draw, (cx - slot / 2, bottom + 12 + offset * 22, cx + slot / 2, bottom + 34 + offset * 22), line, font(15), MUTED)

    draw.text(
        (60, height - 84),
        "No diversity effect survives correction (+2.5 pts at best, Holm p = 0.24); blocked ordering loses 11.9 pts (Holm p = 0.004).",
        font=font(19, True),
        fill=INK,
    )
    draw.text(
        (60, height - 54),
        "Ten seeds, seeded initialization: the earlier +3.8 pt / p = 0.008 result was an initialization artefact.",
        font=font(19),
        fill=INK,
    )
    image.save(ROOT / "experiments.png")


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    comparison()
    families_chart()
    generation_card()
    capability_band()
    vision_chart()
    experiments_chart()
    for name in ("comparison", "families", "generation", "band", "vision", "experiments"):
        print("wrote", ROOT / f"{name}.png")
