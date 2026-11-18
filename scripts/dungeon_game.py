#!/usr/bin/env python3
"""A playable state-first dungeon game.

The avatar and enemies are driven by learned transition models; the cave is
rendered by the known environment. Levels, score, health, a minimap, particles,
a flickering torch and simple sound effects sit on top.

    python3 scripts/dungeon_game.py --record --steps 720 --out /tmp/game
    python3 scripts/dungeon_game.py --play            # needs pygame-ce

In record mode a model-predictive planner plays, so the demo is reproducible
without a display.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dungeon_env import Dungeon, DELTA, GRID  # noqa: E402
from state_model import AvatarMoveNet, EnemyNet, avatar_features, enemy_features  # noqa: E402

SCALE = 6
FRAME = GRID * SCALE
TOP = 42
BOTTOM = 26


class Game:
    def __init__(self, avatar, enemy, seed: int = 0):
        self.avatar = avatar
        self.enemy = enemy
        self.rng = np.random.default_rng(seed)
        self.level = 1
        self.score = 0
        self.health = 3
        self.time = 0
        self.particles: list[list] = []
        self.events: list[tuple[int, str]] = []
        self.new_level(seed)

    def new_level(self, seed: int) -> None:
        dungeon = Dungeon(seed, 2, 5)
        self.grid = dungeon.grid
        self.start = dungeon.start
        self.player = dungeon.start
        self.enemies = list(dungeon.enemies)
        self.diamonds = list(dungeon.diamonds)
        self.exit = dungeon.goal
        self.unlocked = False
        self.dungeon = dungeon
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        self.px = xx.astype(np.float32)
        self.py = yy.astype(np.float32)

    def _free(self, cell) -> bool:
        x, y = cell
        return 0 <= x < GRID and 0 <= y < GRID and self.grid[y, x] == 0

    def move_player(self, action: int) -> None:
        with torch.no_grad():
            logits = self.avatar(torch.from_numpy(avatar_features(self.player, self.grid, action)).unsqueeze(0))
        move = int(logits.argmax(1).item())
        dx, dy = DELTA[move]
        candidate = (self.player[0] + dx, self.player[1] + dy)
        if self._free(candidate):
            self.player = candidate

    def move_enemies(self, action: int) -> None:
        for index, (x, y) in enumerate(self.enemies):
            with torch.no_grad():
                logits = self.enemy(torch.from_numpy(enemy_features((x, y), self.player, self.grid, action)).unsqueeze(0))
                move = int(torch.distributions.Categorical(logits=logits).sample().item())
            dx, dy = DELTA[move]
            target = (x + dx, y + dy)
            if self._free(target):
                self.enemies[index] = target

    def step(self, action: int) -> None:
        self.time += 1
        self.move_player(action)
        if self.player in self.diamonds:
            self.diamonds.remove(self.player)
            self.score += 10
            self.events.append((self.time, "pickup"))
            for _ in range(7):
                self.particles.append([self.player[0], self.player[1],
                                       (70, 220, 255), 12])
            if not self.diamonds:
                self.unlocked = True
                self.events.append((self.time, "unlock"))
        self.move_enemies(action)
        if self.player in self.enemies:
            self.health -= 1
            self.events.append((self.time, "hit"))
            self.player = self.start
            for _ in range(9):
                self.particles.append([self.player[0], self.player[1], (255, 70, 82), 10])
            if self.health <= 0:
                self.events.append((self.time, "gameover"))
                self.health = 3
                self.score = 0
                self.level = 1
                self.new_level(int(self.rng.integers(1 << 20)))
                return
        if self.unlocked and self.player == self.exit:
            self.score += 50
            self.level += 1
            self.health = min(3, self.health + 1)
            self.events.append((self.time, "level"))
            self.new_level(int(self.rng.integers(1 << 20)))
            return
        for particle in self.particles:
            particle[3] -= 1
        self.particles = [p for p in self.particles if p[3] > 0]

    def render(self) -> np.ndarray:
        self.dungeon.player = self.player
        self.dungeon.enemies = list(self.enemies)
        self.dungeon.diamonds = list(self.diamonds)
        self.dungeon.goal = self.exit
        base = self.dungeon.render().astype(np.float32)
        base *= 1.0 + 0.05 * math.sin(self.time * 1.7)
        image = np.clip(base, 0, 255).astype(np.uint8)
        from PIL import Image, ImageDraw
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for x, y, colour, life in self.particles:
            r = max(1, life // 6)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        for gem in self.diamonds:
            draw.polygon([(gem[0], gem[1] - 3), (gem[0] + 3, gem[1]),
                          (gem[0], gem[1] + 3), (gem[0] - 3, gem[1])],
                         fill=(70, 220, 255), outline=(235, 255, 255))
        for enemy in self.enemies:
            draw.ellipse([enemy[0] - 3, enemy[1] - 3, enemy[0] + 3, enemy[1] + 3],
                         fill=(250, 70, 82), outline=(80, 8, 16))
            draw.ellipse([enemy[0] - 1, enemy[1] - 1, enemy[0] + 1, enemy[1] + 1],
                         fill=(255, 220, 220))
        colour = (60, 250, 140) if self.unlocked else (70, 120, 90)
        draw.rectangle([self.exit[0] - 3, self.exit[1] - 3, self.exit[0] + 3, self.exit[1] + 3],
                       outline=colour, width=2)
        draw.ellipse([self.player[0] - 4, self.player[1] - 4, self.player[0] + 4, self.player[1] + 4],
                     fill=(255, 214, 92), outline=(255, 250, 220))
        return np.asarray(canvas)

    def hud(self, frame: np.ndarray) -> "Image.Image":
        from PIL import Image, ImageDraw, ImageFont
        image = Image.fromarray(frame).resize((FRAME, FRAME), Image.BICUBIC)
        canvas = Image.new("RGB", (FRAME, FRAME + TOP + BOTTOM), (12, 13, 17))
        canvas.paste(image, (0, TOP))
        draw = ImageDraw.Draw(canvas)
        def load(size: int):
            for path in (
                "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
                "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ):
                if Path(path).exists():
                    return ImageFont.truetype(path, size)
            return ImageFont.load_default()

        font = load(18)
        small = load(11)
        draw.text((10, 9), f"SCORE {self.score:04d}", fill=(242, 184, 75), font=font)
        draw.text((190, 9), f"LEVEL {self.level}", fill=(200, 205, 215), font=font)
        draw.text((10, FRAME + TOP + 4), "arrows move   space new level   q quit", fill=(140, 146, 158), font=small)
        for i in range(3):
            colour = (230, 60, 70) if i < self.health else (55, 58, 66)
            draw.ellipse([300 + i * 24, 11, 316 + i * 24, 27], fill=colour)
        minimap = Image.new("RGB", (40, 40), (8, 9, 12))
        md = ImageDraw.Draw(minimap)
        for y in range(0, GRID, 2):
            for x in range(0, GRID, 2):
                if self.grid[y, x]:
                    md.point((x * 40 // GRID, y * 40 // GRID), fill=(90, 82, 70))
        for gem in self.diamonds:
            md.point((gem[0] * 40 // GRID, gem[1] * 40 // GRID), fill=(70, 220, 255))
        for enemy in self.enemies:
            md.point((enemy[0] * 40 // GRID, enemy[1] * 40 // GRID), fill=(250, 70, 82))
        md.point((self.exit[0] * 40 // GRID, self.exit[1] * 40 // GRID),
                 fill=(60, 250, 140) if self.unlocked else (70, 120, 90))
        md.point((self.player[0] * 40 // GRID, self.player[1] * 40 // GRID), fill=(255, 214, 92))
        canvas.paste(minimap, (FRAME - 50, 2))
        return canvas


class Planner:
    """Choose an action by scoring the learned next cell with BFS distance."""

    def __init__(self, game: Game):
        self.game = game
        self.last = 0

    def _distances(self, targets):
        from collections import deque
        dist = {target: 0 for target in targets}
        queue = deque(targets)
        while queue:
            x, y = queue.popleft()
            for dx, dy in DELTA[1:]:
                cell = (x + dx, y + dy)
                if self.game._free(cell) and cell not in dist:
                    dist[cell] = dist[(x, y)] + 1
                    queue.append(cell)
        return dist

    def choose(self) -> int:
        game = self.game
        targets = game.diamonds if game.diamonds else [game.exit]
        dist = self._distances(targets)
        reverse = {1: 2, 2: 1, 3: 4, 4: 3}
        best_value, best_action = -1e9, 0
        for action in range(5):
            with torch.no_grad():
                logits = game.avatar(torch.from_numpy(avatar_features(game.player, game.grid, action)).unsqueeze(0))
            move = int(logits.argmax(1).item())
            dx, dy = DELTA[move]
            candidate = (game.player[0] + dx, game.player[1] + dy)
            if not game._free(candidate):
                candidate = game.player
            value = -dist.get(candidate, 4096)
            for enemy in game.enemies:
                gap = abs(candidate[0] - enemy[0]) + abs(candidate[1] - enemy[1])
                value -= 30.0 * math.exp(-gap / 5.0)
            if candidate in game.diamonds:
                value += 12.0
            if action == 0:
                value -= 1.5
            if action == reverse.get(self.last):
                value -= 2.5
            if value > best_value:
                best_value, best_action = value, action
        self.last = best_action
        return best_action


def load_models(out: Path):
    avatar = AvatarMoveNet()
    enemy = EnemyNet()
    avatar.load_state_dict(torch.load(out / "avatar-move.pt"))
    enemy.load_state_dict(torch.load(out / "enemy-model.pt"))
    avatar.eval()
    enemy.eval()
    return avatar, enemy


def record(args) -> None:
    avatar, enemy = load_models(args.model)
    game = Game(avatar, enemy, args.seed)
    planner = Planner(game)
    args.out.mkdir(parents=True, exist_ok=True)
    actions = []
    for step in range(args.steps):
        action = planner.choose()
        actions.append(action)
        game.step(action)
        game.hud(game.render()).save(args.out / f"step-{step:04d}.png")
        if (step + 1) % 100 == 0:
            print(f"step {step + 1} level {game.level} score {game.score} health {game.health}", flush=True)
    np.array([0, 0, 0, 0] + actions, dtype=np.uint8).tofile(args.out / "actions.bin")
    (args.out / "events.json").write_text(json.dumps(game.events))
    print(f"final level {game.level} score {game.score}")


def play(args) -> None:
    import pygame
    avatar, enemy = load_models(args.model)
    game = Game(avatar, enemy, args.seed)
    pygame.init()
    pygame.mixer.init()
    sounds = {}
    for name in ("pickup", "hit", "level"):
        path = args.sounds / f"{name}.wav"
        if path.exists():
            sounds[name] = pygame.mixer.Sound(str(path))
    screen = pygame.display.set_mode((FRAME, FRAME + TOP + BOTTOM))
    pygame.display.set_caption("Dungeon: learned avatar, exact renderer")
    clock = pygame.time.Clock()
    keymap = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap:
                before = len(game.events)
                game.step(keymap[event.key])
                for _, name in game.events[before:]:
                    if name in sounds:
                        sounds[name].play()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game.new_level(int(game.rng.integers(1 << 20)))
        screen.blit(pygame.image.frombuffer(np.asarray(game.hud(game.render())).tobytes(), (FRAME, FRAME + TOP + BOTTOM), "RGB"), (0, 0))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/state"))
    ap.add_argument("--sounds", type=Path, default=Path("/home/adeerkhan/github/burn/burn-lab/docs/audio"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/game"))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()
    if args.record:
        record(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
