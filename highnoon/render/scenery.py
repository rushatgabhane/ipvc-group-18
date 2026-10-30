from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.game.game import Game
from highnoon.render import assets

DUSK_KEYFRAMES = 5
DUSK_LEVELS = 40
DUSK_AFTER_S = 90.0
DUSK_EASE_PER_S = 0.4
WALL_DUSK_SHARE = 0.6
CHARRED = (22, 24, 28)
EMBER = (30, 90, 210)
PALE_WOOD = (140, 185, 220)
HOLE = (15, 15, 18)
GLASS = ((70, 170, 60), (200, 245, 210))
GROUND_SCORCH_REACH = 0.8
CRATER_WIDTH = 0.55
CRATER_DEPTH = 0.12


class Scenery:
    def __init__(self, seed: int = 11):
        self.size: tuple[int, int] | None = None
        self.wall_top = 0
        self.background: np.ndarray | None = None
        self.wall: np.ndarray | None = None
        self.shadow: np.ndarray | None = None
        self.dusk = 0.0
        self._skies: list[np.ndarray] = []
        self._sky_mix: np.ndarray | None = None
        self._land: np.ndarray | None = None
        self._ground: np.ndarray | None = None
        self._horizon = 0
        self._clean_wall: np.ndarray | None = None
        self._damaged_wall: np.ndarray | None = None
        self._round: int | None = None
        self._seen: set[int] = set()
        self._level = -1
        self._t_last: float | None = None
        self._rng = np.random.default_rng(seed)

    def build(self, width: int, height: int, cover_height: float) -> bool:
        wall_top = int(height * (1 - cover_height))
        if self.size == (width, height) and self.wall_top == wall_top:
            return False
        self.size, self.wall_top = (width, height), wall_top
        vignette = assets.vignette(width, height)
        raw = [
            assets.desert_background(width, height, dusk=i / (DUSK_KEYFRAMES - 1))
            for i in range(DUSK_KEYFRAMES)
        ]
        self._skies = [cv2.multiply(sky, vignette, scale=1 / 255) for sky in raw]
        sky_only = assets.sky_with_sun(width, height, assets.daylight(0.0))
        land = np.any(raw[0] != sky_only, axis=2).astype(np.float32)
        self._land = cv2.GaussianBlur(land, (5, 5), 0)
        self._horizon = int(height * assets.HORIZON)
        self._ground = np.full((height, width, 3), 255, np.uint8)
        self._sky_mix = self._skies[0].copy()
        self.background = self._skies[0].copy()
        self._clean_wall = cv2.multiply(
            assets.wooden_wall(width, height - wall_top), vignette[wall_top:], scale=1 / 255
        )
        self._damaged_wall = self._clean_wall.copy()
        self.wall = self._clean_wall.copy()
        shadow_h = max(8, height // 40)
        ramp = (np.linspace(1.0, 0.55, shadow_h, dtype=np.float32) ** 1.5 * 255).astype(np.uint8)
        self.shadow = cv2.merge([np.tile(ramp[:, None], (1, width))] * 3)
        self._level = -1
        return True

    def update(self, game: Game) -> None:
        t = game.now
        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        wall_changed = ground_changed = False
        if game.round != self._round:
            self._round = game.round
            self._damaged_wall[:] = self._clean_wall
            self._ground[:] = 255
            wall_changed = ground_changed = True
        stamped_wall, stamped_ground = self._stamp_new_damage(game)
        wall_changed |= stamped_wall
        ground_changed |= stamped_ground
        step = DUSK_EASE_PER_S * dt
        self.dusk += float(np.clip(dusk_target(game, t) - self.dusk, -step, step))
        level = round(self.dusk * DUSK_LEVELS)
        if level != self._level:
            self._level = level
            self._blend_sky(level / DUSK_LEVELS)
            wall_changed = ground_changed = True
        if wall_changed:
            self._light_wall()
        if ground_changed:
            cv2.multiply(self._sky_mix, self._ground, dst=self.background, scale=1 / 255)

    def _blend_sky(self, dusk: float) -> None:
        position = dusk * (DUSK_KEYFRAMES - 1)
        index = min(int(position), DUSK_KEYFRAMES - 2)
        mix = position - index
        cv2.addWeighted(self._skies[index], 1 - mix, self._skies[index + 1], mix, 0, dst=self._sky_mix)

    def _light_wall(self) -> None:
        land = assets.daylight(self._level / DUSK_LEVELS).land
        tint = tuple(1 - (1 - k) * WALL_DUSK_SHARE for k in land) + (1.0,)
        cv2.multiply(self._damaged_wall, tint, dst=self.wall)

    def _stamp_new_damage(self, game: Game) -> tuple[bool, bool]:
        wall = ground = False
        for effect in game.effects:
            if id(effect) in self._seen:
                continue
            if effect.kind == "blocked":
                wall |= self._bullet_hole(effect.pos)
            elif effect.kind == "explosion":
                radius = game.explosion_radius(effect.detail == "grenade")
                wall |= self._blast(effect.pos, radius)
                ground |= self._scar_ground(effect.pos, radius)
            elif effect.kind == "bottle":
                wall |= self._glass(effect.pos)
        self._seen = {id(effect) for effect in game.effects}
        return wall, ground

    def _bullet_hole(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1]) - self.wall_top
        if y < 0:
            return False
        wall, rng = self._damaged_wall, self._rng
        radius = max(3, int(self.size[1] * 0.007))
        center = (int(x), int(y))
        for angle in rng.uniform(0, math.tau, 7):
            length = radius * rng.uniform(1.6, 3.2)
            end = (int(x + math.cos(angle) * length), int(y + math.sin(angle) * length))
            cv2.line(wall, center, end, PALE_WOOD, 1, cv2.LINE_AA)
        cv2.circle(wall, center, int(radius * 1.5), (40, 60, 85), -1, cv2.LINE_AA)
        cv2.circle(wall, center, radius, HOLE, -1, cv2.LINE_AA)
        return True

    def _blast(self, pos: np.ndarray, radius: float) -> bool:
        x, y = float(pos[0]), float(pos[1]) - self.wall_top
        reach = radius * 0.7
        if y + reach < 0:
            return False
        scorch(self._damaged_wall, x, y, reach, self._rng)
        if abs(y) < radius * 0.5:
            self._notch(x, radius * 0.45, radius * 0.35 * (1 - abs(y) / (radius * 0.5)))
        return True

    def _scar_ground(self, pos: np.ndarray, radius: float) -> bool:
        x, y = float(pos[0]), float(pos[1])
        scorch(self._ground, x, y, radius * GROUND_SCORCH_REACH, self._rng, self._land)
        sand_band = self.wall_top - self._horizon
        if sand_band > 10:
            ground_y = float(np.clip(y, self._horizon + 0.4 * sand_band, self.wall_top - 0.15 * sand_band))
            self._crater(x, ground_y, radius)
        return True

    def _crater(self, x: float, y: float, radius: float) -> None:
        rng = self._rng
        marks = np.full_like(self._ground, 255)
        half_w, half_h = radius * CRATER_WIDTH, radius * CRATER_DEPTH
        center = (int(x), int(y))
        cv2.ellipse(
            marks, center, (int(half_w * 1.6), int(half_h * 2.0)), 0, 0, 360, (165,) * 3, -1, cv2.LINE_AA
        )
        cv2.ellipse(marks, center, (int(half_w), int(half_h)), 0, 0, 360, (75,) * 3, -1, cv2.LINE_AA)
        inner = (int(x), int(y + half_h * 0.2))
        cv2.ellipse(
            marks, inner, (int(half_w * 0.7), int(half_h * 0.6)), 0, 0, 360, (40,) * 3, -1, cv2.LINE_AA
        )
        for _ in range(18):
            debris = (int(x + rng.uniform(-2.0, 2.0) * half_w), int(y + rng.uniform(-2.5, 2.0) * half_h))
            cv2.circle(marks, debris, int(rng.integers(1, 4)), (90,) * 3, -1, cv2.LINE_AA)
        cv2.multiply(self._ground, marks, dst=self._ground, scale=1 / 255)

    def _notch(self, x: float, half_width: float, depth: float) -> None:
        if depth < 3:
            return
        angles = np.linspace(0, math.pi, 13)
        jitter = self._rng.uniform(0.7, 1.15, angles.size)
        points = np.stack(
            [x + np.cos(angles) * half_width * jitter, np.sin(angles) * depth * jitter], axis=1
        ).astype(np.int32)
        cv2.fillPoly(self._damaged_wall, [points], CHARRED, cv2.LINE_AA)
        cv2.polylines(self._damaged_wall, [points], False, EMBER, 2, cv2.LINE_AA)

    def _glass(self, pos: np.ndarray) -> bool:
        wall, rng = self._damaged_wall, self._rng
        rail = max(10, wall.shape[0] // 9)
        spread = self.size[1] * 0.03
        for i in range(10):
            shard = (int(pos[0] + rng.uniform(-spread, spread)), int(rng.uniform(2, rail - 2)))
            cv2.circle(wall, shard, int(rng.integers(1, 3)), GLASS[i % 2], -1, cv2.LINE_AA)
        return True


def dusk_target(game: Game, t: float) -> float:
    if game.mode != "duel":
        return 0.0
    elapsed = (game.round_over_t if game.round_over_t is not None else t) - game.round_start_t
    hurt = max((1 - state.hp / game.cfg.max_hp for state in game.states.values()), default=0.0)
    return float(np.clip(max(elapsed / DUSK_AFTER_S, hurt), 0.0, 1.0))


def scorch(
    image: np.ndarray, cx: float, cy: float, radius: float, rng, mask: np.ndarray | None = None
) -> None:
    height, width = image.shape[:2]
    x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    grain = cv2.resize(rng.uniform(0.6, 1.2, (8, 8)).astype(np.float32), (x1 - x0, y1 - y0))
    darkness = np.clip(np.clip(1 - distance, 0, 1) ** 0.7 * 0.85 * grain, 0, 0.92)
    if mask is not None:
        darkness *= mask[y0:y1, x0:x1]
    keep = 1 - darkness
    roi = image[y0:y1, x0:x1]
    roi[:] = (roi * keep[:, :, None]).astype(np.uint8)
