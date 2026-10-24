from __future__ import annotations

import math

import numpy as np

from highnoon.contracts import Player
from highnoon.game.game import Game
from highnoon.render import assets
from highnoon.render.assets import blit_center

NOSE = 0
VULTURE_HP_FRACTION = 0.35
VULTURE_COUNT = 3
VULTURE_WINGSPAN = 0.09
VULTURE_FADE_S = 1.0
VULTURE_ORBIT_PER_S = 0.9
VULTURE_FOLLOW_PER_S = 1.5


class Wildlife:
    def __init__(self):
        self._vulture_alpha = 0.0
        self._vulture_x: float | None = None

    def draw(self, image: np.ndarray, game: Game, players: list[Player], t: float, dt: float) -> None:
        height, width = image.shape[:2]
        weakest = self._weakest_low_player(game, players)
        fade = dt / VULTURE_FADE_S
        self._vulture_alpha = (
            min(1.0, self._vulture_alpha + fade) if weakest else max(0.0, self._vulture_alpha - fade)
        )
        if weakest is not None:
            target = float(weakest.smoothed[NOSE][0])
            follow = min(1.0, VULTURE_FOLLOW_PER_S * dt)
            self._vulture_x = (
                target if self._vulture_x is None else self._vulture_x + (target - self._vulture_x) * follow
            )
        if self._vulture_alpha <= 0 or self._vulture_x is None:
            return
        wingspan = int(VULTURE_WINGSPAN * height)
        radius_x, radius_y = 0.1 * width, 0.035 * height
        center_x = float(np.clip(self._vulture_x, radius_x + wingspan, width - radius_x - wingspan))
        center_y = 0.14 * height
        for i in range(VULTURE_COUNT):
            angle = t * VULTURE_ORBIT_PER_S + i * math.tau / VULTURE_COUNT
            flapping = (t * 0.7 + i * 0.37) % 1.0 < 0.2
            wings_up = flapping and math.sin(t * 16) > 0
            sprite = assets.vulture_sprite(wingspan, wings_up, -math.sin(angle) > 0)
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            blit_center(image, sprite, x, y, self._vulture_alpha)

    @staticmethod
    def _weakest_low_player(game: Game, players: list[Player]) -> Player | None:
        if game.mode != "duel":
            return None
        low = [
            player
            for player in players
            if player.visible
            and player.smoothed is not None
            and player.id in game.states
            and game.states[player.id].hp <= VULTURE_HP_FRACTION * game.cfg.max_hp
        ]
        return min(low, key=lambda player: game.states[player.id].hp, default=None)
