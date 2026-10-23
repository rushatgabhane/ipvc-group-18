from __future__ import annotations

import cv2
import numpy as np

from highnoon.config import DisplayConfig
from highnoon.contracts import Perception, Player
from highnoon.core.profiler import Profiler
from highnoon.game.game import Game
from highnoon.render import assets
from highnoon.render.fx import FX
from highnoon.render.hud import Hud, draw_signal_debug
from highnoon.render.palette import PLAYER_COLORS, WHITE
from highnoon.render.world import WorldView

TORSO_BONES = [(11, 12), (11, 23), (12, 24), (23, 24)]
ARM_BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]
LEG_BONES = [(23, 25), (25, 27), (24, 26), (26, 28)]
SKELETON = TORSO_BONES + ARM_BONES + LEG_BONES
MASK_LOW, MASK_HIGH = 0.45, 0.9
ERODE_KERNEL = np.ones((3, 3), np.uint8)
MAX_FRAME_STEP_S = 0.1
STATS_SHOWN = ("latency", "inference", "pose", "face", "hand", "render")


class Renderer:
    def __init__(self, cfg: DisplayConfig):
        self.cfg = cfg
        self.text = assets.TextCache()
        self.fx = FX()
        self.world = WorldView(PLAYER_COLORS, self.text, self.fx)
        self.hud = Hud(PLAYER_COLORS, self.text, self.fx)
        self.shake = (0, 0)
        self._size: tuple[int, int] | None = None
        self._wall_top = 0
        self._t_last: float | None = None

    def render(
        self, perception: Perception, players: list[Player], game: Game, profiler: Profiler
    ) -> np.ndarray:
        frame = perception.frame
        height, width = frame.image.shape[:2]
        self._build_scenery(width, height, game.cfg.cover_height)
        t = frame.t_capture
        dt = 0.0 if self._t_last is None else min(t - self._t_last, MAX_FRAME_STEP_S)
        self._t_last = t

        image = self._camera_layer(frame.image, perception.mask)
        if self.cfg.show_debug:
            for player in players:
                if player.visible:
                    draw_skeleton(image, player)
        self._draw_wall(image)

        self.fx.update(t, game.effects)
        self.world.draw(image, game, t, dt)
        self.hud.draw(image, game, players, t, dt)
        if self.cfg.show_debug:
            draw_signal_debug(image, game, players, PLAYER_COLORS)
            draw_stats(image, profiler, len(perception.poses), len(perception.faces))
        self.shake = self.fx.shake_offset(t)
        return image

    def _build_scenery(self, width: int, height: int, cover_height: float) -> None:
        wall_top = int(height * (1 - cover_height))
        if self._size == (width, height) and self._wall_top == wall_top:
            return
        self._size, self._wall_top = (width, height), wall_top
        self.hud.prewarm(height)
        vignette = assets.vignette(width, height)
        self._background = cv2.multiply(assets.desert_background(width, height), vignette, scale=1 / 255)
        self._wall = cv2.multiply(
            assets.wooden_wall(width, height - wall_top), vignette[wall_top:], scale=1 / 255
        )
        shadow_h = max(8, height // 40)
        ramp = (np.linspace(1.0, 0.55, shadow_h, dtype=np.float32) ** 1.5 * 255).astype(np.uint8)
        self._shadow = cv2.merge([np.tile(ramp[:, None], (1, width))] * 3)

    def _draw_wall(self, image: np.ndarray) -> None:
        above_wall = image[self._wall_top - self._shadow.shape[0] : self._wall_top]
        cv2.multiply(above_wall, self._shadow, dst=above_wall, scale=1 / 255)
        image[self._wall_top :] = self._wall

    def _camera_layer(self, camera: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if not self.cfg.background or mask is None:
            return camera.copy()
        if mask.shape != camera.shape[:2]:
            mask = cv2.resize(mask, (camera.shape[1], camera.shape[0]), interpolation=cv2.INTER_LINEAR)
        gain = 1.0 / (MASK_HIGH - MASK_LOW)
        person = cv2.addWeighted(mask, gain, mask, 0.0, -MASK_LOW * gain)
        cv2.threshold(person, 1.0, 1.0, cv2.THRESH_TRUNC, dst=person)
        cv2.threshold(person, 0.0, 0.0, cv2.THRESH_TOZERO, dst=person)
        cv2.erode(person, ERODE_KERNEL, dst=person)
        return cv2.blendLinear(camera, self._background, person, 1.0 - person)


def draw_skeleton(image: np.ndarray, player: Player) -> None:
    color = PLAYER_COLORS.get(player.id, WHITE)
    if player.smoothed is not None:
        points = player.smoothed.astype(np.int32).tolist()
        visible = player.pose.visibility > 0.5
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(image, points[a], points[b], color, 3, cv2.LINE_AA)
    if player.face is not None:
        x0, y0, x1, y1 = player.face.bbox.astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)


def draw_stats(image: np.ndarray, profiler: Profiler, n_poses: int, n_faces: int) -> None:
    stats = profiler.stats()
    lines = [f"FPS {profiler.fps():4.1f}   poses {n_poses}  faces {n_faces}"]
    lines += [
        f"{name:<10s}{stats[name][0]:5.1f} ms  p95 {stats[name][1]:5.1f}"
        for name in STATS_SHOWN
        if name in stats
    ]
    line_h = 22
    panel_h = line_h * len(lines) + 10
    x, y = image.shape[1] // 2 - 165, image.shape[0] - panel_h - 10
    roi = image[y : y + panel_h, x : x + 330]
    cv2.convertScaleAbs(roi, dst=roi, alpha=0.4)
    for i, text in enumerate(lines):
        cv2.putText(
            image, text, (x + 8, y + 22 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA
        )
