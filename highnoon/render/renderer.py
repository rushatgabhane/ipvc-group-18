"""T5: scene compositing.

Rendering runs on the main thread every frame, so all per-pixel work goes through OpenCV
kernels (measured at 720p: cv2.blendLinear 0.35 ms vs 12.7 ms for the equivalent numpy
float expression). Static layers (backdrop, wall, vignette) are generated once per resolution.

Per frame:
  1. person mask: contrast-tightened so less of the real room bleeds in as a halo
  2. blend camera and desert backdrop through the mask
  3. paste the wall over the bottom of the frame, with a soft shadow above it
  4. game layer (game_view.py)
  5. vignette, then screen shake when someone is hit
"""

from __future__ import annotations

import cv2
import numpy as np

from highnoon.config import DisplayConfig
from highnoon.contracts import Perception, Player
from highnoon.core.profiler import Profiler
from highnoon.game.game import Game
from highnoon.render import assets
from highnoon.render.game_view import GameView, draw_signal_debug

PLAYER_COLORS = {1: (70, 70, 235), 2: (235, 170, 50)}  # BGR: red, blue

# MediaPipe pose skeleton, limited to the torso and limbs (no face/hand detail).
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]  # fmt: skip

MASK_LOW, MASK_HIGH = 0.45, 0.9  # mask values remapped so [low, high] -> [0, 1]
_ERODE = np.ones((3, 3), np.uint8)


class Renderer:
    def __init__(self, cfg: DisplayConfig):
        self.cfg = cfg
        self.view = GameView(PLAYER_COLORS)
        self._size: tuple[int, int] | None = None
        self._background: np.ndarray | None = None
        self._wall: np.ndarray | None = None
        self._wall_top = 0
        self._shadow: np.ndarray | None = None
        self._vignette: np.ndarray | None = None

    def render(
        self, perception: Perception, players: list[Player], game: Game, profiler: Profiler
    ) -> np.ndarray:
        frame = perception.frame
        h, w = frame.image.shape[:2]
        self._ensure_assets(w, h, game.cfg.cover_height)
        t = frame.t_capture

        if self.cfg.background and perception.mask is not None:
            image = self._composite(frame.image, perception.mask)
        else:
            image = frame.image.copy()  # never draw on the capture buffer
        if self.cfg.show_debug:
            for p in players:
                if p.visible:
                    self._draw_skeleton(image, p)
        self._draw_wall(image)
        self.view.draw(image, game, players, t)
        cv2.multiply(image, self._vignette, dst=image, scale=1 / 255)
        if self.cfg.show_debug:
            draw_signal_debug(image, game, players, PLAYER_COLORS)
            self._draw_stats(image, profiler, len(perception.poses), len(perception.faces))

        dx, dy = self.view.shake(t)
        if dx or dy:
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            image = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return image

    # ------------------------------------------------------------------ layers
    def _ensure_assets(self, w: int, h: int, cover_height: float) -> None:
        if self._size == (w, h) and self._wall_top == int(h * (1 - cover_height)):
            return
        self._size = (w, h)
        self._background = assets.desert_background(w, h)
        self._wall_top = int(h * (1 - cover_height))
        self._wall = assets.wooden_wall(w, h - self._wall_top)
        shadow_h = max(8, h // 40)
        ramp = np.linspace(1.0, 0.55, shadow_h, dtype=np.float32) ** 1.5
        self._shadow = cv2.merge([np.tile((ramp * 255).astype(np.uint8)[:, None], (1, w))] * 3)
        self._vignette = assets.vignette(w, h)

    def _composite(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Tighten the soft edge: values below MASK_LOW become background, above MASK_HIGH person.
        scale = 1.0 / (MASK_HIGH - MASK_LOW)
        m = cv2.addWeighted(mask, scale, mask, 0.0, -MASK_LOW * scale)
        cv2.threshold(m, 1.0, 1.0, cv2.THRESH_TRUNC, dst=m)
        cv2.threshold(m, 0.0, 0.0, cv2.THRESH_TOZERO, dst=m)
        cv2.erode(m, _ERODE, dst=m)  # pull the edge in by a pixel; the outermost ring is mostly room
        return cv2.blendLinear(image, self._background, m, 1.0 - m)

    def _draw_wall(self, image: np.ndarray) -> None:
        top = self._wall_top
        sh = self._shadow.shape[0]
        roi = image[top - sh : top]
        cv2.multiply(roi, self._shadow, dst=roi, scale=1 / 255)
        image[top:] = self._wall

    def _draw_skeleton(self, image: np.ndarray, p: Player) -> None:
        color = PLAYER_COLORS.get(p.id, (200, 200, 200))
        if p.smoothed is not None:
            pts = p.smoothed.astype(np.int32)
            vis = p.pose.visibility
            for a, b in SKELETON:
                if vis[a] > 0.5 and vis[b] > 0.5:
                    cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
        if p.face is not None:
            x0, y0, x1, y1 = p.face.bbox.astype(int)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    def _draw_stats(self, image: np.ndarray, profiler: Profiler, n_poses: int, n_faces: int) -> None:
        stats = profiler.stats()
        lines = [f"FPS {profiler.fps():4.1f}   poses {n_poses}  faces {n_faces}"]
        for name in ("latency", "inference", "pose", "face", "hand", "render"):
            if name in stats:
                mean, p95 = stats[name]
                lines.append(f"{name:<10s}{mean:5.1f} ms  p95 {p95:5.1f}")
        line_h = 22
        panel_h = line_h * len(lines) + 10
        x, y = image.shape[1] // 2 - 165, image.shape[0] - panel_h - 10
        roi = image[y : y + panel_h, x : x + 330]
        cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0, dst=roi)
        for i, text in enumerate(lines):
            cv2.putText(image, text, (x + 8, y + 22 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
