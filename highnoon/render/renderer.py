"""T5: scene compositing and HUD.

Rendering runs on the main thread every frame, so all per-pixel work goes through OpenCV
kernels (measured at 720p: cv2.blendLinear 0.35 ms vs 12.7 ms for the equivalent numpy
float expression). Static layers such as the background are built once and cached.
"""

from __future__ import annotations

import cv2
import numpy as np

from highnoon.config import DisplayConfig
from highnoon.contracts import Perception, Player
from highnoon.core.profiler import Profiler
from highnoon.game.game import Game
from highnoon.render.game_view import draw_game, draw_signal_debug

PLAYER_COLORS = {1: (60, 60, 230), 2: (230, 160, 40)}  # BGR: red, blue

# MediaPipe pose skeleton, limited to the torso and limbs (no face/hand detail).
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]  # fmt: skip


class Renderer:
    def __init__(self, cfg: DisplayConfig):
        self.cfg = cfg
        self._background: np.ndarray | None = None

    def render(
        self, perception: Perception, players: list[Player], game: Game, profiler: Profiler
    ) -> np.ndarray:
        image = perception.frame.image
        if self.cfg.background and perception.mask is not None:
            image = self._composite(image, perception.mask)
        else:
            image = image.copy()  # never draw on the capture buffer
        for p in players:
            if p.visible:
                self._draw_player(image, p)
        draw_game(image, game, players, perception.frame.t_capture, PLAYER_COLORS)
        if self.cfg.show_debug:
            draw_signal_debug(image, game, players, PLAYER_COLORS)
            self._draw_hud(image, profiler, len(perception.poses), len(perception.faces))
        return image

    def _composite(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bg = self._get_background(image.shape)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.blendLinear(image, bg, mask, 1.0 - mask)

    def _get_background(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._background is None or self._background.shape != shape:
            self._background = _desert_background(shape[1], shape[0])
        return self._background

    def _draw_player(self, image: np.ndarray, p: Player) -> None:
        color = PLAYER_COLORS.get(p.id, (200, 200, 200))
        if p.smoothed is not None and self.cfg.show_debug:
            pts = p.smoothed.astype(np.int32)
            vis = p.pose.visibility
            for a, b in SKELETON:
                if vis[a] > 0.5 and vis[b] > 0.5:
                    cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)
        if p.face is not None:
            x0, y0, x1, y1 = p.face.bbox.astype(int)
            if self.cfg.show_debug:
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            label_anchor = (x0, max(y0 - 12, 20))
        elif p.smoothed is not None:
            nose = p.smoothed[0].astype(int)
            label_anchor = (nose[0] - 40, max(nose[1] - 80, 20))
        else:
            return
        cv2.putText(image, f"P{p.id}", label_anchor, cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

    def _draw_hud(self, image: np.ndarray, profiler: Profiler, n_poses: int, n_faces: int) -> None:
        stats = profiler.stats()
        lines = [f"FPS {profiler.fps():4.1f}   poses {n_poses}  faces {n_faces}"]
        for name in ("latency", "inference", "pose", "face", "hand", "render"):
            if name in stats:
                mean, p95 = stats[name]
                lines.append(f"{name:<10s}{mean:5.1f} ms  p95 {p95:5.1f}")
        line_h = 22
        panel_h = line_h * len(lines) + 10
        # Bottom centre: the only area not used by the banner, play area or player HUDs.
        x, y = image.shape[1] // 2 - 165, image.shape[0] - panel_h - 10
        panel = image[y : y + panel_h, x : x + 330]
        cv2.addWeighted(panel, 0.4, np.zeros_like(panel), 0.6, 0, dst=panel)  # dim the panel in place
        for i, text in enumerate(lines):
            cv2.putText(image, text, (x + 8, y + 22 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip


def _desert_background(w: int, h: int) -> np.ndarray:
    """Placeholder arena: sky gradient, sun and sand. Replace with real art later."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    sky_top = np.array([170, 110, 60], np.float32)  # BGR
    sky_bottom = np.array([120, 190, 245], np.float32)
    horizon = int(h * 0.68)
    bg = np.empty((h, w, 3), np.uint8)
    sky = sky_top + (sky_bottom - sky_top) * (t[:horizon] / t[horizon])
    bg[:horizon] = np.broadcast_to(sky[:, None, :], (horizon, w, 3)).astype(np.uint8)
    bg[horizon:] = (95, 160, 210)
    cv2.circle(bg, (int(w * 0.78), int(h * 0.25)), int(h * 0.08), (140, 230, 255), -1, cv2.LINE_AA)
    return bg
