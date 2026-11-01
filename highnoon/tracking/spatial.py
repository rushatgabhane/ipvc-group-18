"""Player positions in metres from one webcam, using the body as its own ruler.

A pinhole camera maps a length L at distance Z to f * L / Z pixels, so a body part whose real length is
roughly known gives the distance: Z = f * L / pixels. The torso (shoulder centre to hip centre) is used
because it shrinks far less than the shoulder width when a player turns sideways. Lateral and vertical
offsets then follow from the same Z. Errors come from people's torsos differing from the assumed length
(about +-10% between adults, so +-10% in distance) and from the camera's field of view, which can be set
with --hfov.
"""

from __future__ import annotations

import math

import numpy as np

TORSO_M = 0.50  # adult shoulder centre to hip centre
DEFAULT_HFOV_DEG = 70.0  # typical laptop webcam


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def locate(point_px: np.ndarray, torso_px: float, frame_size: tuple[int, int], hfov_deg: float) -> np.ndarray:
    """(x right, y down, z away) in metres for an image point on a body with that torso length."""
    width, height = frame_size
    f = focal_px(width, hfov_deg)
    z = f * TORSO_M / max(torso_px, 1.0)
    x = (float(point_px[0]) - width / 2) * z / f
    y = (float(point_px[1]) - height / 2) * z / f
    return np.array([x, y, z], np.float32)
