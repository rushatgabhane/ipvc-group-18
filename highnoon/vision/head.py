from __future__ import annotations

import math

import numpy as np

from highnoon.contracts import FaceObservation, HeadAnchor
from highnoon.core.filters import OneEuroFilter

# Whole eye outlines: averaging them is far steadier than the single corner landmarks.
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
FOREHEAD, CHIN = 10, 152
NEEDED = max(LEFT_EYE + RIGHT_EYE + [FOREHEAD, CHIN])
HAT_LIFT = 0.69  # eye centre to hat centre, in face heights: unlike eye spacing it survives turning the head
HAT_WIDTH = 1.5  # in face widths
FALLBACK_LIFT = 0.35  # bbox top to hat centre, in face heights, when the eyes are missing

# Position follows quick head moves; tilt and size change slowly, so they are smoothed harder.
CENTER_MIN_CUTOFF, CENTER_BETA = 1.0, 0.01
ANGLE_MIN_CUTOFF, ANGLE_BETA = 0.6, 0.3
WIDTH_MIN_CUTOFF, WIDTH_BETA = 0.4, 0.002
# A face missed for a frame or two keeps its filters; longer than this and the next one starts afresh.
RESET_AFTER_S = 0.5


def measure_head(face: FaceObservation) -> HeadAnchor | None:
    """The hat anchor for one raw detection, before any smoothing."""
    x1, y1, x2, y2 = np.asarray(face.bbox, np.float32).reshape(-1)[:4]
    face_width, face_height = float(x2 - x1), float(y2 - y1)
    if face_width < 4 or face_height < 4:
        return None
    width = face_width * HAT_WIDTH
    points = np.asarray(face.landmarks, np.float32)
    if len(points) <= NEEDED:
        return HeadAnchor(np.array([(x1 + x2) / 2, y1 - face_height * FALLBACK_LIFT], np.float32), 0.0, width)

    left_eye = points[LEFT_EYE, :2].mean(axis=0)
    right_eye = points[RIGHT_EYE, :2].mean(axis=0)
    angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    lift = HAT_LIFT * float(np.linalg.norm(points[CHIN, :2] - points[FOREHEAD, :2]))
    eye_center = (left_eye + right_eye) / 2
    center = eye_center + lift * np.array([math.sin(angle), -math.cos(angle)], np.float32)
    return HeadAnchor(center.astype(np.float32), angle, width)


class HeadFilter:
    """Smooths the few numbers that place the hat, rather than every face landmark."""

    def __init__(self):
        self.center = OneEuroFilter(CENTER_MIN_CUTOFF, CENTER_BETA)
        self.angle = OneEuroFilter(ANGLE_MIN_CUTOFF, ANGLE_BETA)
        self.width = OneEuroFilter(WIDTH_MIN_CUTOFF, WIDTH_BETA)
        self._angle_unwrapped: float | None = None
        self._t: float | None = None

    def __call__(self, raw: HeadAnchor, t: float) -> HeadAnchor:
        if self._t is not None and t - self._t > RESET_AFTER_S:
            self.reset()
        self._t = t
        # Keep the angle continuous so a tilt across +-180 degrees does not spin the hat the long way round.
        angle = raw.angle
        if self._angle_unwrapped is not None:
            angle = self._angle_unwrapped + math.remainder(angle - self._angle_unwrapped, math.tau)
        self._angle_unwrapped = angle
        return HeadAnchor(
            self.center(raw.center, t),
            float(self.angle(np.float32(angle), t)),
            float(self.width(np.float32(raw.width), t)),
        )

    def reset(self) -> None:
        self.center.reset()
        self.angle.reset()
        self.width.reset()
        self._angle_unwrapped = self._t = None
