"""Temporal filters for jitter reduction.

One Euro filter (Casiez, Roussel, Vogel, CHI 2012): a low-pass filter whose cutoff rises
with signal speed. When a keypoint is still, a low cutoff removes jitter. When it moves fast,
a higher cutoff removes lag. That is the smoothness/responsiveness trade-off from Task 2,
controlled by two parameters (min_cutoff, beta).

Vectorised over arrays so all 33 pose keypoints (or 478 face landmarks) are filtered in one
numpy call instead of a Python loop per point.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float, mask: np.ndarray | None = None) -> np.ndarray:
        """Filter sample x at time t (seconds).

        mask: optional boolean array (broadcastable over x's leading axis). Elements where the
        mask is False are not updated and keep their last filtered value. Use it to freeze
        keypoints with low visibility instead of letting noise drag them around.
        """
        x = np.asarray(x, dtype=np.float32)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()

        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        self._t = t

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self._dx + a_d * (dx - self._dx)

        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) if x.ndim > 1 else np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        x_hat = self._x + a * (x - self._x)

        if mask is not None:
            m = np.asarray(mask, dtype=bool).reshape(mask.shape + (1,) * (x.ndim - np.ndim(mask)))
            x_hat = np.where(m, x_hat, self._x)
            dx_hat = np.where(m, dx_hat, 0.0)

        self._x = x_hat.astype(np.float32)
        self._dx = dx_hat.astype(np.float32)
        return self._x.copy()
