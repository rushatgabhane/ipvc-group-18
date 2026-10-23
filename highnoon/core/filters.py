from __future__ import annotations

import math

import numpy as np


def smoothing_factor(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._value: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._value = self._velocity = self._t = None

    def __call__(self, x: np.ndarray, t: float, mask: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._value is None:
            self._value = x.copy()
            self._velocity = np.zeros_like(x)
            self._t = t
            return self._value.copy()

        dt = t - self._t
        if dt <= 0:
            return self._value.copy()
        self._t = t

        raw_velocity = (x - self._value) / dt
        velocity = self._velocity + smoothing_factor(self.d_cutoff, dt) * (raw_velocity - self._velocity)
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True) if x.ndim > 1 else np.abs(velocity)
        cutoff = self.min_cutoff + self.beta * speed
        value = self._value + smoothing_factor(cutoff, dt) * (x - self._value)

        if mask is not None:
            keep = np.asarray(mask, dtype=bool).reshape(mask.shape + (1,) * (x.ndim - np.ndim(mask)))
            value = np.where(keep, value, self._value)
            velocity = np.where(keep, velocity, 0.0)

        self._value = value.astype(np.float32)
        self._velocity = velocity.astype(np.float32)
        return self._value.copy()
