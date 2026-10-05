"""2D ray casting for shots. Rays are origin + s * direction with unit direction and s >= 0."""

from __future__ import annotations

import numpy as np


def ray_circle(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> float | None:
    """Distance along the ray to the first intersection with a circle, or None."""
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    root = np.sqrt(disc)
    for s in (-b - root, -b + root):
        if s >= 0:
            return float(s)
    return None


def ray_polygon(origin: np.ndarray, direction: np.ndarray, polygon: np.ndarray) -> float | None:
    """Distance along the ray to the first crossing of a polygon edge, or None.

    If the origin is already inside, returns 0. Works for any simple polygon (N, 2).
    """
    best = None
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        # Even-odd rule for the inside test.
        if (a[1] > origin[1]) != (b[1] > origin[1]):
            x_cross = a[0] + (origin[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if origin[0] < x_cross:
                inside = not inside
        # Solve origin + s*d = a + u*(b - a).
        edge = b - a
        denom = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denom) < 1e-9:
            continue
        diff = a - origin
        s = (diff[0] * edge[1] - diff[1] * edge[0]) / denom
        u = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
        if s >= 0 and 0 <= u <= 1 and (best is None or s < best):
            best = float(s)
    return 0.0 if inside else best


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def ray_exit_distance(origin: np.ndarray, direction: np.ndarray, width: int, height: int) -> float:
    """Distance along the ray until it leaves the image."""
    limits = []
    for axis, size in ((0, width), (1, height)):
        d = direction[axis]
        if d > 1e-9:
            limits.append((size - origin[axis]) / d)
        elif d < -1e-9:
            limits.append(-origin[axis] / d)
    return float(max(0.0, min(limits))) if limits else 0.0
