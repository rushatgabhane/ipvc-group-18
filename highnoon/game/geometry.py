from __future__ import annotations

import numpy as np

MISS = np.inf


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segment_circle_hits(
    starts: np.ndarray, ends: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    delta = ends - starts
    offset = starts - center
    a = np.einsum("ij,ij->i", delta, delta)
    b = 2 * np.einsum("ij,ij->i", offset, delta)
    c = np.einsum("ij,ij->i", offset, offset) - radius * radius
    discriminant = b * b - 4 * a * c
    root = np.sqrt(np.maximum(discriminant, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        entry = (-b - root) / (2 * a)
    hits = (discriminant >= 0) & (entry >= 0) & (entry <= 1)
    return np.where(c <= 0, 0.0, np.where(hits, entry, MISS))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    a, b = polygon[None, :, :], np.roll(polygon, -1, axis=0)[None, :, :]
    x, y = points[:, None, 0], points[:, None, 1]
    straddles = (a[..., 1] > y) != (b[..., 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = a[..., 0] + (y - a[..., 1]) * (b[..., 0] - a[..., 0]) / (b[..., 1] - a[..., 1])
    return np.count_nonzero(straddles & (x < x_cross), axis=1) % 2 == 1


def segment_polygon_hits(starts: np.ndarray, ends: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    edge_starts = polygon[None, :, :]
    edges = (np.roll(polygon, -1, axis=0) - polygon)[None, :, :]
    deltas = (ends - starts)[:, None, :]
    offsets = edge_starts - starts[:, None, :]
    denominator = cross(deltas, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        along_segment = cross(offsets, edges) / denominator
        along_edge = cross(offsets, deltas) / denominator
    crosses = (np.abs(denominator) > 1e-9) & (along_segment >= 0) & (along_segment <= 1)
    crosses &= (along_edge >= 0) & (along_edge <= 1)
    first = np.where(crosses, along_segment, MISS).min(axis=1)
    return np.where(points_in_polygon(starts, polygon), 0.0, first)


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def ray_exit_distance(origin: np.ndarray, direction: np.ndarray, width: int, height: int) -> float:
    size = np.array([width, height], np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        limits = np.where(
            direction > 1e-9,
            (size - origin) / direction,
            np.where(direction < -1e-9, -origin / direction, MISS),
        )
    nearest = float(limits.min())
    return max(0.0, nearest) if np.isfinite(nearest) else 0.0
