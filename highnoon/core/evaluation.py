"""Metrics for the report, computed from per-frame series recorded while replaying a clip.

Kept free of video and model code so they can be unit-tested on synthetic series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# White noise of std s has second differences of std s * sqrt(6), so this turns them back into s.
SECOND_DIFF_TO_NOISE = 1.0 / np.sqrt(6.0)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    mask = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def noise_level(values: np.ndarray, valid: np.ndarray, min_run: int = 5) -> float:
    """Frame-to-frame jitter (std of the noise) of a signal, ignoring smooth motion.

    Uses second differences inside continuous runs: real movement is smooth over three frames, so it
    mostly cancels, while independent per-frame noise does not.
    """
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    diffs = [np.diff(values[a:b], n=2, axis=0) for a, b in runs(valid) if b - a >= min_run]
    if not diffs:
        return float("nan")
    second = np.concatenate(diffs)
    # Median absolute deviation, so a few real jerks (a duck, a sudden turn) do not dominate.
    mad = np.median(np.abs(second - np.median(second, axis=0)), axis=0) * 1.4826
    return float(np.mean(mad) * SECOND_DIFF_TO_NOISE)


def filter_lag_s(
    raw: np.ndarray, smooth: np.ndarray, valid: np.ndarray, dt: float, max_lag: int = 15
) -> float:
    """How far the smoothed signal trails the raw one: the shift that best lines up their velocities."""
    raw, smooth = np.asarray(raw, np.float64), np.asarray(smooth, np.float64)
    best_lags, weights = [], []
    for a, b in runs(valid):
        if b - a < 4 * max_lag:
            continue
        v_raw, v_smooth = np.diff(raw[a:b]), np.diff(smooth[a:b])
        if np.std(v_raw) < 1e-6:
            continue
        scores = [
            np.dot(v_raw[: len(v_raw) - k], v_smooth[k:]) / (len(v_raw) - k) for k in range(max_lag + 1)
        ]
        best_lags.append(int(np.argmax(scores)))
        weights.append(b - a)
    if not best_lags:
        return float("nan")
    return float(np.average(best_lags, weights=weights) * dt)


@dataclass
class Gap:
    player: int
    start_t: float
    duration_s: float
    recovered: bool
    # Time a body without an id was in view just before the id came back: how long re-identifying took.
    reid_wait_s: float


def gaps(t: np.ndarray, visible: np.ndarray, waiting: np.ndarray, player: int) -> list[Gap]:
    """Every stretch in which an id that had been seen was missing."""
    t, visible, waiting = np.asarray(t), np.asarray(visible, bool), np.asarray(waiting, bool)
    seen = np.flatnonzero(visible)
    if not len(seen):
        return []
    found = []
    for a, b in runs(~visible):
        if a <= seen[0]:
            continue
        recovered = b < len(visible)
        end_t = t[b] if recovered else t[-1]
        wait = 0.0
        if recovered:
            k = b - 1
            while k >= a and waiting[k]:
                k -= 1
            wait = float(t[b] - t[k + 1]) if k + 1 < b else 0.0
        found.append(Gap(player, float(t[a]), float(end_t - t[a]), recovered, wait))
    return found


def position_jumps(t: np.ndarray, x: np.ndarray, visible: np.ndarray, threshold: float) -> list[float]:
    """Times at which an id moved further in one frame than a person can: a likely identity swap."""
    x, visible = np.asarray(x, np.float64), np.asarray(visible, bool)
    both = visible[1:] & visible[:-1]
    jumped = both & (np.abs(np.diff(x)) > threshold)
    return [float(t[i + 1]) for i in np.flatnonzero(jumped)]


def reference_identity(
    embeddings: np.ndarray, references: dict[int, np.ndarray], top_k: int = 3
) -> np.ndarray:
    """For each face embedding, which reference person it resembles most (by mean top-k cosine)."""
    ids = sorted(references)
    scores = np.stack(
        [
            np.sort(embeddings @ references[i].T, axis=1)[:, -min(top_k, len(references[i])) :].mean(axis=1)
            for i in ids
        ],
        axis=1,
    )
    return np.array(ids)[np.argmax(scores, axis=1)], scores


@dataclass
class Disagreement:
    player: int
    start_t: float
    duration_s: float


def disagreements(
    t: np.ndarray, agree: np.ndarray, checked: np.ndarray, player: int, min_s: float
) -> list[Disagreement]:
    """Stretches where the face on an id belonged to someone else for at least `min_s`: identity switches."""
    t, agree, checked = np.asarray(t), np.asarray(agree, bool), np.asarray(checked, bool)
    idx = np.flatnonzero(checked)
    if not len(idx):
        return []
    found = []
    for a, b in runs(~agree[idx]):
        start, end = t[idx[a]], t[idx[b - 1]]
        if end - start >= min_s:
            found.append(Disagreement(player, float(start), float(end - start)))
    return found


def summary_stats(values) -> dict[str, float]:
    values = np.asarray([v for v in values if np.isfinite(v)], np.float64)
    if not len(values):
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }
