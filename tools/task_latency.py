"""Per-task latency table (T1-T5) from a live session's per-frame timings.

    .venv/bin/python -m highnoon --metrics-csv results/session_metrics.csv
    .venv/bin/python tools/task_latency.py results/session_metrics.csv [more.csv ...]

Prints a markdown table per session: each stage's mean and p95 in ms, grouped by the task that owns it,
plus frame rate, dropped frames and capture-to-screen latency. Pose runs on its own thread while face
detection and face ID run one after the other on the perception thread, and hands run asynchronously on
a third, so stage times do not add up to the latency: "inference" is the slower of the pose and face
paths, and "latency" is measured from camera capture to the frame being shown.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

TASKS = [
    (
        "T1 Face tracking & effects",
        [
            ("face", "face landmarks in head crops (perception thread)"),
            ("face_fx", "face damage + hat drawing (render)"),
        ],
    ),
    (
        "T2 Pose & motion",
        [
            ("pose", "pose landmarks + segmentation mask (pose thread)"),
            ("hand", "hand landmarks, 4 hands (own thread, asynchronous)"),
            ("postprocess", "landmarks to arrays, mask copy"),
            ("filter", "One Euro filtering: keypoints, aim, face, hat; control signals"),
        ],
    ),
    (
        "T3 Identity",
        [
            ("face_id", "SFace embedding per face (perception thread, after face)"),
            ("tracking", "assignment, face verification, re-identification"),
        ],
    ),
    (
        "T4 Game control",
        [
            ("game", "actions, shots, hits, powerups, rules"),
        ],
    ),
    (
        "T5 Integration & scene",
        [
            ("convert", "BGR to RGB for the models"),
            ("inference", "pose and face paths in parallel (the slower one)"),
            ("wait", "result waiting for the main loop"),
            ("camera", "background replacement + wall onto the camera frame"),
            ("render", "game layer, HUD, effects (includes face_fx)"),
            ("display", "OpenGL upload + flip"),
        ],
    ),
]
COUNTS = ("dropped", "hand_lag")


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    columns = {}
    for name in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(math.nan)
        columns[name] = np.array(values)
    return columns


def stat(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    return float(values.mean()), float(np.percentile(values, 95))


def report(path: Path) -> str:
    data = load(path)
    t = data["t"]
    frames = len(t)
    duration = float(t[-1] - t[0]) if frames > 1 else 0.0
    fps = (frames - 1) / duration if duration > 0 else math.nan
    lines = [
        f"### `{path.name}`",
        "",
        f"{frames} frames over {duration:.1f} s: **{fps:.1f} fps** average.",
        "",
        "| Task | Stage | What it covers | mean ms | p95 ms |",
        "|---|---|---|---|---|",
    ]
    for task, stages in TASKS:
        for i, (column, what) in enumerate(stages):
            if column not in data:
                continue
            mean, p95 = stat(data[column])
            lines.append(f"| {task if i == 0 else ''} | {column} | {what} | {mean:.1f} | {p95:.1f} |")
    mean, p95 = stat(data["latency"])
    lines.append(
        f"| **End to end** | latency | camera capture to frame on screen | **{mean:.1f}** | **{p95:.1f}** |"
    )
    extras = []
    for column in COUNTS:
        if column in data:
            mean, p95 = stat(data[column])
            extras.append(f"{column} {mean:.2f} frames (p95 {p95:.0f})")
    lines += ["", "Per frame: " + ", ".join(extras) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+", help="--metrics-csv files from live sessions")
    args = parser.parse_args()
    print("\n".join(report(path) for path in args.csv))


if __name__ == "__main__":
    main()
