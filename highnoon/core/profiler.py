"""Per-stage timing, end-to-end latency and FPS, with optional CSV export for the report.

All recording happens on the main thread: the perception thread stores its own timings
on the Perception object, and the main loop merges them in `end_frame`.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class Profiler:
    def __init__(self, window: int = 120, csv_path: Path | None = None):
        self.window = window
        self._samples: dict[str, deque[float]] = {}
        self._frame: dict[str, float] = {}
        self._frame_times: deque[float] = deque(maxlen=window)
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns: list[str] | None = None
        self._csv_path = csv_path

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._frame[name] = (time.perf_counter() - t0) * 1000.0

    def add(self, timings: dict[str, float]) -> None:
        self._frame.update(timings)

    def end_frame(self, frame_id: int, t_capture: float) -> None:
        """Call right after the frame is shown. Latency = capture -> display."""
        now = time.perf_counter()
        self._frame["latency"] = (now - t_capture) * 1000.0
        self._frame_times.append(now)
        for name, ms in self._frame.items():
            self._samples.setdefault(name, deque(maxlen=self.window)).append(ms)
        if self._csv_path is not None:
            self._write_csv(frame_id, now)
        self._frame = {}

    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def stats(self) -> dict[str, tuple[float, float]]:
        """Stage -> (mean ms, p95 ms) over the rolling window."""
        return {
            name: (float(np.mean(s)), float(np.percentile(s, 95))) for name, s in self._samples.items() if s
        }

    def summary(self) -> str:
        lines = [f"fps {self.fps():5.1f}"]
        for name, (mean, p95) in sorted(self.stats().items()):
            unit = "fr" if name == "dropped" else "ms"
            lines.append(f"{name:<14s} mean {mean:6.1f} {unit}   p95 {p95:6.1f} {unit}")
        return "\n".join(lines)

    def _write_csv(self, frame_id: int, now: float) -> None:
        if self._csv_writer is None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, "w", newline="")
            # Columns are fixed by the first frame; later stages that appear are ignored.
            self._csv_columns = sorted(self._frame)
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame_id", "t", "fps", *self._csv_columns])
        row = [frame_id, f"{now:.4f}", f"{self.fps():.2f}"]
        row += [f"{self._frame.get(c, float('nan')):.3f}" for c in self._csv_columns]
        self._csv_writer.writerow(row)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
