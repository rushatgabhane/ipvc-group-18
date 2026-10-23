from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

FRAME_COUNT_METRICS = ("dropped", "hand_lag")


class Profiler:
    def __init__(self, window: int = 120, csv_path: Path | None = None):
        self.window = window
        self._samples: dict[str, deque[float]] = {}
        self._frame: dict[str, float] = {}
        self._frame_times: deque[float] = deque(maxlen=window)
        self._csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns: list[str] = []

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
        now = time.perf_counter()
        self._frame["latency"] = (now - t_capture) * 1000.0
        self._frame_times.append(now)
        for name, value in self._frame.items():
            self._samples.setdefault(name, deque(maxlen=self.window)).append(value)
        if self._csv_path is not None:
            self._write_csv_row(frame_id, now)
        self._frame = {}

    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def stats(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(np.mean(values)), float(np.percentile(values, 95)))
            for name, values in self._samples.items()
            if values
        }

    def summary(self) -> str:
        lines = [f"fps {self.fps():5.1f}"]
        for name, (mean, p95) in sorted(self.stats().items()):
            unit = "fr" if name in FRAME_COUNT_METRICS else "ms"
            lines.append(f"{name:<14s} mean {mean:6.1f} {unit}   p95 {p95:6.1f} {unit}")
        return "\n".join(lines)

    def _write_csv_row(self, frame_id: int, now: float) -> None:
        if self._csv_writer is None:
            self._open_csv()
        row = [frame_id, f"{now:.4f}", f"{self.fps():.2f}"]
        row += [f"{self._frame.get(column, float('nan')):.3f}" for column in self._csv_columns]
        self._csv_writer.writerow(row)

    def _open_csv(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_columns = sorted(self._frame)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["frame_id", "t", "fps", *self._csv_columns])

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
