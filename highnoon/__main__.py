"""Entry point: python -m highnoon [options]"""

from __future__ import annotations

import argparse
from pathlib import Path

from highnoon.app import run
from highnoon.config import Config


def parse_args() -> Config:
    ap = argparse.ArgumentParser(prog="highnoon", description="HIGH NOON - webcam duel shooter")
    ap.add_argument("--source", default="0", help="camera index or video file (default: 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the image horizontally")
    ap.add_argument("--fast", action="store_true", help="video files: process as fast as possible")
    ap.add_argument("--loop", action="store_true", help="video files: loop forever")
    ap.add_argument("--pose-model", choices=["lite", "full"], default="lite")
    ap.add_argument("--no-face", action="store_true", help="disable face landmarker")
    ap.add_argument("--no-seg", action="store_true", help="disable segmentation mask / background")
    ap.add_argument("--sequential", action="store_true", help="run pose and face sequentially")
    ap.add_argument(
        "--delegate", choices=["auto", "cpu", "gpu"], default="auto",
        help="inference device; auto = GPU on Windows/Linux when it works, else CPU",
    )  # fmt: skip
    ap.add_argument("--headless", action="store_true", help="no window (benchmarks)")
    ap.add_argument("--display", choices=["gl", "cv"], default="gl", help="window backend")
    ap.add_argument("--vsync", action="store_true", help="sync to display refresh (adds latency)")
    ap.add_argument("--metrics-csv", type=Path, help="write per-frame timings to this CSV")
    ap.add_argument("--max-frames", type=int, help="stop after N frames")
    a = ap.parse_args()

    cfg = Config()
    cfg.capture.source = a.source
    cfg.capture.width, cfg.capture.height = a.width, a.height
    cfg.capture.mirror = not a.no_mirror
    cfg.capture.realtime = not a.fast
    cfg.capture.loop = a.loop
    cfg.perception.pose_model = f"pose_landmarker_{a.pose_model}"
    cfg.perception.run_face = not a.no_face
    cfg.perception.segmentation = not a.no_seg
    cfg.perception.parallel = not a.sequential
    cfg.perception.delegate = a.delegate
    cfg.display.enabled = not a.headless
    cfg.display.backend = a.display
    cfg.display.vsync = a.vsync
    cfg.display.background = not a.no_seg
    cfg.metrics_csv = a.metrics_csv
    cfg.max_frames = a.max_frames
    return cfg


def main() -> None:
    profiler = run(parse_args())
    print("\n" + profiler.summary())


if __name__ == "__main__":
    main()
