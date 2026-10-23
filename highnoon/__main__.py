from __future__ import annotations

import argparse
from pathlib import Path

from highnoon.app import run
from highnoon.config import Config


def parse_args() -> Config:
    parser = argparse.ArgumentParser(prog="highnoon", description="HIGH NOON - webcam duel shooter")
    parser.add_argument("--source", default="0", help="camera index or video file (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true", help="do not flip the image horizontally")
    parser.add_argument("--fast", action="store_true", help="video files: process as fast as possible")
    parser.add_argument("--loop", action="store_true", help="video files: loop forever")
    parser.add_argument(
        "--delegate",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="inference device; auto = GPU on Windows/Linux when it works, else CPU",
    )
    parser.add_argument("--headless", action="store_true", help="no window")
    parser.add_argument("--mute", action="store_true", help="no sound effects")
    parser.add_argument("--vsync", action="store_true", help="sync to display refresh (adds latency)")
    parser.add_argument("--metrics-csv", type=Path, help="write per-frame timings to this CSV")
    parser.add_argument("--signals-csv", type=Path, help="write per-frame hand/aim/duck signals per player")
    parser.add_argument(
        "--games-csv", type=Path, default=Path("results/games.csv"), help="append per-game eval stats here"
    )
    parser.add_argument("--no-games-csv", action="store_true", help="do not log per-game stats")
    parser.add_argument("--record", type=Path, help="save the raw camera video of the session (for replay)")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    args = parser.parse_args()

    cfg = Config()
    cfg.capture.source = args.source
    cfg.capture.width, cfg.capture.height = args.width, args.height
    cfg.capture.mirror = not args.no_mirror
    cfg.capture.realtime = not args.fast
    cfg.capture.loop = args.loop
    cfg.perception.delegate = args.delegate
    cfg.display.enabled = not args.headless
    cfg.display.vsync = args.vsync
    cfg.display.sound = not args.mute
    cfg.metrics_csv = args.metrics_csv
    cfg.signals_csv = args.signals_csv
    cfg.games_csv = None if args.no_games_csv else args.games_csv
    cfg.record_video = args.record
    cfg.max_frames = args.max_frames
    return cfg


def main() -> None:
    profiler = run(parse_args())
    print("\n" + profiler.summary())


if __name__ == "__main__":
    main()
