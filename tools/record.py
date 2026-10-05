"""Record a raw (unmirrored) webcam clip for repeatable tests and benchmarks.

    python tools/record.py clips/crossing.mp4 --seconds 20

Replay it through the full pipeline with:
    python -m highnoon --source clips/crossing.mp4 --metrics-csv results/crossing.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot read from camera")
    h, w = frame.shape[:2]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))

    print(f"recording {w}x{h} for {a.seconds}s - press q to stop early")
    t_end = time.perf_counter() + a.seconds
    n = 0
    while time.perf_counter() < t_end:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
        cv2.imshow("recording (mirrored preview)", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"wrote {n} frames to {a.output}")


if __name__ == "__main__":
    main()
