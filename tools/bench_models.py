"""Benchmark model inference in isolation (no rendering), to choose models and settings.

python tools/bench_models.py                    # 60 frames from the webcam
python tools/bench_models.py --source clip.mp4  # repeatable, from a recorded clip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from highnoon.config import PerceptionConfig  # noqa: E402
from highnoon.contracts import Frame  # noqa: E402
from highnoon.vision.perception import Perceiver  # noqa: E402


def grab_frames(source: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frames = []
    for i in range(n + 30):
        ok, f = cap.read()
        if not ok:
            break
        if i >= 30 or not source.isdigit():  # let camera exposure settle
            frames.append(cv2.flip(f, 1))
    cap.release()
    return frames[:n]


def bench(label: str, cfg: PerceptionConfig, frames: list[np.ndarray]) -> None:
    perceiver = Perceiver(cfg)
    times, poses, faces = [], [], []
    for i, img in enumerate(frames):
        frame = Frame(i + 1, img, time.perf_counter(), (i + 1) * 33)
        t = time.perf_counter()
        result = perceiver(frame)
        times.append((time.perf_counter() - t) * 1000)
        poses.append(len(result.poses))
        faces.append(len(result.faces))
    perceiver.close()
    t = np.array(times[5:])  # skip warm-up
    print(f"{label:<34s} mean {t.mean():6.1f} ms  p95 {np.percentile(t, 95):6.1f} ms  "
          f"poses {np.mean(poses):.1f}  faces {np.mean(faces):.1f}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--delegate", choices=["auto", "cpu", "gpu"], default="cpu")
    a = ap.parse_args()
    frames = grab_frames(a.source, a.frames)
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}\n")

    variants = {
        "pose lite": dict(run_face=False, segmentation=False),
        "pose full": dict(pose_model="pose_landmarker_full", run_face=False, segmentation=False),
        "pose lite + seg": dict(run_face=False),
        "pose lite + seg + face, sequential": dict(parallel=False),
        "pose lite + seg + face, parallel": dict(),
        "pose full + seg + face, parallel": dict(pose_model="pose_landmarker_full"),
    }
    for label, overrides in variants.items():
        cfg = PerceptionConfig(delegate=a.delegate)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        bench(label, cfg, frames)


if __name__ == "__main__":
    main()
