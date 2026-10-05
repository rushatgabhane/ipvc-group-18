"""Choose CPU or GPU inference for the MediaPipe models.

"auto" (the default) uses the GPU on Windows and Linux when it actually works, and CPU otherwise.
MediaPipe documents GPU support for its Python tasks as "limited to Ubuntu", and when the GPU
delegate is unavailable it can abort the whole process instead of raising an exception
(on macOS it does: `graph_service.h: Check failed`). So GPU support is first tested in a
throwaway subprocess, and the result is cached per machine so the check only runs once.
macOS is never probed: there the GPU delegate is known to abort.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions

from highnoon.config import MODELS_DIR, PerceptionConfig

CACHE_FILE = MODELS_DIR / ".delegate_cache.json"

# Creates every model the game uses on the GPU and runs one inference.
_PROBE = """
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
gpu = BaseOptions.Delegate.GPU
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((360, 640, 3), np.uint8))
pose, face, hand = sys.argv[1:4]
vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose, delegate=gpu))).detect(image)
if face != "-":
    vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face, delegate=gpu))).detect(image)
if hand != "-":
    vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand, delegate=gpu))).detect(image)
print("GPU_OK")
"""


def resolve_delegate(cfg: PerceptionConfig) -> BaseOptions.Delegate:
    choice = cfg.delegate
    if choice == "cpu":
        return BaseOptions.Delegate.CPU
    if choice == "gpu":
        return BaseOptions.Delegate.GPU
    if platform.system() not in ("Windows", "Linux"):
        return BaseOptions.Delegate.CPU
    ok = gpu_available(cfg)
    print(f"[highnoon] inference on {'GPU' if ok else 'CPU (GPU probe failed)'}; override with --delegate")
    return BaseOptions.Delegate.GPU if ok else BaseOptions.Delegate.CPU


def gpu_available(cfg: PerceptionConfig, use_cache: bool = True) -> bool:
    models = [
        cfg.pose_model_path,
        cfg.face_model_path if cfg.run_face else None,
        cfg.hand_model_path if cfg.run_hands else None,
    ]
    key = f"{platform.system()}-{platform.machine()}-mediapipe{mp.__version__}-" + "-".join(
        m.stem for m in models if m is not None
    )
    cache = _read_cache() if use_cache else {}
    if key in cache:
        return bool(cache[key])
    ok = _probe([str(m) if m is not None else "-" for m in models])
    cache[key] = ok
    _write_cache(cache)
    return ok


def _probe(model_args: list[str]) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, *model_args],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and "GPU_OK" in result.stdout


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_cache(cache: dict) -> None:
    try:
        Path(CACHE_FILE).write_text(json.dumps(cache, indent=2))
    except OSError:
        pass  # caching is an optimisation only
