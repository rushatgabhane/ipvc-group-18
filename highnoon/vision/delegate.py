from __future__ import annotations

import json
import platform
import subprocess
import sys

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions

from highnoon.config import MODELS_DIR, PerceptionConfig

CACHE_FILE = MODELS_DIR / ".delegate_cache.json"
GPU_PLATFORMS = ("Windows", "Linux")

GPU_PROBE_SCRIPT = """
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
gpu = BaseOptions.Delegate.GPU
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((360, 640, 3), np.uint8))
pose, face, hand = sys.argv[1:4]
vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose, delegate=gpu))).detect(image)
vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face, delegate=gpu))).detect(image)
vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand, delegate=gpu))).detect(image)
print("GPU_OK")
"""


def resolve_delegate(cfg: PerceptionConfig) -> BaseOptions.Delegate:
    if cfg.delegate == "cpu":
        return BaseOptions.Delegate.CPU
    if cfg.delegate == "gpu":
        return BaseOptions.Delegate.GPU
    if platform.system() not in GPU_PLATFORMS:
        return BaseOptions.Delegate.CPU
    ok = gpu_available(cfg)
    print(f"[highnoon] inference on {'GPU' if ok else 'CPU (GPU probe failed)'}; override with --delegate")
    return BaseOptions.Delegate.GPU if ok else BaseOptions.Delegate.CPU


def gpu_available(cfg: PerceptionConfig) -> bool:
    models = [cfg.pose_model_path, cfg.face_model_path, cfg.hand_model_path]
    machine = f"{platform.system()}-{platform.machine()}-mediapipe{mp.__version__}"
    key = machine + "-" + "-".join(model.stem for model in models)
    cache = read_cache()
    if key not in cache:
        cache[key] = probe_gpu([str(model) for model in models])
        write_cache(cache)
    return bool(cache[key])


def probe_gpu(model_paths: list[str]) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", GPU_PROBE_SCRIPT, *model_paths],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and "GPU_OK" in result.stdout


def read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def write_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except OSError:
        pass
