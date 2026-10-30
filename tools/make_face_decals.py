"""Builds assets/sprites/face/: BGRA decals, BGR multiplies the skin (255 = unchanged), A is coverage."""

from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "sprites" / "face"
FILE_PATH = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width={}"
USER_AGENT = {"User-Agent": "highnoon-ipcv-project/1.0 (University of Twente course project)"}
SIZE = 256


def download(name: str, width: int) -> np.ndarray:
    url = FILE_PATH.format(urllib.parse.quote(name), width)
    data = urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=60).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def crop(image: np.ndarray, cx: float, cy: float, half: float) -> np.ndarray:
    height, width = image.shape[:2]
    r = int(half * width)
    x, y = int(cx * width), int(cy * height)
    return image[max(0, y - r) : y + r, max(0, x - r) : x + r]


def feather(width: int, inner: float = 0.55, height: int | None = None) -> np.ndarray:
    xs = np.linspace(-1, 1, width, dtype=np.float32)
    ys = np.linspace(-1, 1, height or width, dtype=np.float32)
    radius = np.hypot(*np.meshgrid(xs, ys))
    return np.clip((1.0 - radius) / (1.0 - inner), 0.0, 1.0) ** 1.5


def skin_ratio(patch: np.ndarray, strength: float) -> np.ndarray:
    patch = cv2.resize(patch[:, :, :3], (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    ring = feather(SIZE, 0.0) < 0.15
    skin = np.median(patch[ring], axis=0)
    ratio = np.clip(patch / skin, 0.0, 1.0)
    ratio = cv2.GaussianBlur(ratio, (0, 0), 1.2)
    darkness = 1.0 - ratio.mean(axis=2)
    alpha = np.clip(darkness * strength, 0.0, 1.0) * feather(SIZE)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0)
    return np.dstack([ratio * 255, alpha * 255]).astype(np.uint8)


def shiner(photo: np.ndarray) -> np.ndarray:
    xs = np.linspace(-1, 1, SIZE, dtype=np.float32)
    x, y = np.meshgrid(xs, xs)
    distance = np.hypot(x, np.where(y < 0, y / 0.85, y))
    socket = np.clip((1.0 - distance) / 0.6, 0.0, 1.0)
    socket = socket * socket * (3 - 2 * socket) * np.where(y < 0, 0.8, 1.0)

    texture = skin_ratio(photo, 1.0)[:, :, :3].astype(np.float32) / 255
    mottle = texture.mean(axis=2)
    mottle = np.clip(0.75 + 0.6 * (mottle.mean() - mottle) / max(1e-3, mottle.std()) * 0.25, 0.5, 1.2)

    levels = np.array([0.0, 0.3, 0.65, 1.0], np.float32)
    rim, yellow, purple, core = [0.7, 0.88, 0.96], [0.62, 0.7, 0.8], [0.48, 0.34, 0.55], [0.24, 0.14, 0.26]
    tones = np.array([rim, yellow, purple, core], np.float32)
    depth = np.clip(socket * mottle, 0, 1)
    factor = np.stack([np.interp(depth, levels, tones[:, c]) for c in range(3)], axis=2)
    factor = factor * (0.6 + 0.4 * texture)
    coverage = cv2.GaussianBlur(np.clip(socket * 1.5, 0, 1), (0, 0), 3.0)
    return np.dstack([np.clip(factor, 0, 1) * 255, coverage * 255]).astype(np.uint8)


def blood(patch: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    red = patch[:, :, 2] / 255.0
    coverage = np.clip((patch[:, :, 2] - patch[:, :, :2].max(axis=2)) / 80.0, 0, 1)
    coverage *= feather(size[0], 0.3, size[1])
    tone = np.array([0.15, 0.1, 0.84], np.float32)
    factor = tone * (0.55 + 0.45 * red[..., None])
    return np.dstack([factor * 255, coverage * 255]).astype(np.uint8)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rings = download("Bruise - ring resolution.jpg", 1000)
    spray = download("Bloodspray big.png", 1600)

    decals = {
        "shiner.png": shiner(crop(rings, 0.27, 0.25, 0.2)),
        "blood_spray.png": blood(crop(spray, 0.8, 0.36, 0.12), (SIZE, SIZE)),
    }
    for name, decal in decals.items():
        cv2.imwrite(str(OUT_DIR / name), decal)
        print(f"wrote {OUT_DIR / name} {decal.shape[1]}x{decal.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
