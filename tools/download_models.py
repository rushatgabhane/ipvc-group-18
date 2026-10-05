"""Download the MediaPipe model files into models/ and verify their checksums.

python tools/download_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://storage.googleapis.com/mediapipe-models"

# name -> (url path, sha256 of the version we tested against)
MODELS = {
    "pose_landmarker_lite.task": (
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    "pose_landmarker_full.task": (
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    ),
    "hand_landmarker.task": (
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "face_landmarker.task": (
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    ok = True
    for name, (url_path, expected) in MODELS.items():
        dest = MODELS_DIR / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE}/{url_path}", dest)
        digest = sha256(dest)
        if digest == expected:
            print(f"ok       {name}")
        else:
            ok = False
            print(f"MISMATCH {name}: {digest} (expected {expected}); upstream 'latest' may have changed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
