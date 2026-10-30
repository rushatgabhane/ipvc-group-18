from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"
OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = {
    "pose_landmarker_lite.task": (
        f"{MEDIAPIPE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    "hand_landmarker.task": (
        f"{MEDIAPIPE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "face_landmarker.task": (
        f"{MEDIAPIPE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    ok = True
    for name, (url, expected) in MODELS.items():
        dest = MODELS_DIR / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        digest = sha256(dest)
        if digest == expected:
            print(f"ok       {name}")
        else:
            ok = False
            print(f"MISMATCH {name}: {digest} (expected {expected}); upstream 'latest' may have changed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
