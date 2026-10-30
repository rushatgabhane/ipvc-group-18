from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from highnoon.contracts import FaceObservation

# Face mesh landmarks used for alignment: both iris centres, nose tip and the two mouth corners.
IRISES = (468, 473)
NOSE_TIP = 1
MOUTH_CORNERS = (61, 291)
# Where those five points sit in the 112x112 crop the recognition model was trained on (image-left first).
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float32,
)
CROP_SIZE = (112, 112)
MIN_EYE_DISTANCE_PX = 18.0
MAX_NOSE_OFFSET = 0.35


class FaceEmbedder:
    """Turns a tracked face into a 128-d identity vector (OpenCV SFace); unit length, compared by dot product.

    Unlike clothing colour, two different people almost never produce similar vectors, so this is what
    tells a returning player from a stranger.
    """

    def __init__(self, model_path: Path):
        self.model = cv2.FaceRecognizerSF.create(str(model_path), "")

    def __call__(self, image: np.ndarray, face: FaceObservation) -> np.ndarray | None:
        points = alignment_points(face.landmarks)
        if points is None:
            return None
        transform, _ = cv2.estimateAffinePartial2D(points, TEMPLATE, method=cv2.LMEDS)
        if transform is None:
            return None
        crop = cv2.warpAffine(image, transform, CROP_SIZE, borderMode=cv2.BORDER_REPLICATE)
        feature = self.model.feature(crop).flatten()
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else None


def alignment_points(landmarks: np.ndarray) -> np.ndarray | None:
    """The five alignment points, or None when the face is too small or turned too far to identify."""
    eyes = sorted((landmarks[i] for i in IRISES), key=lambda p: p[0])
    mouth = sorted((landmarks[i] for i in MOUTH_CORNERS), key=lambda p: p[0])
    eye_distance = float(np.linalg.norm(eyes[1] - eyes[0]))
    if eye_distance < MIN_EYE_DISTANCE_PX:
        return None
    nose = landmarks[NOSE_TIP]
    if abs(float(nose[0] - (eyes[0][0] + eyes[1][0]) / 2)) > MAX_NOSE_OFFSET * eye_distance:
        return None
    return np.array([eyes[0], eyes[1], nose, mouth[0], mouth[1]], np.float32)
