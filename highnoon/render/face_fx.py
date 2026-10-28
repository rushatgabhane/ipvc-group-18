from __future__ import annotations

import cv2
import numpy as np


class FaceEffectRenderer:

    def __init__(self, alpha: float = 0.9):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))


    def _rotate(points: np.ndarray,
                center: tuple[float, float],
                angle: float) -> np.ndarray: 
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype = np.float32)
        return (points.astype(np.float32) @ rotation.T +
                np.asarray(center, dtype = np.float32).astype(np.int32))

    def _get_face_data(face):
        points = np.asarray(face.points, dtype = np.float32)
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype = np.float32).reshape(-1)[:4]
        return points, x1, y1, x2, y2