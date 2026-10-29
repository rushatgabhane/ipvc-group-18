from __future__ import annotations

import cv2
import numpy as np


class FaceEffectRenderer:

    def __init__(self, alpha: float = 0.9):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def _rotate(self,
                points: np.ndarray,
                center: tuple[float, float],
                angle: float) -> np.ndarray: 
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype = np.float32)
        return points.astype(np.float32) @ rotation.T + np.asarray(
                    center, dtype = np.float32).astype(np.int32)

    def draw(self, image: np.darray, face,
             color: tuple[int, int, int]) -> np.darray:
        points = np.asarray(face.landmarks, dtype = np.float32)
        x1, y1, x2, y2 = np.asarray(face.bbox).reshape(-1)[:4]
        width, height = float(x2-x1), float(y2-y1)

        if width < 8 or height < 8:
            return image

        angle = 0.0
        if len(points) > 473:
            left_eye, right_eye = points[468], points[473] #mediapipe eyes iris center landmarks
            angle = float(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
        
        hat_width = width * 1.1
        hat_height = height * 0.3

        center = ((x1 + x2) / 2, y1 + height * 0.02)

        theta = np.linspace(0, 2*np.pi, 48, endpoint = False)
        brim = np.column_stack((
            hat_width * 0.5 * np.cos(theta),
            hat_height * 0.2 + hat_height * 0.2 * np.sin(theta)
        ))

        brim = self._rotate(brim, center, angle)

        cover = np.array([
            [-0.3 * hat_width, 0.15 * hat_height],
            [-0.2 * hat_width, -0.7 * hat_height],
            [0.15 * hat_width, -0.7 * hat_height],
            [0.3 * hat_width, 0.1 * hat_height]
        ], dtype = np.float32)

        cover = self._rotate(cover, center, angle)

        overlay = image.copy()
        cv2.fillPoly(overlay, 
                    [np.round(brim).astype(np.int32)],
                    color, cv2.LINE_AA)
        cv2.fillPoly(overlay,
                    [np.round(cover).astype(np.int32)],
                    color, cv2.LINE_AA)

        h, w = image.shape[:2]
        cx, cy = map(int, center)
        pad = int(hat_width * 0.65)
        xa, xb = max(0, cx - pad), min(w, cx + pad)
        ya, yb = max(0, cy - pad), min(h, cy + int(hat_height * 0.5))

        if xa < xb and ya < yb:
            image[ya:yb, xa:xb] = cv2.addWeighted(
                overlay[ya:yb, xa:xb], self.alpha,
                image[ya:yb, xa:xb], 1 - self.alpha, 0
            )
            return image