from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np


class FaceEffectRenderer:

    def __init__(self, alpha: float = 0.9):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

        hat_path = Path(__file__).resolve().parents[2] / "assets" / "sprites" / "cowboy_hat.png"
        self.hat = cv2.imread(str(hat_path), cv2.IMREAD_UNCHANGED)

        if self.hat is None:
            raise FileNotFoundError(f"Could not load hat image: {hat_path}")

        if self.hat.ndim != 3 or self.hat.shape[2] != 4:
            raise ValueError("Cowboy hat PNG must have an alpha channel")

    @staticmethod
    def rotate_hat(
                hat: np.ndarray,
                angle: float) -> np.ndarray: 
        h, w = hat.shape[:2]
        center = (w//2, h//2)

        matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)

        cos = abs(matrix[0,0])
        sin = abs(matrix[0,1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        matrix[0, 2] += new_w /2 - center[0]
        matrix[1, 2] += new_h /2 - center[1]

        return cv2.warpAffine(hat, matrix, (new_w, new_h),
                              flags = cv2.INTER_LINEAR,
                              borderMode = cv2.BORDER_CONSTANT,
                              borderValue = (0,0,0,0))


    def draw(self, image: np.darray, face,
             color=None) -> np.darray:
        points = np.asarray(face.landmarks, dtype = np.float32)
        x1, y1, x2, y2 = np.asarray(face.bbox).reshape(-1)[:4]
        face_width = float(x2-x1)
        face_height = float(y2-y1)

        if face_width < 4 or face_height < 4:
            return image

        angle = 0.0

        if len(points) > 473:
            left_eye, right_eye = points[468], points[473] #mediapipe eyes iris center landmarks

            angle = float(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))

            eye_center = (left_eye + right_eye) / 2
            eye_distance = np.linalg.norm(right_eye - left_eye)

            offset_x = 0
            offset_y = -eye_distance * 1.8

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            rotated_offset_x = offset_x * cos_a - offset_y * sin_a
            rotated_offset_y = offset_x * sin_a + offset_y * cos_a

            center_x = int(eye_center[0] + rotated_offset_x)
            center_y = int(eye_center[1] + rotated_offset_y)

        else:
            center_x = int((x1 + x2) / 2)
            center_y = int(y1 - target_height * 0.35)

        original_h, original_w = self.hat.shape[:2]

        target_width = int(face_width * 1.75)
        scale = target_width / original_w

        target_height = max(1, int(original_h * scale))

        hat = cv2.resize(self.hat,
                         (target_width, target_height),
                         interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        hat = self.rotate_hat(hat, -angle)


        h, w = hat.shape[:2]

        x_start = center_x  - w // 2
        y_start = center_y - h // 2

        x0 = max(0, x_start)
        y0 = max(0, y_start)
        x_end = min(image.shape[1], x_start + w)
        y_end = min(image.shape[0], y_start + h)

        if x0 >= x_end or y0 >= y_end:
            return image

        hat_x0 = x0 - x_start
        hat_y0 = y0 - y_start

        hat_x1 = hat_x0 + (x_end - x0)
        hat_y1 = hat_y0 + (y_end - y0)

        hat_crop = hat[hat_y0:hat_y1, hat_x0:hat_x1]
        
        rgb = hat_crop[:, :, :3].astype(np.float32)
        alpha = hat_crop[:, :, 3:4].astype(np.float32) / 255
        alpha *= self.alpha

        roi = image[y0:y_end, x0:x_end].astype(np.float32)

        blended = rgb * alpha + roi * (1.0 - alpha)

        image[y0:y_end, x0:x_end] = np.clip(blended, 0, 255).astype(np.uint8)

        return image