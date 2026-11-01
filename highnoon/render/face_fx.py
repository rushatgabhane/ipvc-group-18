from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from highnoon.contracts import HeadAnchor


class FaceEffectRenderer:
    def __init__(self, alpha: float = 0.9):

        self.alpha = float(np.clip(alpha, 0.0, 1.0))

        hat_path = Path(__file__).resolve().parents[2] / "assets" / "sprites" / "cowboy_hat.png"
        self.hat = cv2.imread(str(hat_path), cv2.IMREAD_UNCHANGED)

        if self.hat is None:
            raise FileNotFoundError(f"Could not load hat image: {hat_path}")

        if self.hat.ndim != 3 or self.hat.shape[2] != 4:
            raise ValueError("Cowboy hat PNG must have an alpha channel")

        # warpAffine aliases when shrinking by more than half, so it starts from a suitable size.
        self.sprites = [self.hat]
        while min(self.sprites[-1].shape[:2]) > 32:
            sprite = self.sprites[-1]
            size = (sprite.shape[1] // 2, sprite.shape[0] // 2)
            self.sprites.append(cv2.resize(sprite, size, interpolation=cv2.INTER_AREA))

    def draw(self, image: np.ndarray, head: HeadAnchor) -> np.ndarray:
        """Draws the hat where the (already smoothed) head anchor puts it."""
        center_x, center_y = (float(v) for v in head.center)
        angle = head.angle
        target_width = head.width
        if target_width < 4:
            return image
        sprite = next((s for s in reversed(self.sprites) if s.shape[1] >= target_width), self.hat)
        h, w = sprite.shape[:2]
        scale = target_width / w

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(-angle), scale)
        matrix[0, 2] += center_x - w / 2
        matrix[1, 2] += center_y - h / 2

        corners = matrix @ np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T

        x0 = max(0, int(np.floor(corners[0].min())))
        y0 = max(0, int(np.floor(corners[1].min())))
        x_end = min(image.shape[1], int(np.ceil(corners[0].max())) + 1)
        y_end = min(image.shape[0], int(np.ceil(corners[1].max())) + 1)

        if x0 >= x_end or y0 >= y_end:
            return image

        matrix[0, 2] -= x0
        matrix[1, 2] -= y0
        hat_crop = cv2.warpAffine(
            sprite,
            matrix,
            (x_end - x0, y_end - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        rgb = hat_crop[:, :, :3].astype(np.float32)
        alpha = hat_crop[:, :, 3:4].astype(np.float32) / 255
        alpha *= self.alpha

        roi = image[y0:y_end, x0:x_end].astype(np.float32)

        blended = rgb * alpha + roi * (1.0 - alpha)

        image[y0:y_end, x0:x_end] = np.clip(blended, 0, 255).astype(np.uint8)

        return image
