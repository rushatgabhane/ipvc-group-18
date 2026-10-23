from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from highnoon.contracts import FaceObservation, HandObservation, Perception, PoseObservation

SHOULDERS = [11, 12]
NOSE = 0
WRISTS = [15, 16]
HAND_WRIST = 0
FOLLOW = 0.2


class StageSpread:
    def __init__(self, min_gap: float):
        self.min_gap = min_gap
        self.split: float | None = None
        self.shift = 0.0

    def apply(self, perception: Perception) -> Perception:
        width = perception.frame.size[0]
        if self.min_gap <= 0 or not perception.poses:
            return perception
        poses = perception.poses
        centers = np.array([pose.keypoints[SHOULDERS, 0].mean() for pose in poses], np.float32)
        self._follow(centers, width)
        shift = int(round(self.shift))
        if shift == 0:
            return perception
        offsets = np.where(centers < self.split, -shift, shift)
        image, mask = spread_pixels(perception, offsets)
        return replace(
            perception,
            frame=replace(perception.frame, image=image),
            mask=mask,
            poses=[moved_pose(pose, offset) for pose, offset in zip(poses, offsets, strict=True)],
            faces=[
                moved_face(face, nearest_offset(face.center(), poses, [NOSE], offsets))
                for face in perception.faces
            ],
            hands=[
                moved_hand(hand, nearest_offset(hand.landmarks[HAND_WRIST], poses, WRISTS, offsets))
                for hand in perception.hands
            ],
            person_masks=[],
        )

    def _follow(self, centers: np.ndarray, width: int) -> None:
        if self.split is None:
            self.split = width / 2
        if centers.size < 2:
            return
        left, right = np.sort(centers)[:2]
        wanted_shift = max(0.0, (self.min_gap * width - float(right - left)) / 2)
        self.split += FOLLOW * (float(left + right) / 2 - self.split)
        self.shift += FOLLOW * (wanted_shift - self.shift)


def spread_pixels(perception: Perception, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    image = perception.frame.image
    masks = [fit_mask(mask, image) for mask in perception.person_masks]
    if len(masks) != len(offsets):
        mask = None if perception.mask is None else fit_mask(perception.mask, image)
        return shifted(image, int(offsets[0])), None if mask is None else shifted(mask, int(offsets[0]))
    out_image = np.zeros_like(image)
    out_mask = np.zeros_like(masks[0])
    for mask, owned, offset in zip(masks, ownership(masks), offsets, strict=True):
        src, dst = column_windows(image.shape[1], int(offset))
        cv2.copyTo(image[:, src], owned[:, src], out_image[:, dst])
        cv2.copyTo(mask[:, src], owned[:, src], out_mask[:, dst])
    return out_image, out_mask


def fit_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    if mask.shape == image.shape[:2]:
        return mask
    return cv2.resize(mask, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def ownership(masks: list[np.ndarray]) -> list[np.ndarray]:
    if len(masks) == 2:
        second = cv2.compare(masks[1], masks[0], cv2.CMP_GT)
        return [cv2.bitwise_not(second), second]
    owner = np.stack(masks).argmax(axis=0)
    return [(owner == i).astype(np.uint8) * 255 for i in range(len(masks))]


def column_windows(width: int, offset: int) -> tuple[slice, slice]:
    if offset >= 0:
        return slice(0, width - offset), slice(offset, width)
    return slice(-offset, width), slice(0, width + offset)


def shifted(array: np.ndarray, offset: int) -> np.ndarray:
    out = np.zeros_like(array)
    src, dst = column_windows(array.shape[1], offset)
    out[:, dst] = array[:, src]
    return out


def nearest_offset(
    point: np.ndarray, poses: list[PoseObservation], joints: list[int], offsets: np.ndarray
) -> float:
    anchors = np.stack([pose.keypoints[joints] for pose in poses])
    distances = np.linalg.norm(anchors - point, axis=2).min(axis=1)
    return float(offsets[int(np.argmin(distances))])


def moved_pose(pose: PoseObservation, offset: float) -> PoseObservation:
    keypoints = pose.keypoints.copy()
    keypoints[:, 0] += offset
    return PoseObservation(keypoints, pose.visibility)


def moved_face(face: FaceObservation, offset: float) -> FaceObservation:
    landmarks = face.landmarks.copy()
    landmarks[:, 0] += offset
    bbox = face.bbox.copy()
    bbox[[0, 2]] += offset
    return FaceObservation(landmarks, bbox)


def moved_hand(hand: HandObservation, offset: float) -> HandObservation:
    landmarks = hand.landmarks.copy()
    landmarks[:, 0] += offset
    return HandObservation(landmarks, hand.openness, hand.frame_id)
