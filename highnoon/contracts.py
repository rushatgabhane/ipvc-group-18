from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TORSO_KEYPOINTS = [11, 12, 23, 24]


@dataclass(slots=True)
class Frame:
    id: int
    image: np.ndarray
    t_capture: float
    timestamp_ms: int

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        return width, height


@dataclass(slots=True)
class PoseObservation:
    keypoints: np.ndarray
    visibility: np.ndarray

    def center(self) -> np.ndarray:
        return self.keypoints[TORSO_KEYPOINTS].mean(axis=0)


@dataclass(slots=True)
class FaceObservation:
    landmarks: np.ndarray
    bbox: np.ndarray

    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)


@dataclass(slots=True)
class HandObservation:
    landmarks: np.ndarray
    openness: float
    frame_id: int


@dataclass(slots=True)
class Perception:
    frame: Frame
    poses: list[PoseObservation]
    faces: list[FaceObservation]
    mask: np.ndarray | None
    t_done: float
    timings: dict[str, float] = field(default_factory=dict)
    hands: list[HandObservation] = field(default_factory=list)


@dataclass(slots=True)
class MotionSignals:
    torso_len: float
    aim_valid: bool
    extension: float
    aim_origin: np.ndarray
    aim_dir: np.ndarray
    elevation: float
    hand_openness: float | None = None
    head_y: float = 0.0


@dataclass
class Player:
    id: int
    pose: PoseObservation | None = None
    face: FaceObservation | None = None
    hands: dict[int, HandObservation] = field(default_factory=dict)
    smoothed: np.ndarray | None = None
    signals: MotionSignals | None = None
    last_seen: float = 0.0
    visible: bool = False
