"""Data contracts between the five tasks.

The pipeline for one frame:

    Frame -> [T2 pose + T1 face inference] -> Perception
          -> [T3 identity]  -> Players with persistent ids
          -> [T2 filtering] -> smoothed keypoints / motion signals per player
          -> [T4 game]      -> GameState
          -> [T5 render]    -> displayed image

All image coordinates are pixels in the (mirrored) capture frame, float32.
Each module only reads the fields defined here. That keeps the task boundaries clean
and lets every owner unit-test their module with synthetic inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Frame:
    id: int
    image: np.ndarray  # HxWx3 uint8, BGR
    t_capture: float  # time.perf_counter() seconds when the frame was grabbed
    timestamp_ms: int  # monotonic ms for MediaPipe VIDEO mode

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass(slots=True)
class PoseObservation:
    """One detected body, before identity assignment (MediaPipe 33-keypoint layout)."""

    keypoints: np.ndarray  # (33, 2) pixels
    visibility: np.ndarray  # (33,) in [0, 1]
    world: np.ndarray  # (33, 3) metres, hip-centred (MediaPipe world landmarks)

    def center(self) -> np.ndarray:
        """Torso centre (mean of shoulders and hips), robust to waving arms."""
        return self.keypoints[[11, 12, 23, 24]].mean(axis=0)


@dataclass(slots=True)
class FaceObservation:
    landmarks: np.ndarray  # (478, 2) pixels
    bbox: np.ndarray  # (4,) x0, y0, x1, y1 pixels

    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2], np.float32)


@dataclass(slots=True)
class HandObservation:
    """One hand (MediaPipe 21-keypoint layout)."""

    landmarks: np.ndarray  # (21, 2) pixels
    world: np.ndarray  # (21, 3) metres, hand-centred: distances do not depend on viewing angle
    openness: float  # mean fingertip-wrist distance / wrist-middle-knuckle distance (~1.9 open, ~0.9 fist)
    frame_id: int  # frame the hand was detected in (hands run asynchronously, may lag 1-2 frames)


@dataclass(slots=True)
class Perception:
    """Raw model outputs for one frame. Produced on the perception thread."""

    frame: Frame
    poses: list[PoseObservation]
    faces: list[FaceObservation]
    mask: np.ndarray | None  # HxW float32 in [0, 1], union of all people; None if disabled
    t_done: float  # perf_counter seconds when inference finished
    timings: dict[str, float] = field(default_factory=dict)  # stage name -> ms, for the profiler
    hands: list[HandObservation] = field(default_factory=list)


@dataclass(slots=True)
class MotionSignals:
    """Geometric control signals derived from smoothed keypoints (T2). Scale-invariant where possible."""

    torso_len: float  # px, shoulder-mid to hip-mid; the unit for distance-independent thresholds
    shoulder_y: float  # px, shoulder midpoint height (for ducking)
    aim_valid: bool  # an arm is extended with reliable shoulder/elbow/wrist
    extension: float  # aiming arm shoulder-wrist distance, in torso lengths
    aim_origin: np.ndarray  # (2,) px, wrist of the aiming arm (where the shot starts)
    aim_dir: np.ndarray  # (2,) unit vector shoulder -> wrist
    elevation: float  # rad, aiming-arm angle above horizontal (left/right independent)
    arms_down: bool  # both wrists hang below the hips (gun lowered)
    hand_openness: float | None = None  # openness of the aiming hand, None if no hand detected
    head_y: float = 0.0  # px, head centre height (face box centre, else nose)


@dataclass
class Player:
    """A persistent identity (T3). Filled in progressively by T2/T1/T4."""

    id: int  # 1-based: Player 1, Player 2
    pose: PoseObservation | None = None  # latest raw pose assigned to this player
    face: FaceObservation | None = None
    hands: dict[int, HandObservation] = field(default_factory=dict)  # pose wrist index (15/16) -> hand
    smoothed: np.ndarray | None = None  # (33, 2) filtered keypoints (T2)
    signals: MotionSignals | None = None  # (T2)
    last_seen: float = 0.0
    visible: bool = False
