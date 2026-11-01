from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from highnoon.config import PerceptionConfig
from highnoon.contracts import FaceObservation, Frame, HandObservation, Perception, PoseObservation
from highnoon.core.capture import FrameSource
from highnoon.core.guard import ModuleGuard
from highnoon.vision.delegate import create_with_fallback, resolve_delegate
from highnoon.vision.face_id import FaceEmbedder

FINGERTIPS = [8, 12, 16, 20]
WRIST, MIDDLE_KNUCKLE = 0, 9
# The face detector is short-range: it shrinks its input to 128x128, so on a whole 1280x720 frame a face
# 2-3 m away is only a few pixels wide and is missed. Faces are therefore looked for in a square around
# each body's head (from the previous frame's pose), sized from the body so the face fills the crop.
POSE_NOSE, POSE_EARS, POSE_SHOULDERS = 0, (7, 8), (11, 12)
HEAD_CROP_PER_SHOULDER = 1.6
HEAD_CROP_PER_EARS = 3.5
MIN_HEAD_CROP_PX = 160


class Perceiver:
    def __init__(self, cfg: PerceptionConfig, guard: ModuleGuard | None = None):
        self.cfg = cfg
        self.guard = guard or ModuleGuard()
        for path in (cfg.pose_model_path, cfg.face_model_path, cfg.hand_model_path, cfg.face_id_model_path):
            if not path.exists():
                print(f"[highnoon] {path} missing - run: python tools/download_models.py")
        delegate = resolve_delegate(cfg)
        guard = self.guard
        self.pose = guard.call("pose", create_with_fallback, create_pose_landmarker, cfg, delegate, "pose")
        self.face = guard.call("face", create_with_fallback, create_face_landmarker, cfg, delegate, "face")
        self.face_crops = guard.call(
            "face", create_with_fallback, create_face_crop_landmarker, cfg, delegate, "face crops"
        )
        self._head_boxes: list[tuple[int, int, int, int]] = []
        hand = guard.call("hand", create_with_fallback, create_hand_landmarker, cfg, delegate, "hand")
        self.hands = HandWorker(hand, guard) if hand is not None else None
        self.face_id = guard.call("face_id", FaceEmbedder, cfg.face_id_model_path)
        self._pose_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")

    def __call__(self, frame: Frame) -> Perception:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timings["convert"] = elapsed_ms(t0)
        if self.hands is not None:
            self.hands.submit(frame, image)

        def detect(name, landmarker):
            started = time.perf_counter()
            result = None
            if landmarker is not None:
                result = self.guard.call(name, landmarker.detect_for_video, image, frame.timestamp_ms)
            timings[name] = elapsed_ms(started)
            return result

        t_infer = time.perf_counter()
        pose_future = self._pose_thread.submit(detect, "pose", self.pose)
        width, height = frame.size
        faces = []
        if self._head_boxes and self.face_crops is not None:
            started = time.perf_counter()
            faces = self.guard.call(
                "face", faces_in_crops, self.face_crops, rgb, self._head_boxes, fallback=[]
            )
            timings["face"] = elapsed_ms(started)
        else:  # nobody seen yet: search the whole frame (finds only faces close to the camera)
            face_result = detect("face", self.face)
            if face_result is not None:
                faces = self.guard.call("face", convert_faces, face_result, width, height, fallback=[])
        if self.face_id is not None and frame.id % self.cfg.face_id_every == 0:
            started = time.perf_counter()
            for face in faces:
                face.embedding = self.guard.call("face_id", self.face_id, frame.image, face)
            timings["face_id"] = elapsed_ms(started)
        pose_result = pose_future.result()
        timings["inference"] = elapsed_ms(t_infer)

        t_post = time.perf_counter()
        poses, mask = [], None
        if pose_result is not None:
            poses = self.guard.call("pose", convert_poses, pose_result, width, height, fallback=[])
            mask = self.guard.call("pose", merge_person_masks, pose_result)
        self._head_boxes = [head_box(pose, width, height) for pose in poses]
        hands, timings["hand"] = self.hands.latest() if self.hands is not None else ([], 0.0)
        timings["hand_lag"] = float(frame.id - hands[0].frame_id) if hands else 0.0
        timings["postprocess"] = elapsed_ms(t_post)
        return Perception(frame, poses, faces, mask, time.perf_counter(), timings, hands)

    def close(self) -> None:
        self._pose_thread.shutdown(wait=True)
        if self.hands is not None:
            self.guard.call("hand", self.hands.close)
        for name, landmarker in (("pose", self.pose), ("face", self.face), ("face", self.face_crops)):
            if landmarker is not None:
                self.guard.call(name, landmarker.close)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def base_options(path, delegate) -> BaseOptions:
    return BaseOptions(model_asset_path=str(path), delegate=delegate)


def create_pose_landmarker(cfg: PerceptionConfig, delegate):
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options(cfg.pose_model_path, delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg.max_players + cfg.extra_people,
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_segmentation_masks=True,
        )
    )


def create_face_landmarker(cfg: PerceptionConfig, delegate):
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base_options(cfg.face_model_path, delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg.max_players + cfg.extra_people,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
    )


def create_face_crop_landmarker(cfg: PerceptionConfig, delegate):
    """Single images (no tracking between frames), because each crop moves with its player."""
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base_options(cfg.face_model_path, delegate),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_presence_confidence,
        )
    )


def head_box(pose: PoseObservation, width: int, height: int) -> tuple[int, int, int, int]:
    """A square around a body's head, big enough for the face at any distance, kept inside the frame."""
    kp = pose.keypoints
    shoulders = float(np.linalg.norm(kp[POSE_SHOULDERS[0]] - kp[POSE_SHOULDERS[1]]))
    ears = float(np.linalg.norm(kp[POSE_EARS[0]] - kp[POSE_EARS[1]]))
    side = max(HEAD_CROP_PER_SHOULDER * shoulders, HEAD_CROP_PER_EARS * ears, MIN_HEAD_CROP_PX)
    side = int(min(side, width, height))
    cx, cy = kp[POSE_NOSE]
    x0 = int(np.clip(cx - side / 2, 0, width - side))
    y0 = int(np.clip(cy - side / 2, 0, height - side))
    return x0, y0, x0 + side, y0 + side


def faces_in_crops(
    landmarker, rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> list[FaceObservation]:
    faces: list[FaceObservation] = []
    for x0, y0, x1, y1 in boxes:
        crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb[y0:y1, x0:x1]))
        for face in convert_faces(landmarker.detect(crop), x1 - x0, y1 - y0):
            face.landmarks += (x0, y0)
            face.bbox += (x0, y0, x0, y0)
            faces.append(face)
    return without_duplicates(faces)


def without_duplicates(faces: list[FaceObservation]) -> list[FaceObservation]:
    """Two heads close together put the same face in both crops: keep it once."""
    kept: list[FaceObservation] = []
    for face in faces:
        size = float(face.bbox[2] - face.bbox[0])
        if all(np.linalg.norm(face.center() - other.center()) > 0.5 * size for other in kept):
            kept.append(face)
    return kept


def create_hand_landmarker(cfg: PerceptionConfig, delegate):
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base_options(cfg.hand_model_path, delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
    )


class HandWorker:
    def __init__(self, landmarker, guard: ModuleGuard):
        self.landmarker = landmarker
        self.guard = guard
        self._pending: tuple[Frame, mp.Image] | None = None
        self._result: list[HandObservation] = []
        self._ms = 0.0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, name="hands", daemon=True)
        self._thread.start()

    def submit(self, frame: Frame, image: mp.Image) -> None:
        with self._cond:
            self._pending = (frame, image)
            self._cond.notify()

    def latest(self) -> tuple[list[HandObservation], float]:
        with self._cond:
            return self._result, self._ms

    def _detect_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or not self._running)
                if not self._running:
                    return
                frame, image = self._pending
                self._pending = None
            started = time.perf_counter()
            hands = self.guard.call("hand", self._detect, frame, image, fallback=[])
            with self._cond:
                self._result = hands
                self._ms = elapsed_ms(started)

    def _detect(self, frame: Frame, image: mp.Image) -> list[HandObservation]:
        result = self.landmarker.detect_for_video(image, frame.timestamp_ms)
        return convert_hands(result, *frame.size, frame.id)

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self.landmarker.close()


class PerceptionWorker:
    def __init__(self, source: FrameSource, perceiver: Perceiver):
        self.source = source
        self.perceiver = perceiver
        self._latest: Perception | None = None
        self._cond = threading.Condition()
        self._running = False
        self._error: BaseException | None = None
        self._finished = False
        self._thread = threading.Thread(target=self._perceive_loop, name="perception", daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def _perceive_loop(self) -> None:
        last_id = 0
        try:
            while self._running:
                frame = self.source.read(last_id, timeout=0.5)
                if frame is None:
                    if self.source.ended:
                        break
                    continue
                last_id = frame.id
                result = self.perceiver.guard.call(
                    "perception",
                    self.perceiver,
                    frame,
                    fallback=Perception(frame, [], [], None, time.perf_counter()),
                )
                with self._cond:
                    self._latest = result
                    self._cond.notify_all()
        except BaseException as error:
            self._error = error
        finally:
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    def _has_newer(self, after_id: int) -> bool:
        return self._latest is not None and self._latest.frame.id > after_id

    def get(self, after_id: int, timeout: float = 1.0) -> Perception | None:
        with self._cond:
            self._cond.wait_for(lambda: self._finished or self._has_newer(after_id), timeout)
            if self._error is not None:
                raise RuntimeError("perception thread failed") from self._error
            return self._latest if self._has_newer(after_id) else None

    @property
    def finished(self) -> bool:
        return self._finished

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.perceiver.close()


def to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(lm.x, lm.y) for lm in landmarks], np.float32) * np.array([width, height], np.float32)


def convert_poses(result, width: int, height: int) -> list[PoseObservation]:
    return [
        PoseObservation(
            to_pixels(landmarks, width, height), np.array([lm.visibility for lm in landmarks], np.float32)
        )
        for landmarks in result.pose_landmarks
    ]


def convert_faces(result, width: int, height: int) -> list[FaceObservation]:
    faces = []
    for landmarks in result.face_landmarks:
        points = to_pixels(landmarks, width, height)
        faces.append(FaceObservation(points, np.concatenate([points.min(axis=0), points.max(axis=0)])))
    return faces


def hand_openness(world: np.ndarray) -> float:
    palm = float(np.linalg.norm(world[MIDDLE_KNUCKLE] - world[WRIST]))
    if palm < 1e-6:
        return 0.0
    return float(np.linalg.norm(world[FINGERTIPS] - world[WRIST], axis=1).mean()) / palm


def convert_hands(result, width: int, height: int, frame_id: int) -> list[HandObservation]:
    hands = []
    for landmarks, world_landmarks in zip(result.hand_landmarks, result.hand_world_landmarks, strict=True):
        world = np.array([(lm.x, lm.y, lm.z) for lm in world_landmarks], np.float32)
        hands.append(HandObservation(to_pixels(landmarks, width, height), hand_openness(world), frame_id))
    return hands


def merge_person_masks(result) -> np.ndarray | None:
    masks = result.segmentation_masks
    if not masks:
        return None
    merged = masks[0].numpy_view()
    for mask in masks[1:]:
        merged = np.maximum(merged, mask.numpy_view())
    return np.array(merged.squeeze(), dtype=np.float32)
