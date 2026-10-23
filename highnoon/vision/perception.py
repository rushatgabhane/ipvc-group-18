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
from highnoon.vision.delegate import resolve_delegate

FINGERTIPS = [8, 12, 16, 20]
WRIST, MIDDLE_KNUCKLE = 0, 9


class Perceiver:
    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        for path in (cfg.pose_model_path, cfg.face_model_path, cfg.hand_model_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run: python tools/download_models.py")
        delegate = resolve_delegate(cfg)
        self.pose = create_pose_landmarker(cfg, delegate)
        self.face = create_face_landmarker(cfg, delegate)
        self.hands = HandWorker(create_hand_landmarker(cfg, delegate))
        self._pose_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")

    def __call__(self, frame: Frame) -> Perception:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        timings["convert"] = elapsed_ms(t0)
        self.hands.submit(frame, image)

        def detect(name, landmarker):
            started = time.perf_counter()
            result = landmarker.detect_for_video(image, frame.timestamp_ms)
            timings[name] = elapsed_ms(started)
            return result

        t_infer = time.perf_counter()
        pose_future = self._pose_thread.submit(detect, "pose", self.pose)
        face_result = detect("face", self.face)
        pose_result = pose_future.result()
        timings["inference"] = elapsed_ms(t_infer)

        t_post = time.perf_counter()
        width, height = frame.size
        poses = convert_poses(pose_result, width, height)
        faces = convert_faces(face_result, width, height)
        person_masks = [
            np.array(m.numpy_view().squeeze(), np.float32) for m in pose_result.segmentation_masks or []
        ]
        mask = np.maximum.reduce(person_masks) if person_masks else None
        hands, timings["hand"] = self.hands.latest()
        timings["hand_lag"] = float(frame.id - hands[0].frame_id) if hands else 0.0
        timings["postprocess"] = elapsed_ms(t_post)
        return Perception(frame, poses, faces, mask, time.perf_counter(), timings, hands, person_masks)

    def close(self) -> None:
        self._pose_thread.shutdown(wait=True)
        self.hands.close()
        self.pose.close()
        self.face.close()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def base_options(path, delegate) -> BaseOptions:
    return BaseOptions(model_asset_path=str(path), delegate=delegate)


def create_pose_landmarker(cfg: PerceptionConfig, delegate):
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options(cfg.pose_model_path, delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg.max_players,
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
            num_faces=cfg.max_players,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
    )


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
    def __init__(self, landmarker):
        self.landmarker = landmarker
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
            result = self.landmarker.detect_for_video(image, frame.timestamp_ms)
            hands = convert_hands(result, *frame.size, frame.id)
            with self._cond:
                self._result = hands
                self._ms = elapsed_ms(started)

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
                result = self.perceiver(frame)
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
