"""Model inference (T1 face + T2 pose), run on a dedicated thread.

Performance design (numbers from an M1 Pro, CPU; see docs/PERFORMANCE.md):
  * Pose (25 ms) and face (11 ms) run in parallel on two threads. MediaPipe releases the
    GIL inside its C++ graph, so this measured 36.7 ms -> 25.7 ms.
  * The whole worker runs beside the main thread. While frame N is rendered, frame N+1
    is already being inferred, so throughput is set by the slowest stage, not their sum.
  * Full resolution is passed in. Downscaling to 0.5x gave no speedup (the models resize
    internally) and would only hurt small faces far from the camera.
  * The person segmentation mask comes out of the pose graph for ~0.3 ms extra, so no
    separate segmentation model is needed for background replacement (T5).
  * The GPU delegate aborts the process on macOS (mediapipe 0.10.35 / 1.0.1), so CPU only.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from highnoon.config import PerceptionConfig
from highnoon.contracts import FaceObservation, Frame, Perception, PoseObservation
from highnoon.core.capture import FrameSource


class Perceiver:
    """Synchronous inference on one frame. Owns the MediaPipe graphs."""

    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        for path in [cfg.pose_model_path] + ([cfg.face_model_path] if cfg.run_face else []):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run: python tools/download_models.py")

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(cfg.pose_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=cfg.max_players,
                min_pose_detection_confidence=cfg.min_detection_confidence,
                min_pose_presence_confidence=cfg.min_presence_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                output_segmentation_masks=cfg.segmentation,
            )
        )
        self.face = None
        if cfg.run_face:
            self.face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(cfg.face_model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=cfg.max_players,
                    min_face_detection_confidence=cfg.min_detection_confidence,
                    min_face_presence_confidence=cfg.min_presence_confidence,
                    min_tracking_confidence=cfg.min_tracking_confidence,
                )
            )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer") if cfg.parallel else None

    def __call__(self, frame: Frame) -> Perception:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timings["convert"] = (time.perf_counter() - t0) * 1000

        def timed(name, fn):
            t = time.perf_counter()
            result = fn(image, frame.timestamp_ms)
            timings[name] = (time.perf_counter() - t) * 1000
            return result

        t_infer = time.perf_counter()
        if self._pool is not None and self.face is not None:
            pose_future = self._pool.submit(timed, "pose", self.pose.detect_for_video)
            face_result = timed("face", self.face.detect_for_video)
            pose_result = pose_future.result()
        else:
            pose_result = timed("pose", self.pose.detect_for_video)
            face_result = timed("face", self.face.detect_for_video) if self.face is not None else None
        timings["inference"] = (time.perf_counter() - t_infer) * 1000

        t_post = time.perf_counter()
        w, h = frame.size
        poses = _convert_poses(pose_result, w, h)
        faces = _convert_faces(face_result, w, h) if face_result is not None else []
        mask = _merge_masks(pose_result) if self.cfg.segmentation else None
        timings["postprocess"] = (time.perf_counter() - t_post) * 1000
        return Perception(frame, poses, faces, mask, time.perf_counter(), timings)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self.pose.close()
        if self.face is not None:
            self.face.close()


class PerceptionWorker:
    """Runs a Perceiver on its own thread, always on the newest available frame."""

    def __init__(self, source: FrameSource, perceiver: Perceiver):
        self.source = source
        self.perceiver = perceiver
        self._latest: Perception | None = None
        self._cond = threading.Condition()
        self._running = False
        self._error: BaseException | None = None
        self._finished = False
        self._thread = threading.Thread(target=self._loop, name="perception", daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def _loop(self) -> None:
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
        except BaseException as e:  # surface model errors on the main thread
            self._error = e
        finally:
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    def get(self, after_id: int, timeout: float = 1.0) -> Perception | None:
        """Newest result for a frame with id > after_id. None on timeout or when finished."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._finished or (self._latest is not None and self._latest.frame.id > after_id),
                timeout,
            )
            if self._error is not None:
                raise RuntimeError("perception thread failed") from self._error
            if self._latest is not None and self._latest.frame.id > after_id:
                return self._latest
            return None

    @property
    def finished(self) -> bool:
        return self._finished

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.perceiver.close()


def _convert_poses(result, w: int, h: int) -> list[PoseObservation]:
    poses = []
    scale = np.array([w, h], np.float32)
    for i, landmarks in enumerate(result.pose_landmarks):
        arr = np.array([(lm.x, lm.y, lm.visibility) for lm in landmarks], np.float32)
        world = np.array([(lm.x, lm.y, lm.z) for lm in result.pose_world_landmarks[i]], np.float32)
        poses.append(PoseObservation(arr[:, :2] * scale, arr[:, 2], world))
    return poses


def _convert_faces(result, w: int, h: int) -> list[FaceObservation]:
    faces = []
    scale = np.array([w, h], np.float32)
    for landmarks in result.face_landmarks:
        pts = np.array([(lm.x, lm.y) for lm in landmarks], np.float32) * scale
        bbox = np.concatenate([pts.min(axis=0), pts.max(axis=0)])
        faces.append(FaceObservation(pts, bbox))
    return faces


def _merge_masks(result) -> np.ndarray | None:
    masks = result.segmentation_masks
    if not masks:
        return None
    merged = masks[0].numpy_view()
    for m in masks[1:]:
        merged = np.maximum(merged, m.numpy_view())
    # Copy: the view points into MediaPipe-owned memory that is freed with the result.
    return np.array(merged.squeeze(), dtype=np.float32)
