from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from highnoon.config import CaptureConfig
from highnoon.contracts import Frame


class FrameSource:
    def start(self) -> None: ...

    def read(self, after_id: int, timeout: float = 1.0) -> Frame | None:
        raise NotImplementedError

    @property
    def ended(self) -> bool:
        return False

    def stop(self) -> None: ...


def open_source(cfg: CaptureConfig) -> FrameSource:
    if cfg.source.isdigit():
        return CameraSource(cfg)
    if not Path(cfg.source).exists():
        raise FileNotFoundError(cfg.source)
    return VideoFileSource(cfg)


class CameraSource(FrameSource):
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(int(cfg.source))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cfg.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._latest: Frame | None = None
        self._cond = threading.Condition()
        self._running = False
        self._failed = False
        self._thread = threading.Thread(target=self._capture_loop, name="capture", daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def _capture_loop(self) -> None:
        frame_id = 0
        last_timestamp = -1
        while self._running:
            ok, image = self.cap.read()
            t = time.perf_counter()
            if not ok:
                with self._cond:
                    self._failed = True
                    self._cond.notify_all()
                return
            if self.cfg.mirror:
                image = cv2.flip(image, 1)
            frame_id += 1
            last_timestamp = max(int(t * 1000), last_timestamp + 1)
            with self._cond:
                self._latest = Frame(frame_id, image, t, last_timestamp)
                self._cond.notify_all()

    def _has_newer(self, after_id: int) -> bool:
        return self._latest is not None and self._latest.id > after_id

    def read(self, after_id: int, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            self._cond.wait_for(lambda: self._failed or self._has_newer(after_id), timeout)
            return self._latest if self._has_newer(after_id) else None

    @property
    def ended(self) -> bool:
        return self._failed

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


class VideoFileSource(FrameSource):
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {cfg.source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0
        self._ended = False
        self._lock = threading.Lock()
        self._t_start = 0.0

    def start(self) -> None:
        self._t_start = time.perf_counter()

    def read(self, after_id: int, timeout: float = 1.0) -> Frame | None:
        with self._lock:
            ok, image = self._read_next()
            if not ok:
                self._ended = True
                return None
            self._index += 1
            if self.cfg.realtime:
                self._wait_until_due()
            if self.cfg.mirror:
                image = cv2.flip(image, 1)
            timestamp_ms = int(self._index * 1000 / self.fps)
            return Frame(self._index, image, time.perf_counter(), timestamp_ms)

    def _read_next(self):
        ok, image = self.cap.read()
        if not ok and self.cfg.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self.cap.read()
        return ok, image

    def _wait_until_due(self) -> None:
        delay = self._t_start + self._index / self.fps - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    @property
    def ended(self) -> bool:
        return self._ended

    def stop(self) -> None:
        self.cap.release()
