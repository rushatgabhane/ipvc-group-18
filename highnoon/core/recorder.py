from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

QUEUE_FRAMES = 30


class SessionRecorder:
    def __init__(self, path: Path, fps: int, unmirror: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.unmirror = unmirror
        self.dropped = 0
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._writer: cv2.VideoWriter | None = None
        self._thread = threading.Thread(target=self._write_loop, name="recorder", daemon=True)
        self._thread.start()

    def add(self, image: np.ndarray) -> None:
        try:
            self._frames.put_nowait(image)
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        while (image := self._frames.get()) is not None:
            if self._writer is None:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            self._writer.write(cv2.flip(image, 1) if self.unmirror else image)

    def close(self) -> None:
        self._frames.put(None)
        self._thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()
