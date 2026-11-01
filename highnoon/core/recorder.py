from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

QUEUE_FRAMES = 30


class SessionRecorder:
    """Writes frames to an MP4 that plays back at real speed.

    The game often runs below the file's frame rate (about 24 fps with two players), so each frame is
    written as many times as needed for the video time to keep up with the wall-clock time it was shown
    at. Without that a 87 s session played back in 69 s, which is a speed change the peer video may not
    have, and it also shortened every time measured on a replayed clip.
    """

    def __init__(self, path: Path, fps: int, unmirror: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.unmirror = unmirror
        self.dropped = 0
        self.written = 0
        self._t0: float | None = None
        self._frames: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._writer: cv2.VideoWriter | None = None
        self._thread = threading.Thread(target=self._write_loop, name="recorder", daemon=True)
        self._thread.start()

    def add(self, image: np.ndarray, t: float) -> None:
        """`t`: when the frame was captured or shown, in seconds (any clock, the same for every frame)."""
        try:
            self._frames.put_nowait((image, t))
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        while (item := self._frames.get()) is not None:
            image, t = item
            if self._writer is None:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                self._t0 = t
            due = frames_due(t - self._t0, self.fps) - self.written
            if due <= 0:  # shown faster than the file's frame rate: this one would not be seen anyway
                continue
            image = cv2.flip(image, 1) if self.unmirror else image
            for _ in range(due):
                self._writer.write(image)
            self.written += due

    def close(self) -> None:
        self._frames.put(None)
        self._thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()


def frames_due(elapsed_s: float, fps: int) -> int:
    """How many video frames should exist once a frame shown `elapsed_s` after the first one is written."""
    return int(elapsed_s * fps + 1e-6) + 1
