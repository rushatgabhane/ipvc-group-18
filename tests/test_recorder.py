import time

import cv2
import numpy as np

from highnoon.core.recorder import SessionRecorder, frames_due


def test_frames_due_keeps_video_time_with_wall_time():
    assert frames_due(0.0, 30) == 1
    assert frames_due(1 / 30, 30) == 2
    assert frames_due(1.0, 30) == 31


def test_a_slow_session_plays_back_at_real_speed(tmp_path):
    path = tmp_path / "slow.mp4"
    recorder = SessionRecorder(path, 30, unmirror=False)
    for i in range(48):  # 2 s shown at 24 fps
        recorder.add(np.full((64, 64, 3), i, np.uint8), i / 24)
        time.sleep(0.005)  # frames arrive over time in a real session, not all at once
    recorder.close()
    video = cv2.VideoCapture(str(path))
    frames, fps = video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FPS)
    assert fps == 30 and abs(frames / fps - 47 / 24) < 0.05
