import numpy as np

from highnoon.config import FilterConfig
from highnoon.contracts import Player, PoseObservation
from highnoon.vision.motion import MotionFilter


def body(vis_shoulders=True, vis_hips=True):
    kp = np.zeros((33, 2), np.float32)
    kp[11], kp[12] = (560, 400), (720, 400)  # shoulders, 160 px apart
    kp[23], kp[24] = (580, 600), (700, 600)  # hips, torso 200 px
    kp[13], kp[15] = (500, 520), (480, 640)  # left arm hanging
    kp[14], kp[16] = (780, 450), (840, 370)  # right arm bent, forearm raised ~53 deg
    vis = np.ones(33, np.float32)
    if not vis_shoulders:
        vis[[11, 12]] = 0.1
    if not vis_hips:
        vis[[23, 24]] = 0.1
    return kp, vis


def signals(kp, vis):
    mf = MotionFilter(FilterConfig())
    p = Player(id=1, pose=PoseObservation(kp, vis, np.zeros((33, 3), np.float32)), visible=True)
    mf.update([p], 0.0, set())
    return p.signals


def test_bent_arm_aims_along_forearm():
    s = signals(*body())
    assert s.aim_valid
    expected = np.array([60, -80], np.float32) / 100
    np.testing.assert_allclose(s.aim_dir, expected, atol=1e-5)


def test_aims_when_shoulders_and_hips_out_of_frame():
    s = signals(*body(vis_shoulders=False, vis_hips=False))
    assert s.aim_valid
    assert np.degrees(s.elevation) > 45


def test_scale_from_shoulders_when_hips_hidden():
    s = signals(*body(vis_hips=False))
    assert abs(s.torso_len - 160 / 0.8) < 1e-3


def test_forearm_pointing_at_camera_gives_no_aim():
    kp, vis = body()
    kp[16] = kp[14] + (10, -5)  # forearm foreshortened to ~11 px
    kp[15] = kp[13] + (0, 10)
    assert not signals(kp, vis).aim_valid


def test_aim_smoothing_reduces_jitter_and_follows_swings():
    rng = np.random.default_rng(0)
    mf = MotionFilter(FilterConfig())
    angles = []
    for i in range(90):
        kp, vis = body()
        kp[16] = kp[14] + (60, -80) + rng.normal(0, 4, 2)  # jittery forearm
        p = Player(id=1, pose=PoseObservation(kp, vis, np.zeros((33, 3), np.float32)), visible=True)
        mf.update([p], i / 30, set())
        angles.append(np.arctan2(p.signals.aim_dir[1], p.signals.aim_dir[0]))
    raw_jitter = np.degrees(np.std([np.arctan2(-80 + d[1], 60 + d[0]) for d in rng.normal(0, 4, (60, 2))]))
    assert np.degrees(np.std(angles[30:])) < raw_jitter * 0.5

    for i in range(90, 110):  # swing the forearm to point sideways: the aim must follow
        kp, vis = body()
        kp[16] = kp[14] + (60, 0)  # forearm horizontal (arm stays bent, so the forearm ray is used)
        p = Player(id=1, pose=PoseObservation(kp, vis, np.zeros((33, 3), np.float32)), visible=True)
        mf.update([p], i / 30, set())
    assert abs(np.degrees(np.arctan2(p.signals.aim_dir[1], p.signals.aim_dir[0]))) < 5
