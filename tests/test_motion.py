import numpy as np

from highnoon.config import FilterConfig
from highnoon.contracts import HandObservation, Player, PoseObservation
from highnoon.vision.motion import HAND_HOLD_S, MotionFilter

L_WRIST, R_WRIST = 15, 16


def body(vis_shoulders=True, vis_hips=True):
    kp = np.zeros((33, 2), np.float32)
    kp[11], kp[12] = (560, 400), (720, 400)
    kp[23], kp[24] = (580, 600), (700, 600)
    kp[13], kp[15] = (500, 520), (480, 640)
    kp[14], kp[16] = (780, 450), (840, 370)
    vis = np.ones(33, np.float32)
    if not vis_shoulders:
        vis[[11, 12]] = 0.1
    if not vis_hips:
        vis[[23, 24]] = 0.1
    return kp, vis


def hand(pointer=(60, -80), wrist=(840, 370), openness=2.0, fingers_curled=False):
    """Knuckles spread across the end of the palm, fingertips beyond them or folded back."""
    wrist = np.array(wrist, np.float32)
    pointer = np.array(pointer, np.float32)
    across = np.array([-pointer[1], pointer[0]], np.float32) * 0.3
    points = np.tile(wrist, (21, 1))
    for offset, knuckle in zip((1.5, 0.5, -0.5, -1.5), (5, 9, 13, 17), strict=True):
        points[knuckle] = wrist + pointer + across * offset
        tip = wrist + pointer * (0.6 if fingers_curled else 1.8) + across * offset
        points[knuckle + 3] = tip
    return HandObservation(points, openness, 0)


def player(hands=None, vis_shoulders=True, vis_hips=True, kp=None):
    keypoints, vis = body(vis_shoulders, vis_hips)
    pose = PoseObservation(kp if kp is not None else keypoints, vis)
    return Player(id=1, pose=pose, hands=hands or {}, visible=True)


def signals(p):
    MotionFilter(FilterConfig()).update([p], 0.0, set())
    return p.signals


def angle(signal):
    return np.degrees(np.arctan2(signal.aim_dir[1], signal.aim_dir[0]))


def test_aims_where_the_palm_points():
    s = signals(player({R_WRIST: hand()}))
    assert s.aim_valid
    np.testing.assert_allclose(s.aim_dir, np.array([60, -80], np.float32) / 100, atol=1e-5)


def test_the_arm_does_not_change_the_aim():
    kp, _ = body()
    kp[14] = (700, 200)  # the elbow somewhere else entirely
    s = signals(player({R_WRIST: hand(pointer=(90, 0))}, kp=kp))
    np.testing.assert_allclose(s.aim_dir, (1, 0), atol=1e-5)


def test_closing_the_fist_keeps_the_aim():
    still_open = signals(player({R_WRIST: hand()}))
    fist = signals(player({R_WRIST: hand(fingers_curled=True, openness=1.0)}))
    np.testing.assert_allclose(fist.aim_dir, still_open.aim_dir, atol=1e-5)


def test_no_hand_means_no_aim():
    assert not signals(player()).aim_valid


def test_aims_when_shoulders_and_hips_out_of_frame():
    s = signals(player({R_WRIST: hand()}, vis_shoulders=False, vis_hips=False))
    assert s.aim_valid
    assert np.degrees(s.elevation) > 45


def test_scale_from_shoulders_when_hips_hidden():
    s = signals(player(vis_hips=False))
    assert abs(s.torso_len - 160 / 0.8) < 1e-3


def test_palm_pointing_at_the_camera_gives_no_aim():
    assert not signals(player({R_WRIST: hand(pointer=(4, -3))})).aim_valid


def test_hand_hanging_down_gives_no_aim():
    assert not signals(player({R_WRIST: hand(pointer=(5, 90))})).aim_valid


def test_left_hand_takes_over_when_the_right_cannot_aim():
    s = signals(player({L_WRIST: hand(pointer=(-90, 0), wrist=(480, 640))}))
    assert s.aim_valid
    np.testing.assert_allclose(s.aim_dir, (-1, 0), atol=1e-5)
    np.testing.assert_allclose(s.aim_origin, (480, 640))


def test_a_brief_tracking_dropout_holds_the_aim_then_stops():
    mf = MotionFilter(FilterConfig())
    mf.update([player({R_WRIST: hand(pointer=(90, 0))})], 0.0, set())

    during = player()
    mf.update([during], HAND_HOLD_S - 0.05, set())
    assert during.signals.aim_valid
    assert abs(angle(during.signals)) < 1

    after = player()
    mf.update([after], HAND_HOLD_S + 0.05, set())
    assert not after.signals.aim_valid


def test_aim_smoothing_reduces_jitter_and_follows_swings():
    rng = np.random.default_rng(0)
    mf = MotionFilter(FilterConfig())
    angles = []
    raw = []
    for i in range(90):
        observed = hand()
        observed.landmarks += rng.normal(0, 2, observed.landmarks.shape).astype(np.float32)
        p = player({R_WRIST: observed})
        pointer = observed.landmarks[[5, 9, 13, 17]].mean(axis=0) - observed.landmarks[0]
        raw.append(np.degrees(np.arctan2(pointer[1], pointer[0])))
        mf.update([p], i / 30, set())
        angles.append(angle(p.signals))
    assert np.std(angles[30:]) < np.std(raw[30:]) * 0.5

    for i in range(90, 110):
        p = player({R_WRIST: hand(pointer=(90, 0))})
        mf.update([p], i / 30, set())
    assert abs(angle(p.signals)) < 5
