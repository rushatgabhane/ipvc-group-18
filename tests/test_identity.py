import numpy as np

from highnoon.config import TrackingConfig
from highnoon.contracts import FaceObservation, Frame, Perception, PoseObservation
from highnoon.tracking.identity import IdentityTracker

W, H = 1280, 720


def pose_at(x: float, y: float = 360.0) -> PoseObservation:
    kp = np.tile(np.array([x, y], np.float32), (33, 1))
    kp[0] = (x, y - 150)  # nose above the torso
    return PoseObservation(kp, np.ones(33, np.float32), np.zeros((33, 3), np.float32))


def face_at(x: float, y: float) -> FaceObservation:
    bbox = np.array([x - 40, y - 50, x + 40, y + 50], np.float32)
    return FaceObservation(np.zeros((478, 2), np.float32), bbox)


def perception(t: float, poses, faces=()) -> Perception:
    frame = Frame(int(t * 30) + 1, np.zeros((H, W, 3), np.uint8), t, int(t * 1000))
    return Perception(frame, list(poses), list(faces), None, t)


def ids_by_x(players):
    return {p.id: round(float(p.pose.center()[0])) for p in players if p.visible}


def test_new_players_numbered_left_to_right():
    tr = IdentityTracker(TrackingConfig())
    players = tr.update(perception(0.0, [pose_at(900), pose_at(300)]))
    assert ids_by_x(players) == {1: 300, 2: 900}


def test_identity_follows_motion_regardless_of_detection_order():
    tr = IdentityTracker(TrackingConfig())
    tr.update(perception(0.0, [pose_at(300), pose_at(900)]))
    players = tr.update(perception(0.033, [pose_at(920), pose_at(320)]))
    assert ids_by_x(players) == {1: 320, 2: 920}


def test_identity_kept_through_short_loss_and_released_after_timeout():
    cfg = TrackingConfig(lost_timeout_s=1.0)
    tr = IdentityTracker(cfg)
    tr.update(perception(0.0, [pose_at(300), pose_at(900)]))
    tr.update(perception(0.5, [pose_at(300)]))  # player 2 occluded
    players = tr.update(perception(0.8, [pose_at(300), pose_at(880)]))
    assert ids_by_x(players) == {1: 300, 2: 880}
    assert tr.reassigned == set()

    tr.update(perception(1.0, [pose_at(300)]))
    tr.update(perception(2.5, [pose_at(300)]))  # player 2 gone > timeout -> slot released
    tr.update(perception(2.6, [pose_at(300), pose_at(700)]))
    assert 2 in tr.reassigned


def test_faces_assigned_to_nearest_nose():
    tr = IdentityTracker(TrackingConfig())
    players = tr.update(perception(0.0, [pose_at(300), pose_at(900)], [face_at(905, 210), face_at(298, 212)]))
    p1, p2 = players
    assert p1.face.center()[0] == 298
    assert p2.face.center()[0] == 905


def test_far_face_not_assigned():
    tr = IdentityTracker(TrackingConfig())
    players = tr.update(perception(0.0, [pose_at(300)], [face_at(1100, 600)]))
    assert players[0].face is None
