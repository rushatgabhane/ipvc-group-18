import numpy as np

from highnoon.config import TrackingConfig
from highnoon.contracts import FaceObservation, Frame, Perception, PoseObservation
from highnoon.tracking.identity import IdentityTracker

W, H = 1280, 720


def pose_at(x: float, y: float = 360.0) -> PoseObservation:
    kp = np.tile(np.array([x, y], np.float32), (33, 1))
    kp[0] = (x, y - 150)
    return PoseObservation(kp, np.ones(33, np.float32))


def face_at(x: float, y: float) -> FaceObservation:
    bbox = np.array([x - 40, y - 50, x + 40, y + 50], np.float32)
    return FaceObservation(np.zeros((478, 2), np.float32), bbox)


def perception(t: float, poses, faces=(), shirts=None) -> Perception:
    """Poses are drawn onto the frame in the given BGR shirt colours (default: all the same grey)."""
    image = np.full((H, W, 3), 40, np.uint8)
    for pose, color in zip(poses, shirts or [(120, 120, 120)] * len(poses), strict=False):
        x0, y0 = pose.keypoints[11].astype(int)
        x1, y1 = pose.keypoints[24].astype(int)
        image[max(0, y0) : y1, max(0, x0) : x1] = color
    frame = Frame(int(t * 30) + 1, image, t, int(t * 1000))
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


def test_identity_kept_through_short_loss():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    tr.update(perception(0.0, [pose_at(300), pose_at(900)]))
    tr.update(perception(0.5, [pose_at(300)]))
    players = tr.update(perception(0.8, [pose_at(300), pose_at(880)]))
    assert ids_by_x(players) == {1: 300, 2: 880}
    assert tr.reassigned == set()


RED, BLUE, GREEN = (0, 0, 220), (220, 0, 0), (0, 200, 0)


def wide_pose(x: float) -> PoseObservation:
    pose = pose_at(x)
    pose.keypoints[11], pose.keypoints[12] = (x - 60, 300), (x + 60, 300)
    pose.keypoints[23], pose.keypoints[24] = (x - 50, 500), (x + 50, 500)
    return pose


def enrolled_p2_then_left(tr):
    for i in range(10):
        tr.update(perception(i * 0.05, [wide_pose(300), wide_pose(900)], shirts=[RED, BLUE]))
    tr.lock()
    tr.update(perception(1.0, [wide_pose(300)], shirts=[RED]))
    tr.update(perception(5.0, [wide_pose(300)], shirts=[RED]))


def arrivals(tr, shirt, t0=6.0, x=100, seconds=0.6):
    players = []
    for i in range(int(seconds / 0.05) + 1):
        players = tr.update(perception(t0 + i * 0.05, [wide_pose(300), wide_pose(x)], shirts=[RED, shirt]))
    return players


def test_player_returning_from_the_other_side_keeps_their_id():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    enrolled_p2_then_left(tr)
    assert ids_by_x(arrivals(tr, BLUE, seconds=0.1)) == {1: 300}
    assert ids_by_x(arrivals(tr, BLUE)) == {1: 300, 2: 100}
    assert tr.reassigned == set()


def test_player_with_hips_out_of_view_is_still_recognised_by_the_upper_body():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    enrolled_p2_then_left(tr)
    players = []
    for i in range(8):
        arrival = wide_pose(100)
        arrival.visibility[[23, 24]] = 0.0
        players = tr.update(perception(6.0 + i * 0.05, [wide_pose(300), arrival], shirts=[RED, BLUE]))
    assert ids_by_x(players) == {1: 300, 2: 100}


def test_short_dropout_does_not_restart_the_confirmation():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    enrolled_p2_then_left(tr)
    tr.update(perception(6.0, [wide_pose(300), wide_pose(100)], shirts=[RED, BLUE]))
    tr.update(perception(6.1, [wide_pose(300)], shirts=[RED]))
    players = tr.update(perception(6.25, [wide_pose(300), wide_pose(100)], shirts=[RED, BLUE]))
    assert ids_by_x(players) == {1: 300, 2: 100}


def test_third_person_gets_no_id_while_a_player_is_away():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    enrolled_p2_then_left(tr)
    assert ids_by_x(arrivals(tr, GREEN)) == {1: 300}
    assert tr.reassigned == set()
    assert ids_by_x(arrivals(tr, BLUE, t0=8.0)) == {1: 300, 2: 100}


def test_lookalike_of_the_present_player_is_not_given_the_missing_slot():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    enrolled_p2_then_left(tr)
    assert ids_by_x(arrivals(tr, RED)) == {1: 300}


def test_leaving_before_the_lock_frees_the_slot():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    tr.update(perception(0.0, [wide_pose(300), wide_pose(900)], shirts=[RED, BLUE]))
    tr.update(perception(0.5, [wide_pose(300)], shirts=[RED]))
    tr.update(perception(2.0, [wide_pose(300)], shirts=[RED]))
    players = tr.update(perception(2.1, [wide_pose(300), wide_pose(700)], shirts=[RED, GREEN]))
    assert ids_by_x(players) == {1: 300, 2: 700}
    assert 2 in tr.reassigned


def test_new_players_are_flagged_once_per_session():
    tr = IdentityTracker(TrackingConfig())
    tr.update(perception(0.0, [pose_at(300), pose_at(900)]))
    assert tr.reassigned == {1, 2}
    tr.update(perception(0.1, [pose_at(300), pose_at(900)]))
    assert tr.reassigned == set()


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
