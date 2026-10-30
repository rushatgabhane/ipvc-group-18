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


def perception(t: float, poses, faces=()) -> Perception:
    image = np.full((H, W, 3), 40, np.uint8)
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


def wide_pose(x: float) -> PoseObservation:
    pose = pose_at(x)
    pose.keypoints[11], pose.keypoints[12] = (x - 60, 300), (x + 60, 300)
    pose.keypoints[23], pose.keypoints[24] = (x - 50, 500), (x + 50, 500)
    return pose


def test_leaving_before_the_lock_frees_the_slot():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    tr.update(perception(0.0, [wide_pose(300), wide_pose(900)]))
    tr.update(perception(0.5, [wide_pose(300)]))
    tr.update(perception(2.0, [wide_pose(300)]))
    players = tr.update(perception(2.1, [wide_pose(300), wide_pose(700)]))
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


rng = np.random.default_rng(7)
ALICE, BOB, STRANGER = (v / np.linalg.norm(v) for v in rng.normal(size=(3, 128)))


def seen_face(identity: np.ndarray, x: float) -> FaceObservation:
    """A face over a wide_pose's nose whose embedding is `identity` plus a little per-frame noise."""
    face = face_at(x, 210)
    noisy = identity + rng.normal(scale=0.03, size=identity.shape)
    face.embedding = noisy / np.linalg.norm(noisy)
    return face


def scene(t: float, people) -> Perception:
    """`people` is a list of (x, face identity or None for a face that cannot be seen)."""
    poses = [wide_pose(x) for x, _ in people]
    faces = [seen_face(identity, x) for x, identity in people if identity is not None]
    return perception(t, poses, faces)


def enrolled_faces_p2_then_left():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    for i in range(10):
        tr.update(scene(i * 0.05, [(300, ALICE), (900, BOB)]))
    tr.lock()
    tr.update(scene(1.0, [(300, ALICE)]))
    tr.update(scene(5.0, [(300, ALICE)]))
    return tr


def play(tr, people, t0, seconds=0.6):
    players = []
    for i in range(int(seconds / 0.05) + 1):
        players = tr.update(scene(t0 + i * 0.05, people))
    return players


def test_stranger_gets_no_id_while_a_player_is_away():
    tr = enrolled_faces_p2_then_left()
    assert ids_by_x(play(tr, [(300, ALICE), (100, STRANGER)], 6.0, seconds=2.0)) == {1: 300}
    assert tr.reassigned == set()


def test_returning_player_gets_their_id_back_once_confirmed():
    tr = enrolled_faces_p2_then_left()
    assert ids_by_x(play(tr, [(300, ALICE), (100, BOB)], 6.0, seconds=0.05)) == {1: 300}
    assert ids_by_x(play(tr, [(300, ALICE), (100, BOB)], 6.1)) == {1: 300, 2: 100}
    assert tr.reassigned == set()


def test_returning_player_waits_without_an_id_until_their_face_is_seen():
    tr = enrolled_faces_p2_then_left()
    assert ids_by_x(play(tr, [(300, ALICE), (100, None)], 6.0, seconds=3.0)) == {1: 300}
    assert ids_by_x(play(tr, [(300, ALICE), (100, BOB)], 9.05)) == {1: 300, 2: 100}


def test_lookalike_of_the_present_player_is_not_given_the_missing_slot():
    tr = enrolled_faces_p2_then_left()
    assert ids_by_x(play(tr, [(300, ALICE), (100, ALICE)], 6.0, seconds=2.0)) == {1: 300}


def test_third_person_present_when_the_player_returns_does_not_take_the_id():
    tr = enrolled_faces_p2_then_left()
    play(tr, [(300, ALICE), (1000, STRANGER)], 6.0, seconds=1.0)
    players = play(tr, [(300, ALICE), (1000, STRANGER), (650, BOB)], 7.1)
    assert ids_by_x(players) == {1: 300, 2: 650}


def test_stranger_stepping_into_a_players_spot_is_released():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    for i in range(10):
        tr.update(scene(i * 0.05, [(300, ALICE), (900, BOB)]))
    tr.lock()
    tr.update(scene(1.0, [(300, ALICE)]))
    players = play(tr, [(300, ALICE), (920, STRANGER)], 1.2)
    assert ids_by_x(players) == {1: 300}
    assert ids_by_x(play(tr, [(300, ALICE), (920, STRANGER), (600, BOB)], 2.0)) == {
        1: 300,
        2: 600,
    }


def test_ids_swapped_by_position_are_corrected_by_face():
    tr = IdentityTracker(TrackingConfig(lost_timeout_s=1.0))
    for i in range(10):
        tr.update(scene(i * 0.05, [(300, ALICE), (900, BOB)]))
    tr.lock()
    players = play(tr, [(300, BOB), (900, ALICE)], 1.0, seconds=1.0)
    assert ids_by_x(players) == {1: 900, 2: 300}
