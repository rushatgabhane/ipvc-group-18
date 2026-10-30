import math

import numpy as np

from highnoon.contracts import FaceObservation
from highnoon.render.face_damage import damage_stage, draw_face_damage

# The landmarks the damage marks use, for a 100 px face looking at the camera.
KEY_POINTS = {
    33: (72.2, 83.4),
    133: (88.4, 83.8),
    145: (80.1, 85.7),
    50: (70.2, 104.4),
    263: (127.8, 83.4),
    362: (111.6, 83.8),
    374: (119.9, 85.7),
    280: (129.8, 104.4),
    6: (100.0, 84.5),
    1: (100.0, 107.0),
}


def make_face(squash_side: int | None = None, scale: float = 1.0) -> FaceObservation:
    angles = np.linspace(0, math.tau, 478, endpoint=False)
    points = np.stack([100 + 45 * np.cos(angles), 100 + 55 * np.sin(angles)], axis=1).astype(np.float32)
    for index, xy in KEY_POINTS.items():
        points[index] = xy
    if squash_side is not None:
        outer, inner = (33, 133) if squash_side == 0 else (263, 362)
        middle = (points[outer] + points[inner]) / 2
        points[[outer, inner]] = middle + (points[[outer, inner]] - middle) * 0.2
    points = 100 + (points - 100) * scale
    return FaceObservation(points, np.concatenate([points.min(axis=0), points.max(axis=0)]))


def blank():
    return np.full((200, 200, 3), 128, np.uint8)


def changed(before, after):
    return np.any(before != after, axis=2)


def test_stages_follow_hp():
    assert [damage_stage(hp) for hp in (1.0, 0.76, 0.75, 0.5, 0.3, 0.25, 0.0)] == [0, 0, 1, 2, 2, 3, 3]


def test_full_hp_and_no_recent_hit_leave_the_face_clean():
    image = blank()
    draw_face_damage(image, make_face(), 1, 1.0, math.inf)
    assert np.all(image == 128)


def test_more_damage_marks_more_of_the_face():
    areas = []
    for hp in (0.75, 0.5, 0.25):
        image = blank()
        draw_face_damage(image, make_face(), 1, hp, math.inf)
        areas.append(int(changed(blank(), image).sum()))
    assert 0 < areas[0] < areas[1] < areas[2]


def test_marks_stay_near_the_face():
    image = blank()
    draw_face_damage(image, make_face(), 1, 0.2, math.inf)
    rows, cols = np.nonzero(changed(blank(), image))
    assert rows.min() >= 30 and rows.max() <= 170 and cols.min() >= 40 and cols.max() <= 160


def test_a_fresh_hit_flashes_the_face_and_fades():
    fresh, old = blank(), blank()
    draw_face_damage(fresh, make_face(), 1, 1.0, 0.0)
    draw_face_damage(old, make_face(), 1, 1.0, 1.0)
    assert changed(blank(), fresh).any()
    assert np.all(old == 128)


def test_the_side_turned_away_hides_its_marks():
    facing, turned = blank(), blank()
    draw_face_damage(facing, make_face(), 1, 0.75, math.inf)
    draw_face_damage(turned, make_face(squash_side=1), 1, 0.75, math.inf)
    right_cheek = (slice(90, 120), slice(115, 160))
    assert changed(blank(), facing)[right_cheek].any()
    assert not changed(blank(), turned)[right_cheek].any()


def test_players_get_the_bruise_under_opposite_eyes():
    one, two = blank(), blank()
    draw_face_damage(one, make_face(), 1, 0.5, math.inf)
    draw_face_damage(two, make_face(), 2, 0.5, math.inf)
    under_left_eye, under_right_eye = (slice(86, 92), slice(70, 91)), (slice(86, 92), slice(109, 130))
    assert changed(blank(), one)[under_left_eye].any() and not changed(blank(), one)[under_right_eye].any()
    assert changed(blank(), two)[under_right_eye].any() and not changed(blank(), two)[under_left_eye].any()


def test_tiny_faces_are_skipped():
    image = blank()
    draw_face_damage(image, make_face(scale=0.2), 1, 0.0, 0.0)
    assert np.all(image == 128)
