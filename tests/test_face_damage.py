import math

import numpy as np

from highnoon.contracts import FaceObservation
from highnoon.render.face_damage import UV, damage_stage, draw_face_damage

GRAY = 128


def make_face(scale: float = 200.0, squash_side: int | None = None, offset=(100, 100)) -> FaceObservation:
    """A face looking at the camera: the flat mesh layout itself, placed in the image."""
    points = (UV - 0.5).copy()
    if squash_side is not None:
        # Fold one half of the mesh over the other, as when that side is turned away from the camera.
        side = points[:, 0] < 0 if squash_side == 0 else points[:, 0] > 0
        points[side, 0] *= -0.5
    points = points * scale + np.array(offset) + scale / 2
    points = points.astype(np.float32)
    return FaceObservation(points, np.concatenate([points.min(axis=0), points.max(axis=0)]))


def blank():
    return np.full((400, 400, 3), GRAY, np.uint8)


def changed(before, after):
    return np.any(before != after, axis=2)


def darkness(pixels):
    return int((GRAY - pixels.astype(int)).clip(0).sum())


def draw(hp=1.0, age=math.inf, player=1, face=None):
    image = blank()
    draw_face_damage(image, face or make_face(), player, hp, age)
    return image


def test_stages_follow_hp():
    assert [damage_stage(hp) for hp in (1.0, 0.76, 0.75, 0.5, 0.3, 0.25, 0.0)] == [0, 0, 1, 2, 2, 3, 3]


def test_full_hp_and_no_recent_hit_leave_the_face_clean():
    assert np.all(draw(1.0) == GRAY)


def test_more_damage_marks_more_of_the_face():
    amounts = [darkness(draw(hp)) for hp in (0.75, 0.5, 0.25)]
    assert 0 < amounts[0] < amounts[1] < amounts[2]


def test_marks_stay_on_the_mesh():
    rows, cols = np.nonzero(changed(blank(), draw(0.2)))
    assert rows.min() >= 100 and rows.max() <= 300 and cols.min() >= 100 and cols.max() <= 300


def test_marks_blend_with_the_skin_instead_of_covering_it():
    # Multiplying keeps the camera's detail: a brighter pixel stays brighter under the same mark.
    bright, dark = np.full((400, 400, 3), 200, np.uint8), np.full((400, 400, 3), 100, np.uint8)
    draw_face_damage(bright, make_face(), 1, 0.5, math.inf)
    draw_face_damage(dark, make_face(), 1, 0.5, math.inf)
    marked = changed(np.full((400, 400, 3), 200, np.uint8), bright)
    assert marked.any()
    assert np.all(bright[marked].astype(int) >= dark[marked])
    assert np.abs(bright[marked] / 200.0 - dark[marked] / 100.0).max() < 0.03


def test_a_fresh_hit_reddens_the_face_and_fades():
    fresh = draw(1.0, 0.0)
    assert changed(blank(), fresh).any()
    blue, red = fresh[:, :, 0], fresh[:, :, 2]
    assert blue.min() < GRAY and red.min() == GRAY
    assert np.all(draw(1.0, 1.0) == GRAY)


def test_the_marks_follow_the_face():
    moved = make_face(offset=(160, 130))
    still, shifted = draw(0.25), draw(0.25, face=moved)
    assert np.array_equal(still[100:300, 100:240], shifted[130:330, 160:300])


def test_the_side_turned_away_is_not_painted():
    # On the folded side only the triangles still facing the camera are drawn.
    facing = changed(blank(), draw(0.5))
    folded = changed(blank(), draw(0.5, face=make_face(squash_side=1)))
    far_cheek = (slice(150, 260), slice(240, 300))
    assert facing[far_cheek].any()
    assert not folded[far_cheek].any()


def test_players_get_the_bruise_on_opposite_sides():
    one, two = draw(0.5, player=1), draw(0.5, player=2)
    # The bruise is the first mark on the hurt side; Player 2 gets the same marks mirrored.
    under_left_eye, under_right_eye = (slice(165, 190), slice(150, 190)), (slice(165, 190), slice(210, 250))
    assert darkness(one[under_left_eye]) > 2 * darkness(one[under_right_eye])
    assert darkness(two[under_right_eye]) > 2 * darkness(two[under_left_eye])


def test_tiny_faces_are_skipped():
    assert np.all(draw(0.0, 0.0, face=make_face(scale=20.0)) == GRAY)
