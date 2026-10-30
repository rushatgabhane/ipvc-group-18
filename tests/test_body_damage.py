import math

import numpy as np

from highnoon.contracts import HandObservation, PoseObservation, Player
from highnoon.render.body_damage import draw_body_damage, player_columns


def make_player(pid: int = 1, cx: float = 200.0, with_hand: bool = False) -> Player:
    keypoints = np.zeros((33, 2), np.float32)
    keypoints[11], keypoints[12] = (cx - 50, 100), (cx + 50, 100)
    keypoints[23], keypoints[24] = (cx - 40, 250), (cx + 40, 250)
    player = Player(pid, pose=PoseObservation(keypoints, np.ones(33, np.float32)), visible=True)
    if with_hand:
        hand = np.zeros((21, 2), np.float32)
        hand[0], hand[5], hand[9], hand[17] = (60, 300), (80, 270), (70, 265), (50, 275)
        player.hands[0] = HandObservation(hand, 1.0, 0)
    return player


def blank():
    return np.full((400, 400, 3), 128, np.uint8)


def changed(image):
    return np.any(image != 128, axis=2)


def person_mask(x0=100, x1=300):
    mask = np.zeros((400, 400), np.float32)
    mask[50:350, x0:x1] = 1.0
    return mask


def test_full_hp_and_no_recent_hit_leave_the_body_clean():
    image = blank()
    draw_body_damage(image, make_player(with_hand=True), 1.0, math.inf, person_mask())
    assert np.all(image == 128)


def test_more_damage_marks_more_of_the_torso():
    areas = []
    for hp in (0.75, 0.5, 0.25):
        image = blank()
        draw_body_damage(image, make_player(), hp, math.inf)
        areas.append(int(changed(image).sum()))
    assert 0 < areas[0] < areas[1] < areas[2]


def test_torso_marks_stay_on_the_torso():
    image = blank()
    draw_body_damage(image, make_player(), 0.2, math.inf)
    rows, cols = np.nonzero(changed(image))
    assert rows.min() >= 70 and rows.max() <= 270 and cols.min() >= 130 and cols.max() <= 270


def test_hands_get_a_bandage():
    image = blank()
    draw_body_damage(image, make_player(with_hand=True), 0.5, math.inf)
    assert changed(image)[255:310, 40:100].any()
    plain = blank()
    draw_body_damage(plain, make_player(), 0.5, math.inf)
    assert not changed(plain)[255:310, 40:100].any()


def test_a_fresh_hit_flashes_the_silhouette_and_fades():
    fresh, old = blank(), blank()
    draw_body_damage(fresh, make_player(), 1.0, 0.0, person_mask())
    draw_body_damage(old, make_player(), 1.0, 1.0, person_mask())
    assert changed(fresh)[200, 200] and not changed(fresh)[200, 20]
    assert np.all(old == 128)


def test_silhouette_tint_needs_the_person_mask():
    image = blank()
    draw_body_damage(image, make_player(), 1.0, 0.0, None)
    assert np.all(image == 128)


def test_each_player_only_tints_their_own_side_of_the_mask():
    players = [make_player(1, cx=120), make_player(2, cx=280)]
    columns = player_columns(players)
    image = blank()
    draw_body_damage(image, players[0], 1.0, 0.0, person_mask(0, 400), columns[1])
    assert changed(image)[200, 120] and not changed(image)[200, 280]


def test_nothing_is_drawn_below_the_wall():
    image = blank()
    draw_body_damage(image, make_player(), 0.2, 0.0, person_mask(), floor_y=150)
    assert not changed(image)[150:].any()
    assert changed(image)[100:150].any()


def test_tiny_torsos_are_skipped():
    player = make_player()
    player.pose.keypoints[12] = player.pose.keypoints[11] + 5
    image = blank()
    draw_body_damage(image, player, 0.2, math.inf)
    assert np.all(image == 128)


def test_low_hp_drains_the_colour_from_the_player_but_not_the_background():
    image = np.full((400, 400, 3), (50, 100, 200), np.uint8)
    draw_body_damage(image, make_player(), 0.2, math.inf, person_mask())
    spread = lambda pixel: int(pixel.max()) - int(pixel.min())  # noqa: E731
    assert spread(image[70, 115]) < 150  # on the person, clear of the marks
    assert spread(image[70, 20]) == 150  # background in the same region
