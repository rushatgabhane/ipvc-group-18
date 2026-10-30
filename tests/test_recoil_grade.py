from types import SimpleNamespace

import numpy as np
import pytest

from highnoon.game.entities import Effect
from highnoon.render.renderer import build_grade_lut
from highnoon.render.world import RECOIL_S, draw_weapon, recoil


def muzzle(player: int, t: float) -> Effect:
    return Effect("muzzle", np.zeros(2, np.float32), t, player, detail="gun")


def make_game(effects=()):
    return SimpleNamespace(
        effects=list(effects), height=720, cfg=SimpleNamespace(hand_reach=0.06), reach=lambda weapon: 0.27
    )


def make_aim(direction):
    return SimpleNamespace(
        origin=np.array([300.0, 300.0], np.float32),
        direction=np.array(direction, np.float32),
        weapon="gun",
    )


def weapon_pixels(game, aim, kick):
    image = np.zeros((720, 1280, 3), np.uint8)
    draw_weapon(image, game, aim, (200, 200, 200), kick)
    rows, cols = np.nonzero(image.any(axis=2))
    return rows, cols


def test_recoil_peaks_at_the_shot_and_fades():
    game = make_game([muzzle(1, 10.0)])
    assert recoil(game, 1, 10.0) == pytest.approx(1.0)
    assert 0.0 < recoil(game, 1, 10.0 + RECOIL_S / 2) < 1.0
    assert recoil(game, 1, 10.0 + RECOIL_S) == 0.0


def test_recoil_only_follows_the_player_who_fired():
    game = make_game([muzzle(1, 10.0)])
    assert recoil(game, 2, 10.0) == 0.0
    assert recoil(make_game(), 1, 10.0) == 0.0


def test_the_latest_shot_wins():
    game = make_game([muzzle(1, 10.0), muzzle(1, 10.1)])
    assert recoil(game, 1, 10.1) == pytest.approx(1.0)


def test_the_weapon_is_kicked_back_and_raised():
    game, aim = make_game(), make_aim((1.0, 0.0))
    rest_rows, rest_cols = weapon_pixels(game, aim, 0.0)
    kick_rows, kick_cols = weapon_pixels(game, aim, 1.0)
    assert kick_cols.max() < rest_cols.max()
    assert kick_rows.min() < rest_rows.min()


def test_a_weapon_aimed_left_is_raised_too():
    game, aim = make_game(), make_aim((-1.0, 0.0))
    rest_rows, rest_cols = weapon_pixels(game, aim, 0.0)
    kick_rows, kick_cols = weapon_pixels(game, aim, 1.0)
    assert kick_cols.min() > rest_cols.min()
    assert kick_rows.min() < rest_rows.min()


def test_grade_lut_is_warm_and_keeps_black_and_white_ordered():
    lut = build_grade_lut()
    assert lut.shape == (256, 1, 3) and lut.dtype == np.uint8
    mid = lut[128, 0]
    assert mid[2] > mid[1] > mid[0]  # BGR: red above green above blue
    assert np.all(lut[0, 0] == 0)
    assert np.all(np.diff(lut[:, 0, :].astype(int), axis=0) >= 0)
