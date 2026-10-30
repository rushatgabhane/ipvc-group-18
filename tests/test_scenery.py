import numpy as np

from highnoon.config import GameConfig
from highnoon.game.entities import Effect
from highnoon.game.game import Game
from highnoon.render.scenery import Scenery


def live_game():
    game = Game(GameConfig(), (320, 180))
    game.mode = "duel"
    game.reset_round()
    return game


def test_it_stays_day_however_long_and_hard_the_round_goes():
    game = live_game()
    scenery = Scenery()
    scenery.build(320, 180, game.cfg.cover_height)
    scenery.update(game)
    day_sky, day_wall = scenery.background.copy(), scenery.wall.copy()
    game.states[1] = type("State", (), {"hp": 5})()
    for minutes in range(1, 6):
        game._t = minutes * 60.0
        scenery.update(game)
    assert np.array_equal(scenery.background, day_sky)
    assert np.array_equal(scenery.wall, day_wall)


def test_wall_damage_lasts_for_the_round_only():
    game = live_game()
    scenery = Scenery()
    scenery.build(320, 180, game.cfg.cover_height)
    scenery.update(game)
    clean = scenery.wall.copy()
    impact = np.array([160.0, scenery.wall_top + 20.0], np.float32)
    game.effects.append(Effect("blocked", impact, 0.0, 1))
    scenery.update(game)
    assert not np.array_equal(scenery.wall, clean)
    game.effects.clear()
    game.reset_round()
    scenery.update(game)
    assert np.array_equal(scenery.wall, clean)


def test_shots_above_the_wall_leave_no_mark():
    game = live_game()
    scenery = Scenery()
    scenery.build(320, 180, game.cfg.cover_height)
    scenery.update(game)
    clean = scenery.wall.copy()
    game.effects.append(Effect("blocked", np.array([160.0, 10.0], np.float32), 0.0, 1))
    scenery.update(game)
    assert np.array_equal(scenery.wall, clean)


def test_blast_above_the_wall_scars_the_ground_but_not_the_sky_until_next_round():
    game = live_game()
    scenery = Scenery()
    scenery.build(320, 180, game.cfg.cover_height)
    scenery.update(game)
    clean = scenery.background.copy()
    game.effects.append(Effect("explosion", np.array([160.0, 100.0], np.float32), 0.0, 1))
    scenery.update(game)
    changed = np.any(scenery.background != clean, axis=2)
    assert changed[100 : scenery.wall_top].any()
    assert not changed[:20].any()
    game.effects.clear()
    game.reset_round()
    scenery.update(game)
    assert np.array_equal(scenery.background, clean)
