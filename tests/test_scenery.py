import numpy as np

from highnoon.config import GameConfig
from highnoon.game.entities import Effect
from highnoon.game.game import Game
from highnoon.render.scenery import SUNSET_AFTER_S, Scenery, sunset_target


def live_game():
    game = Game(GameConfig(), (320, 180))
    game.mode = "duel"
    game.reset_round()
    return game


def test_sunset_follows_round_time_only():
    game = live_game()
    assert sunset_target(game, 0.0) == 0.0
    assert sunset_target(game, SUNSET_AFTER_S / 2) == 0.5
    game.states[1] = type("State", (), {"hp": 5})()
    assert sunset_target(game, 1.0) == 1.0 / SUNSET_AFTER_S
    game.mode = "idle"
    assert sunset_target(game, 30.0) == 0.0


def test_sky_fades_gradually_to_sunset_and_stops_there():
    game = live_game()
    scenery = Scenery()
    scenery.build(320, 180, game.cfg.cover_height)
    scenery.update(game)
    day = scenery.background.astype(int)
    changes = []
    for second in range(1, 121):
        game._t = float(second)
        before = scenery.background.astype(int)
        scenery.update(game)
        changes.append(np.abs(scenery.background.astype(int) - before).max())
    assert scenery.sunset == 1.0
    assert np.abs(scenery.background.astype(int) - day).mean() > 10
    assert max(changes) <= 8, "each second only nudges the sky"
    assert max(changes[int(SUNSET_AFTER_S) + 1 :]) == 0, "it stops at sunset"


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
