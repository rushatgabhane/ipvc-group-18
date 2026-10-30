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


def hp(value: int):
    return type("State", (), {"hp": value})()


def test_sunset_follows_round_time_or_the_lowest_players_hp_whichever_is_further():
    game = live_game()
    game.states = {1: hp(100), 2: hp(100)}
    assert sunset_target(game, 0.0) == 0.0
    assert sunset_target(game, SUNSET_AFTER_S / 4) == 0.25
    game.states = {1: hp(80), 2: hp(50)}
    assert sunset_target(game, SUNSET_AFTER_S / 4) == 0.5
    assert sunset_target(game, SUNSET_AFTER_S * 0.75) == 0.75
    assert sunset_target(game, SUNSET_AFTER_S * 2) == 1.0
    game.mode = "idle"
    assert sunset_target(game, SUNSET_AFTER_S) == 0.0


def test_sky_eases_to_sunset_after_a_big_hit_and_stops_there():
    game = live_game()
    game.states = {1: hp(100), 2: hp(100)}
    scenery = Scenery()
    scenery.build(320, 180, game.cfg.cover_height)
    scenery.update(game)
    day = scenery.background.astype(int)
    game.states[2] = hp(0)
    changes = []
    for frame in range(1, 301):
        game._t = frame / 30
        before = scenery.background.astype(int)
        scenery.update(game)
        changes.append(np.abs(scenery.background.astype(int) - before).max())
    assert scenery.sunset == 1.0
    assert np.abs(scenery.background.astype(int) - day).mean() > 10
    assert max(changes) <= 8, "it fades instead of jumping"
    assert max(changes[-30:]) == 0, "it stops at sunset"


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
