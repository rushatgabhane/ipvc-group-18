import csv

from highnoon.core.match_log import MatchLog
from tests.test_game import DT, FIST, RIGHT, STANDING, duel, start_duel


def play_frames(game, log, players_fn, frames, t):
    for _ in range(frames):
        players = players_fn()
        game.update(players, t, set())
        log.update(game, players, t, {"latency": 20.0})
        t += DT
    return t


def shoot(game, log, t):
    t = play_frames(game, log, duel, 20, t)
    t = play_frames(game, log, lambda: duel(RIGHT, STANDING, FIST), 1, t)
    return play_frames(game, log, duel, 20, t)


def test_knockout_appends_one_row_per_game(tmp_path):
    path = tmp_path / "games.csv"
    for _ in range(2):
        game, t = start_duel()
        log = MatchLog(path)
        game.states[2].hp = game.cfg.body_damage
        t = shoot(game, log, t)
        assert game.winner == 1
        play_frames(game, log, duel, 30, t)
        log.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2
    assert rows[0]["result"] == "ko" and rows[0]["winner"] == "1"
    assert rows[0]["p1_shots"] == "1" and rows[0]["p1_bullet_hits"] == "1"
    assert rows[0]["p1_accuracy"] == "1.00" and rows[0]["p1_fists"] == "1"
