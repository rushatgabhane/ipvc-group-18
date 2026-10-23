import numpy as np
import pytest

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals, Player, PoseObservation
from highnoon.game.entities import Powerup
from highnoon.game.game import Game
from highnoon.game.geometry import (
    ray_exit_distance,
    rect_polygon,
    segment_circle_hits,
    segment_polygon_hits,
)

DT = 1 / 30
OPEN, FIST = 1.9, 1.0
STANDING, DUCKED = 300.0, 650.0
RIGHT, UP_RIGHT, DOWN_RIGHT = (1.0, 0.0), (1.0, -1.0), (1.0, 0.45)


def segment(start, end):
    return np.array([start], np.float32), np.array([end], np.float32)


def test_segment_circle_hits():
    center = np.array([10.0, 0.0], np.float32)
    assert segment_circle_hits(*segment((0, 0), (20, 0)), center, 2.0)[0] == pytest.approx(0.4)
    assert segment_circle_hits(*segment((0, 5), (20, 5)), center, 2.0)[0] == np.inf
    assert segment_circle_hits(*segment((0, 0), (5, 0)), center, 2.0)[0] == np.inf
    assert segment_circle_hits(*segment((10, 0), (20, 0)), center, 2.0)[0] == 0.0


def test_segment_polygon_hits_and_batches():
    square = rect_polygon(10, 0, 20, 10)
    starts = np.array([[0, 5], [15, 5], [0, 5], [0, 50]], np.float32)
    ends = np.array([[20, 5], [30, 5], [-10, 5], [20, 50]], np.float32)
    hits = segment_polygon_hits(starts, ends, square)
    assert hits[0] == pytest.approx(0.5)
    assert hits[1] == 0.0
    assert hits[2] == np.inf and hits[3] == np.inf


def test_ray_exit():
    exit_distance = ray_exit_distance(np.array([100.0, 100.0]), np.array([1.0, 0.0]), 1280, 720)
    assert exit_distance == pytest.approx(1180)


def make_player(pid: int, x: float, shoulder_y: float = STANDING, aim_dir=RIGHT, hand=OPEN) -> Player:
    keypoints = np.zeros((33, 2), np.float32)
    keypoints[:] = (x, shoulder_y + 100)
    keypoints[0] = (x, shoulder_y - 70)
    keypoints[11], keypoints[12] = (x - 60, shoulder_y), (x + 60, shoulder_y)
    keypoints[23], keypoints[24] = (x - 45, shoulder_y + 200), (x + 45, shoulder_y + 200)
    pose = PoseObservation(keypoints, np.ones(33, np.float32))
    direction = np.array(aim_dir, np.float32)
    direction /= np.linalg.norm(direction)
    elevation = float(np.arctan2(-direction[1], abs(direction[0])))
    aim_origin = keypoints[12] + direction * 150
    signals = MotionSignals(200.0, True, 1.0, aim_origin, direction, elevation, hand, keypoints[0][1])
    return Player(id=pid, pose=pose, smoothed=keypoints, signals=signals, visible=True)


def duel(aim=RIGHT, p2_y=STANDING, hand=OPEN):
    return [make_player(1, 300, aim_dir=aim, hand=hand), make_player(2, 900, shoulder_y=p2_y)]


def p1_ducked():
    return [make_player(1, 300, shoulder_y=DUCKED), make_player(2, 900)]


def alone():
    return [make_player(1, 300), Player(id=2)]


def step(game, players_fn, frames, t):
    for _ in range(frames):
        game.update(players_fn(), t, set())
        t += DT
    return t


def start_duel(players_fn=duel):
    game = Game(GameConfig(), (1280, 720))
    t = step(game, players_fn, int(game.cfg.countdown_s / DT) + 2, 0.0)
    assert game.mode == "duel"
    return game, t


def fire(game, t, aim=RIGHT, p2_y=STANDING, settle_frames=1):
    t = step(game, lambda: duel(aim, p2_y), 12, t)
    t = step(game, lambda: duel(aim, p2_y, FIST), 1, t)
    return step(game, lambda: duel(aim, p2_y), settle_frames, t)


def test_shot_hits_standing_player_and_is_blocked_when_ducked():
    game, t = start_duel()
    t = fire(game, t, settle_frames=12)
    assert game.states[2].hp == 100 - game.cfg.body_damage
    assert game.states[1].ammo == game.cfg.max_ammo - 1

    t = fire(game, t + 1.0, settle_frames=0)
    t = step(game, lambda: duel(p2_y=DUCKED), 12, t)
    assert game.states[2].frame.ducked
    assert game.states[2].hp == 100 - game.cfg.body_damage


def test_wall_is_static_and_blocks_low_shots():
    game = Game(GameConfig(), (1280, 720))
    game.update(duel(), 0.0, set())
    assert game.wall_top == 540.0
    distance, (kind, _) = game._cast(1, np.array([400.0, 300.0]), np.array([0.6, 0.8]))
    assert kind == "cover"
    assert 300.0 + 0.8 * distance == pytest.approx(540.0)


def test_ducking_reloads():
    game, t = start_duel()
    assert game.cfg.max_ammo == 3
    game.states[1].ammo = 0
    t = step(game, p1_ducked, 5, t)
    assert game.states[1].ammo == 0
    step(game, p1_ducked, int(game.cfg.reload_s / DT) + 1, t)
    assert game.states[1].ammo == game.cfg.max_ammo


def test_one_player_is_not_a_game():
    game = Game(GameConfig(), (1280, 720))
    game.update([make_player(1, 300), Player(id=2)], 0.0, set())
    assert game.mode == "idle"


def test_powerup_spawns_every_interval_in_duel_and_flies_an_arc():
    game, t = start_duel()
    cfg = game.cfg
    t = step(game, duel, int((cfg.powerup_interval_s - 1) / DT), t)
    assert game.powerup is None
    step(game, duel, int(2 / DT), t)
    powerup = game.powerup
    assert powerup is not None
    start, peak, end = (powerup.center(powerup.t_spawn + k * cfg.powerup_rise_s) for k in (0, 1, 2))
    assert start[1] > 720 and end[1] == pytest.approx(start[1], abs=1e-2)
    assert 0.2 * 720 < peak[1] < 0.75 * 720


def test_powerups_only_in_duel():
    game = Game(GameConfig(), (1280, 720))
    step(game, alone, int(40 / DT), 0.0)
    assert game.powerup is None


def place_powerup(game, kind, t):
    hand = make_player(1, 300, aim_dir=UP_RIGHT).signals.aim_origin
    center = hand + np.array([150.0, -150.0], np.float32)
    game.powerup = Powerup(kind, center, np.zeros(2, np.float32), 0.0, 30.0, t, t + 99)


def test_shooting_a_powerup_gives_it_to_the_shooter():
    game, t = start_duel()
    place_powerup(game, "rocket", t)
    fire(game, t, aim=UP_RIGHT, settle_frames=12)
    assert game.powerup is None
    assert game.states[1].rockets == 1 and game.states[2].rockets == 0
    assert any(e.kind == "pickup" and e.player == 1 and e.detail == "rocket" for e in game.effects)


def test_rocket_flies_an_arc_and_uses_no_ammo():
    game, t = start_duel()
    game.states[1].rockets = 1
    t = step(game, duel, 1, t)
    path = game.aims[1].path
    assert path is not None and path[-1][1] > path[0][1] + 5
    t = fire(game, t, settle_frames=0)
    assert game.states[1].rockets == 0 and game.states[1].ammo == game.cfg.max_ammo
    assert len(game.rockets) == 1 and not game.bullets
    rocket = game.rockets[0]
    assert rocket.at(rocket.t_fire + 0.3)[1] > rocket.origin[1]
    step(game, duel, 15, t)
    assert not game.rockets
    assert any(e.kind == "explosion" for e in game.effects)
    assert game.states[2].hp <= 100 - game.cfg.rocket_min_damage


def test_rocket_blast_reaches_a_ducked_player():
    game, t = start_duel(lambda: duel(DOWN_RIGHT, DUCKED))
    game.states[1].rockets = 1
    fire(game, t, aim=DOWN_RIGHT, p2_y=DUCKED, settle_frames=40)
    assert game.states[2].frame.ducked
    assert game.states[2].hp < 100


def throw_grenade(p2_y: float) -> Game:
    game, t = start_duel(lambda: duel(UP_RIGHT, p2_y))
    game.states[1].grenades = 1
    t = fire(game, t, aim=UP_RIGHT, p2_y=p2_y, settle_frames=0)
    assert len(game.grenades) == 1 and game.states[1].ammo == game.cfg.max_ammo
    step(game, lambda: duel(UP_RIGHT, p2_y), int(game.cfg.grenade_flight_s / DT) + 2, t)
    assert not game.grenades
    return game


def test_grenade_kills_a_standing_player():
    game = throw_grenade(STANDING)
    assert game.states[2].hp == 0 and game.winner == 1


def test_grenade_spares_a_ducked_player():
    game = throw_grenade(DUCKED)
    assert game.states[2].hp == 100 and game.winner is None
    assert any(e.kind == "safe" for e in game.effects)


def test_countdown_starts_only_with_both_players_and_blocks_shooting():
    game = Game(GameConfig(), (1280, 720))
    t = step(game, alone, 10, 0.0)
    assert game.mode == "idle"
    t = step(game, duel, 1, t)
    assert game.mode == "countdown" and game.countdown_left(t) > 2.5
    t = fire(game, t, settle_frames=12)
    assert game.states[2].hp == 100 and game.states[1].ammo == game.cfg.max_ammo
    step(game, duel, int(game.cfg.countdown_s / DT), t)
    assert game.mode == "duel"


def test_short_detection_loss_does_not_cancel_the_duel():
    game, t = start_duel()
    t = step(game, alone, 5, t)
    assert game.mode == "duel"
    step(game, alone, int(game.cfg.player_grace_s / DT) + 2, t)
    assert game.mode == "idle"


def test_three_headshots_do_not_kill():
    game, _ = start_duel()
    assert 3 * game.cfg.head_damage < game.cfg.max_hp
