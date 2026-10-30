import numpy as np
import pytest

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals, Player, PoseObservation
from highnoon.game.entities import Bottle, Powerup
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
    t = fire(game, t, settle_frames=20)
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


def test_powerup_spawns_every_interval_and_falls_down_the_middle():
    game, t = start_duel()
    cfg = game.cfg
    t = step(game, duel, int((cfg.powerup_interval_s - 1) / DT), t)
    assert game.powerup is None
    step(game, duel, int(2 / DT), t)
    powerup = game.powerup
    assert powerup is not None
    start, end = powerup.center(powerup.t_spawn), powerup.center(powerup.t_end)
    assert start[1] < 0 and end[1] == pytest.approx(game.wall_top, abs=0.5)
    assert start[0] == end[0] and abs(start[0] - 640) <= cfg.powerup_center_spread * 1280


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
    fire(game, t, aim=UP_RIGHT, settle_frames=20)
    assert game.powerup is None
    assert game.states[1].rockets == 1 and game.states[2].rockets == 0
    assert any(e.kind == "pickup" and e.player == 1 and e.detail == "rocket" for e in game.effects)


def test_rocket_flies_an_arc_and_uses_no_ammo():
    game, t = start_duel()
    game.states[1].rockets = 1
    t = step(game, duel, 1, t)
    path = game.aims[1].path
    assert path is not None and path[-1][1] > path[0][1]
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


AT_OWN_FEET = (0.3, 1.0)


def throw_grenade(
    aim=RIGHT,
    p2_y=STANDING,
    p2_x=900.0,
    p2_x_on_landing=None,
    p1_y_on_landing=STANDING,
    p1_hp=None,
    p2_hp=None,
) -> Game:
    def players(hand=OPEN, p1_y=STANDING, x=p2_x):
        return [
            make_player(1, 300, shoulder_y=p1_y, aim_dir=aim, hand=hand),
            make_player(2, x, shoulder_y=p2_y),
        ]

    game, t = start_duel(players)
    game.states[1].grenades = 1
    game.states[1].hp = p1_hp or game.states[1].hp
    game.states[2].hp = p2_hp or game.states[2].hp
    t = step(game, players, 12, t)
    t = step(game, lambda: players(FIST), 1, t)
    assert len(game.grenades) == 1 and game.states[1].ammo == game.cfg.max_ammo
    moved_x = p2_x if p2_x_on_landing is None else p2_x_on_landing
    step(game, lambda: players(p1_y=p1_y_on_landing, x=moved_x), int(game.cfg.grenade_flight_s / DT) + 2, t)
    assert not game.grenades
    return game


def test_grenade_on_target_badly_hurts_without_killing_from_full_hp():
    game = throw_grenade()
    assert game.states[2].hp == 100 - game.cfg.grenade_damage and game.winner is None
    assert game.round_over_t is None
    assert game.states[1].hp == 100


def test_grenade_finishes_a_wounded_player():
    game = throw_grenade(p2_hp=50)
    assert game.states[2].hp == 0 and game.winner == 1


def test_grenade_spares_a_ducked_player():
    game = throw_grenade(aim=DOWN_RIGHT, p2_y=DUCKED)
    assert game.states[2].hp == 100 and game.winner is None
    assert any(e.kind == "safe" for e in game.effects)


def test_grenade_goes_where_it_is_aimed_not_to_the_opponent():
    game = throw_grenade(aim=(0.2, -1.0))
    assert game.states[2].hp == 100
    blast = next(e for e in game.effects if e.kind == "explosion")
    assert blast.pos[1] < 5 and blast.pos[0] < 500  # left through the top of the screen, far from them


def test_grenade_does_not_follow_the_target_and_damage_falls_off_with_distance():
    hp = [throw_grenade(p2_x_on_landing=x).states[2].hp for x in (900.0, 1000.0, 1100.0, 1180.0)]
    cfg = GameConfig()
    assert hp[0] == cfg.max_hp - cfg.grenade_damage
    assert hp == sorted(hp) and len(set(hp)) == len(hp)
    assert hp[-1] < cfg.max_hp - cfg.grenade_min_damage


def test_splash_reaches_most_of_a_blast_radius_away():
    game = throw_grenade(p2_x_on_landing=1180.0)
    assert game.states[2].hp < 100
    assert game.blast_radius(grenade=True) > game.explosion_radius(grenade=True)


def test_running_clear_of_the_blast_avoids_the_grenade():
    assert throw_grenade(p2_x_on_landing=1300.0).states[2].hp == 100


def test_grenade_splash_hurts_a_nearby_thrower_with_falloff():
    game = throw_grenade(aim=AT_OWN_FEET)
    cfg = game.cfg
    assert cfg.max_hp - cfg.grenade_damage < game.states[1].hp < cfg.max_hp - cfg.grenade_min_damage
    assert game.states[2].hp == cfg.max_hp


def test_ducking_protects_the_thrower_from_the_splash():
    assert throw_grenade(aim=AT_OWN_FEET, p1_y_on_landing=DUCKED).states[1].hp == 100


def test_blowing_yourself_up_gives_the_opponent_the_round():
    game = throw_grenade(aim=AT_OWN_FEET, p1_hp=5)
    assert game.states[1].hp == 0 and game.winner == 2
    assert game.scores.get(1, 0) == 0 and game.scores[2] == 1


def test_a_blast_that_would_kill_both_gives_the_round_to_the_thrower():
    game = throw_grenade(aim=AT_OWN_FEET, p2_x=550.0, p1_hp=5, p2_hp=5)
    assert game.states[2].hp == 0 and game.states[1].hp == 5
    assert game.winner == 1 and game.scores == {1: 1}


def test_grenades_are_the_rarer_powerup():
    game, _ = start_duel()
    kinds = [game._toss_powerup(0.0).kind for _ in range(1000)]
    assert 0.2 < kinds.count("grenade") / len(kinds) < 0.3
    assert set(kinds) == {"rocket", "grenade"}


def test_countdown_starts_only_with_both_players_and_blocks_shooting():
    game = Game(GameConfig(), (1280, 720))
    t = step(game, alone, 10, 0.0)
    assert game.mode == "idle"
    t = step(game, duel, 1, t)
    assert game.mode == "countdown" and game.countdown_left(t) > 2.5
    t = fire(game, t, settle_frames=20)
    assert game.states[2].hp == 100 and game.states[1].ammo == game.cfg.max_ammo
    step(game, duel, int(game.cfg.countdown_s / DT), t)
    assert game.mode == "duel"


def test_three_headshots_do_not_kill():
    game, _ = start_duel()
    assert 3 * game.cfg.head_damage < game.cfg.max_hp


def test_camping_drains_health_after_grace_period():
    game, t = start_duel()
    cfg = game.cfg
    t = step(game, p1_ducked, int(cfg.camp_grace_s / DT) - 3, t)
    assert game.states[1].hp == cfg.max_hp
    t = step(game, p1_ducked, int(3.2 / DT), t)
    drain = round(cfg.max_hp * cfg.camp_drain_fraction)
    assert game.states[1].hp == cfg.max_hp - 4 * drain
    assert any(e.kind == "camping" for e in game.effects)
    t = step(game, duel, 3, t)
    hp_after_standing = game.states[1].hp
    step(game, p1_ducked, int(2 / DT), t)
    assert game.states[1].hp == hp_after_standing


def test_camping_to_zero_hands_the_round_to_the_opponent():
    game, t = start_duel()
    game.states[1].hp = 5
    step(game, p1_ducked, int((game.cfg.camp_grace_s + 0.2) / DT), t)
    assert game.states[1].hp == 0 and game.winner == 2


def test_player_leaving_keeps_the_duel_and_asks_them_back():
    game, t = start_duel()
    game.states[2].hp = 60
    left_at = t

    def p2_gone():
        return [make_player(1, 300), Player(id=2, last_seen=left_at)]

    t = step(game, p2_gone, int(0.5 / DT), t)
    assert game.mode == "duel" and game.missing_players == []
    t = step(game, p2_gone, int(2 / DT), t)
    assert game.mode == "duel" and game.missing_players == [2]
    assert 0 < game.leave_countdown(2) < game.cfg.leave_timeout_s
    step(game, duel, 2, t)
    assert game.mode == "duel" and game.missing_players == [] and game.states[2].hp == 60


def test_player_who_stays_away_forfeits_after_the_countdown():
    game, t = start_duel()
    left_at = t

    def p2_gone():
        return [make_player(1, 300), Player(id=2, last_seen=left_at)]

    step(game, p2_gone, int((game.cfg.come_back_after_s + game.cfg.leave_timeout_s) / DT) + 3, t)
    assert game.mode == "forfeit" and game.forfeited == 2


def test_grenade_aim_previews_a_lob_onto_the_opponent():
    game, t = start_duel()
    game.states[1].grenades = 1
    step(game, duel, 1, t)
    aim = game.aims[1]
    assert aim.weapon == "grenade" and aim.path is not None
    assert aim.path[len(aim.path) // 2][1] < min(aim.path[0][1], aim.path[-1][1]) - 50
    assert np.allclose(aim.path[-1], aim.end)


def test_in_between_updates_on_the_same_frame_do_not_fire_twice():
    game, t = start_duel()
    t = step(game, duel, 12, t)
    fist = duel(hand=FIST)
    game.update(fist, t, set())
    game.update(fist, t + DT / 2, set())
    assert len(game.bullets) == 1 and game.states[1].ammo == game.cfg.max_ammo - 1


def start_practice():
    game = Game(GameConfig(practice=True), (1280, 720))
    t = step(game, alone, int(game.cfg.countdown_s / DT) + 2, 0.0)
    assert game.mode == "duel"
    return game, t


def test_practice_starts_with_one_player_and_spawns_bottles_above_the_wall():
    game, t = start_practice()
    step(game, alone, 2, t)
    assert len(game.bottles) == game.cfg.bottle_count
    assert all(b.center[1] < game.wall_top for b in game.bottles)


def test_practice_bullet_shatters_bottle_and_scores():
    game, t = start_practice()
    step(game, alone, 1, t)
    muzzle = game.muzzle(alone()[0].signals.aim_origin, np.array([1.0, 0.0], np.float32))
    game.bottles = [Bottle(muzzle + np.array([200.0, 0.0], np.float32), 40.0, 0.0)]
    game._next_bottle_t = np.inf
    t = fire_alone(game, t)
    assert game.scores[1] == 1
    assert not game.bottles


def fire_alone(game, t):
    t = step(game, alone, 12, t)
    t = step(game, lambda: [make_player(1, 300, hand=FIST), Player(id=2)], 1, t)
    return step(game, alone, 20, t)


def test_practice_ignores_second_player_as_a_target():
    game, t = start_practice()
    t = fire(game, t, settle_frames=20)
    assert game.states[2].hp == game.cfg.max_hp


def test_practice_shatters_one_of_several_bottles():
    game, t = start_practice()
    step(game, alone, 1, t)
    first, second = game.bottles[:2]
    game._shatter(1, second, t)
    assert first in game.bottles and second not in game.bottles


def test_practice_with_one_player_never_asks_anyone_to_come_back():
    game, t = start_practice()
    step(game, lambda: [make_player(1, 300)], int((game.cfg.come_back_after_s + game.cfg.leave_timeout_s) / DT) + 5, t)
    assert game.missing_players == []


def test_practice_bomb_hurts_whoever_shoots_it():
    game, t = start_practice()
    step(game, alone, 1, t)
    bomb = Powerup("bomb", np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0, 40.0, t, t + 5)
    game._collect(1, bomb, np.array([600.0, 200.0], np.float32), t)
    assert game.states[1].hp == game.cfg.max_hp - game.cfg.bomb_damage
    assert any(e.kind == "explosion" and e.detail == "bomb" for e in game.effects)
    assert game.states[1].rockets == 0 and game.states[1].grenades == 0


def test_blowing_yourself_up_ends_the_round_without_a_winner():
    game, t = start_practice()
    step(game, alone, 1, t)
    round_before = game.round
    game.states[1].hp = game.cfg.bomb_damage
    bomb = Powerup("bomb", np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0, 40.0, t, t + 5)
    game._collect(1, bomb, np.array([600.0, 200.0], np.float32), t)
    assert game.states[1].hp == 0 and game.round_over_t is not None
    assert game.winner is None and game.scores.get(1, 0) == 0
    step(game, alone, int(game.cfg.round_end_s / DT) + 2, t)
    assert game.round > round_before and game.states[1].hp == game.cfg.max_hp


def test_bombs_only_drop_in_practice():
    duel_game, _ = start_duel()
    practice_game, _ = start_practice()
    duel_kinds = {duel_game._toss_powerup(0.0).kind for _ in range(40)}
    practice_kinds = {practice_game._toss_powerup(0.0).kind for _ in range(40)}
    assert duel_kinds == {"rocket", "grenade"}
    assert practice_kinds == {"rocket", "bomb"}
