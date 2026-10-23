import numpy as np
import pytest

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals, Player, PoseObservation
from highnoon.game.game import Game, Powerup
from highnoon.game.geometry import ray_circle, ray_exit_distance, ray_polygon, rect_polygon

DT = 1 / 30


def test_ray_circle():
    o, d = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert ray_circle(o, d, np.array([10.0, 0.0]), 2.0) == pytest.approx(8.0)
    assert ray_circle(o, d, np.array([10.0, 5.0]), 2.0) is None
    assert ray_circle(o, d, np.array([-10.0, 0.0]), 2.0) is None  # behind the shooter


def test_ray_polygon():
    o, d = np.array([0.0, 5.0]), np.array([1.0, 0.0])
    sq = rect_polygon(10, 0, 20, 10)
    assert ray_polygon(o, d, sq) == pytest.approx(10.0)
    assert ray_polygon(np.array([15.0, 5.0]), d, sq) == 0.0  # starts inside
    assert ray_polygon(o, np.array([-1.0, 0.0]), sq) is None


def test_ray_exit():
    assert ray_exit_distance(np.array([100.0, 100.0]), np.array([1.0, 0.0]), 1280, 720) == pytest.approx(1180)


def make_player(pid: int, x: float, shoulder_y: float = 300.0, aim_dir=(1.0, 0.0), aim=True) -> Player:
    kp = np.zeros((33, 2), np.float32)
    kp[:] = (x, shoulder_y + 100)
    kp[0] = (x, shoulder_y - 70)  # nose
    kp[11], kp[12] = (x - 60, shoulder_y), (x + 60, shoulder_y)
    kp[23], kp[24] = (x - 45, shoulder_y + 200), (x + 45, shoulder_y + 200)
    pose = PoseObservation(kp, np.ones(33, np.float32), np.zeros((33, 3), np.float32))
    d = np.array(aim_dir, np.float32)
    d /= np.linalg.norm(d)
    elevation = float(np.arctan2(-d[1], abs(d[0])))
    signals = MotionSignals(
        200.0, shoulder_y, aim, 1.0, kp[12] + d * 150, d, elevation, False, None, kp[0][1]
    )
    return Player(id=pid, pose=pose, smoothed=kp, signals=signals, visible=True)


def flick_dirs(direction):
    """Aim directions for a recoil flick starting from `direction` (rotate upward)."""
    d0 = np.array(direction, np.float32)
    out = []
    for deg in (0, 15, 30, 10):
        a = np.radians(deg) * (1 if d0[0] >= 0 else -1)
        c, s = np.cos(a), np.sin(a)
        out.append((d0[0] * c + d0[1] * s, -d0[0] * s + d0[1] * c))
    return out


def step(game, players_fn, n, t):
    for _ in range(n):
        game.update(players_fn(), t, set())
        t += DT
    return t


def test_duel_shot_hits_standing_player_and_misses_when_ducked():
    game = Game(GameConfig(trigger="flick"), (1280, 720))
    aim = (1.0, 0.0)  # P1 at x=300 aims right along y=300, through P2's chest
    t = step(game, lambda: [make_player(1, 300, aim_dir=aim), make_player(2, 900)], 20, 0.0)

    for d in flick_dirs(aim):
        t = step(game, lambda d=d: [make_player(1, 300, aim_dir=d), make_player(2, 900)], 1, t)
    t = step(game, lambda: [make_player(1, 300, aim_dir=aim), make_player(2, 900)], 12, t)
    assert game.states[2].hp == 100 - game.cfg.body_damage
    assert game.states[1].ammo == game.cfg.max_ammo - 1

    # Second shot: P2 ducks (head at y=580, below the wall top at 540) before the bullet lands.
    t += 1.0
    for d in flick_dirs(aim):
        t = step(game, lambda d=d: [make_player(1, 300, aim_dir=d), make_player(2, 900)], 1, t)
    t = step(game, lambda: [make_player(1, 300, aim_dir=aim), make_player(2, 900, shoulder_y=650)], 12, t)
    assert game.states[2].frame.ducked
    assert game.states[2].hp == 100 - game.cfg.body_damage


def test_wall_is_static_and_blocks_low_shots():
    game = Game(GameConfig(), (1280, 720))
    game.update([make_player(1, 300), make_player(2, 900)], 0.0, set())
    wall = game.cover
    assert (wall.x0, wall.y0, wall.x1, wall.y1) == (0.0, 540.0, 1280.0, 720.0)
    game.update([make_player(1, 200), make_player(2, 1100)], 1.0, set())
    assert game.cover.y0 == wall.y0 and game.cover.x0 == wall.x0  # does not follow the players
    # A shot angled down into the wall stops at it.
    origin, direction = np.array([400.0, 300.0]), np.array([0.6, 0.8])
    dist, (kind, _) = game._cast(1, origin, direction)
    assert kind == "cover"
    assert (origin + direction * dist)[1] == pytest.approx(540.0)


def test_practice_mode_with_one_player():
    game = Game(GameConfig(), (1280, 720))
    game.update([make_player(1, 300), Player(id=2)], 0.0, set())
    assert game.mode == "practice"
    assert len(game.targets) == game.cfg.practice_targets
    radius = 0.04 * 720
    keep_out = radius + 0.4 * 200  # around the nose (x=300) when no face is tracked
    for target in game.targets:
        assert not (300 - keep_out < target.center[0] < 300 + keep_out)
        assert target.center[1] < 540 - radius  # above the wall
    centers = [t.center for t in game.targets]
    assert min(np.linalg.norm(a - b) for i, a in enumerate(centers) for b in centers[i + 1 :]) > 0


def test_ducking_reloads():
    game = Game(GameConfig(), (1280, 720))
    assert game.cfg.max_ammo == 3
    t = step(game, lambda: [make_player(1, 300), make_player(2, 900)], 2, 0.0)
    game.states[1].ammo = 0
    t = step(game, lambda: [make_player(1, 300, shoulder_y=650), make_player(2, 900)], 2, t)
    assert game.states[1].ammo == game.cfg.max_ammo


def test_powerup_spawns_every_interval_in_duel_and_flies_an_arc():
    cfg = GameConfig()
    game = Game(cfg, (1280, 720))
    duel = lambda: [make_player(1, 300), make_player(2, 900)]  # noqa: E731
    t = step(game, duel, 1, 0.0)
    t = step(game, duel, int((cfg.powerup_interval_s - 1) / DT), t)
    assert game.powerup is None
    t = step(game, duel, int(2 / DT), t)
    pu = game.powerup
    assert pu is not None
    start, peak, end = pu.center(pu.t_spawn), pu.center(pu.t_spawn + cfg.powerup_rise_s), pu.center(pu.t_end)
    assert start[1] > 720 and end[1] == pytest.approx(start[1], abs=1e-2)  # tossed up from below and back
    assert 0.2 * 720 < peak[1] < 0.75 * 720  # peaks above the wall
    t = step(game, duel, int(2 * cfg.powerup_rise_s / DT) + 2, t)
    assert game.powerup is None  # fell back out without being shot


def test_powerups_only_in_duel():
    game = Game(GameConfig(), (1280, 720))
    step(game, lambda: [make_player(1, 300), Player(id=2)], int(40 / DT), 0.0)
    assert game.powerup is None


def put_powerup(game, kind, center, t):
    """A powerup hovering at `center` (no gravity, no motion) for tests."""
    game.powerup = Powerup(kind, np.array(center, np.float32), np.zeros(2, np.float32), 0.0, 30.0, t, t + 99)


def shoot(game, aim, t):
    """P1 at x=300 fires one flick shot along `aim`; returns the time after the bullet lands."""
    for d in flick_dirs(aim):
        t = step(game, lambda d=d: [make_player(1, 300, aim_dir=d), make_player(2, 900)], 1, t)
    return step(game, lambda: [make_player(1, 300, aim_dir=aim), make_player(2, 900)], 12, t)


def test_shooting_a_powerup_gives_it_to_the_shooter():
    game = Game(GameConfig(trigger="flick"), (1280, 720))
    up = (1.0, -1.0)  # 45 degrees up from P1's hand, away from P2
    t = step(game, lambda: [make_player(1, 300, aim_dir=up), make_player(2, 900)], 20, 0.0)
    hand = make_player(1, 300, aim_dir=up).signals.aim_origin
    put_powerup(game, "shield", hand + np.array([150.0, -150.0]), t)
    t = shoot(game, up, t)
    assert game.powerup is None
    assert game.states[1].shield and not game.states[2].shield
    assert any(e.kind == "pickup" and e.player == 1 and e.detail == "shield" for e in game.effects)


def test_shield_absorbs_one_hit_and_double_damage_doubles():
    game = Game(GameConfig(trigger="flick"), (1280, 720))
    aim = (1.0, 0.0)
    t = step(game, lambda: [make_player(1, 300, aim_dir=aim), make_player(2, 900)], 20, 0.0)
    game.states[2].shield = True
    t = shoot(game, aim, t)
    assert game.states[2].hp == 100 and not game.states[2].shield
    game.states[1].double_until = t + 10
    t = shoot(game, aim, t + 1.0)
    assert game.states[2].hp == 100 - 2 * game.cfg.body_damage


def test_heal_is_capped():
    game = Game(GameConfig(trigger="flick"), (1280, 720))
    up = (1.0, -1.0)
    t = step(game, lambda: [make_player(1, 300, aim_dir=up), make_player(2, 900)], 20, 0.0)
    game.states[1].hp = 80
    hand = make_player(1, 300, aim_dir=up).signals.aim_origin
    put_powerup(game, "heal", hand + np.array([150.0, -150.0]), t)
    shoot(game, up, t)
    assert game.states[1].hp == game.cfg.max_hp
