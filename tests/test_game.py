import numpy as np
import pytest

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals, Player, PoseObservation
from highnoon.game.game import Game
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
