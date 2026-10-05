import math

import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals
from highnoon.game.actions import PlayerActions

DT = 1 / 30


def sig(elev_deg=0.0, aim=True, shoulder_y=300.0, arms_down=False) -> MotionSignals:
    e = math.radians(elev_deg)
    return MotionSignals(
        torso_len=200.0,
        shoulder_y=shoulder_y,
        aim_valid=aim,
        aim_origin=np.array([500.0, 300.0], np.float32),
        aim_dir=np.array([math.cos(e), -math.sin(e)], np.float32),
        elevation=e,
        arms_down=arms_down,
    )


def run(actions, signals, t0=0.0):
    frames = []
    for i, s in enumerate(signals):
        frames.append(actions.update(s, t0 + i * DT))
    return frames


def shots(frames):
    return [f.shot for f in frames if f.shot is not None]


def test_flick_fires_once_using_pre_flick_direction():
    acts = PlayerActions(GameConfig())
    steady = [sig(0)] * 20  # aim steadily for 0.66 s
    flick = [sig(0), sig(12), sig(28), sig(35), sig(30), sig(10), sig(0)]
    frames = run(acts, steady + flick + [sig(0)] * 10)
    fired = shots(frames)
    assert len(fired) == 1
    # Direction from before the flick: horizontal, not tilted up.
    np.testing.assert_allclose(fired[0].direction, [1.0, 0.0], atol=1e-6)


def test_raising_arm_into_aim_does_not_fire():
    acts = PlayerActions(GameConfig())
    # Arm comes up from the side: invalid aim, then a fast upward sweep into the aim range.
    raise_arm = [sig(-80, aim=False)] * 10 + [sig(-45), sig(-30), sig(-10), sig(5), sig(10)] + [sig(10)] * 20
    assert shots(run(acts, raise_arm)) == []


def test_slow_aim_drift_does_not_fire():
    acts = PlayerActions(GameConfig())
    drift = [sig(i * 0.5) for i in range(90)]  # 15 deg/s upward
    assert shots(run(acts, drift)) == []


def test_cooldown_blocks_rapid_second_shot():
    acts = PlayerActions(GameConfig())
    flick = [sig(0), sig(15), sig(30), sig(0)]
    frames = run(acts, [sig(0)] * 20 + flick + flick + [sig(0)] * 30 + flick)
    assert len(shots(frames)) == 2  # first flick, the immediate second is blocked, the late third fires


def test_duck_hysteresis_and_no_shooting_while_ducked():
    acts = PlayerActions(GameConfig())
    standing = [sig(0, shoulder_y=300)] * 15
    ducking = [sig(0, shoulder_y=300 + 200 * d) for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    half_up = [sig(0, shoulder_y=300 + 200 * 0.3)] * 5  # between exit (0.2) and enter (0.35)
    frames = run(acts, standing + ducking + half_up)
    assert not frames[14].ducked
    assert frames[-6].ducked  # entered at drop 0.4
    assert frames[-1].ducked  # still ducked at 0.3 thanks to hysteresis
    assert not frames[-1].aiming
    up = run(acts, [sig(0, shoulder_y=300)] * 3, t0=10)
    assert not up[-1].ducked


def test_reload_after_hold_fires_once():
    acts = PlayerActions(GameConfig())
    frames = run(acts, [sig(aim=False, arms_down=True)] * 40)
    reloads = [i for i, f in enumerate(frames) if f.reload]
    assert len(reloads) == 1
    assert reloads[0] * DT >= GameConfig().reload_hold_s - 1e-6
