import math

import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals
from highnoon.game.actions import PlayerActions

DT = 1 / 30
H = 720  # frame height; the wall top is at 0.75 * H = 540
FLICK = GameConfig(trigger="flick")


def sig(elev_deg=0.0, aim=True, head_y=200.0, arms_down=False, hand=None) -> MotionSignals:
    e = math.radians(elev_deg)
    return MotionSignals(
        torso_len=200.0,
        shoulder_y=head_y + 100,
        aim_valid=aim,
        extension=1.0,
        aim_origin=np.array([500.0, 300.0], np.float32),
        aim_dir=np.array([math.cos(e), -math.sin(e)], np.float32),
        elevation=e,
        arms_down=arms_down,
        hand_openness=hand,
        head_y=head_y,
    )


def run(actions, signals, t0=0.0):
    frames = []
    for i, s in enumerate(signals):
        frames.append(actions.update(s, t0 + i * DT, H))
    return frames


def shots(frames):
    return [f.shot for f in frames if f.shot is not None]


def test_flick_fires_once_using_pre_flick_direction():
    acts = PlayerActions(FLICK)
    steady = [sig(0)] * 20  # aim steadily for 0.66 s
    flick = [sig(0), sig(12), sig(28), sig(35), sig(30), sig(10), sig(0)]
    frames = run(acts, steady + flick + [sig(0)] * 10)
    fired = shots(frames)
    assert len(fired) == 1
    # Direction from before the flick: horizontal, not tilted up.
    np.testing.assert_allclose(fired[0].direction, [1.0, 0.0], atol=1e-6)


def test_raising_arm_into_aim_does_not_fire():
    acts = PlayerActions(FLICK)
    # Arm comes up from the side: invalid aim, then a fast upward sweep into the aim range.
    raise_arm = [sig(-80, aim=False)] * 10 + [sig(-45), sig(-30), sig(-10), sig(5), sig(10)] + [sig(10)] * 20
    assert shots(run(acts, raise_arm)) == []


def test_slow_aim_drift_does_not_fire():
    acts = PlayerActions(FLICK)
    drift = [sig(i * 0.5) for i in range(90)]  # 15 deg/s upward
    assert shots(run(acts, drift)) == []


def test_cooldown_blocks_rapid_second_shot():
    acts = PlayerActions(FLICK)
    flick = [sig(0), sig(15), sig(30), sig(0)]
    frames = run(acts, [sig(0)] * 20 + flick + flick + [sig(0)] * 30 + flick)
    assert len(shots(frames)) == 2  # first flick, the immediate second is blocked, the late third fires


def test_duck_when_head_goes_below_the_wall():
    acts = PlayerActions(FLICK)
    standing = [sig(0, head_y=300)] * 5
    down = [sig(0, head_y=y) for y in (400, 500, 545)]  # crosses the wall top at 540
    near_edge = [sig(0, head_y=y) for y in (530, 525, 535)]  # within the 0.03 * 720 = 21.6 px margin
    frames = run(acts, standing + down + near_edge)
    assert not frames[6].ducked  # head at 500: still above the wall
    assert frames[7].ducked
    assert all(f.ducked for f in frames[8:])  # hysteresis: no flicker at the edge
    assert not frames[-1].aiming  # cannot shoot while ducked
    up = run(acts, [sig(0, head_y=510)], t0=10)
    assert not up[-1].ducked  # 510 < 540 - 21.6


def test_duck_is_absolute_not_relative_to_standing_height():
    # A player who just stays low (e.g. sitting close to the camera) is ducked from the start,
    # and one standing tall never is, however long they stay.
    assert run(PlayerActions(FLICK), [sig(0, head_y=600)] * 3)[-1].ducked
    assert not any(f.ducked for f in run(PlayerActions(FLICK), [sig(0, head_y=520)] * 90))


def test_reload_after_hold_fires_once():
    acts = PlayerActions(FLICK)
    frames = run(acts, [sig(aim=False, arms_down=True)] * 40)
    reloads = [i for i, f in enumerate(frames) if f.reload]
    assert len(reloads) == 1
    assert reloads[0] * DT >= FLICK.reload_hold_s - 1e-6


def test_fist_fires_once_per_close():
    acts = PlayerActions(GameConfig(trigger="fist"))
    open_aim = [sig(0, hand=1.9)] * 15
    close = [sig(0, hand=1.0)] * 10  # holding the fist must not keep firing
    frames = run(acts, open_aim + close + [sig(0, hand=1.9)] * 15 + close)
    fired = [i for i, f in enumerate(frames) if f.shot is not None]
    assert fired == [15, 40]


def test_fist_needs_open_hand_first_and_steady_aim():
    acts = PlayerActions(GameConfig(trigger="fist"))
    # Fist already closed while raising the arm -> no shot; opening then closing -> shot.
    frames = run(acts, [sig(0, hand=1.0)] * 15 + [sig(0, hand=1.9)] * 3 + [sig(0, hand=1.0)])
    assert [i for i, f in enumerate(frames) if f.shot is not None] == [18]
    # Closing right after the aim starts (not yet armed) is ignored.
    acts = PlayerActions(GameConfig(trigger="fist"))
    frames = run(acts, [sig(aim=False, hand=1.9)] * 5 + [sig(0, hand=1.9)] * 3 + [sig(0, hand=1.0)])
    assert all(f.shot is None for f in frames)


def test_fist_hysteresis_and_lost_hand():
    acts = PlayerActions(GameConfig(trigger="fist"))
    # Openness wobbling between the thresholds, or the hand briefly lost, must not fire.
    seq = [sig(0, hand=1.9)] * 15 + [sig(0, hand=h) for h in (1.4, 1.3, None, 1.45, 1.35)]
    assert all(f.shot is None for f in run(acts, seq))


def test_fist_uses_aim_from_before_the_trigger():
    acts = PlayerActions(GameConfig(trigger="fist"))
    steady = [sig(0, hand=1.9)] * 15
    drift = [sig(8, hand=1.9), sig(16, hand=1.9), sig(16, hand=1.0)]  # arm jerks up while closing
    shot = next(f.shot for f in run(acts, steady + drift) if f.shot is not None)
    assert np.degrees(np.arctan2(-shot.direction[1], shot.direction[0])) < 8.5
