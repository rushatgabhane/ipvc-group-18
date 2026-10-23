import math

import numpy as np

from highnoon.config import GameConfig
from highnoon.contracts import MotionSignals
from highnoon.game.actions import PlayerActions

DT = 1 / 30
H = 720
OPEN, FIST = 1.9, 1.0
CFG = GameConfig()


def sig(elev_deg=0.0, aim=True, head_y=200.0, hand=OPEN) -> MotionSignals:
    e = math.radians(elev_deg)
    return MotionSignals(
        torso_len=200.0,
        aim_valid=aim,
        extension=1.0,
        aim_origin=np.array([500.0, 300.0], np.float32),
        aim_dir=np.array([math.cos(e), -math.sin(e)], np.float32),
        elevation=e,
        hand_openness=hand,
        head_y=head_y,
    )


def run(actions, signals, t0=0.0):
    return [actions.update(s, t0 + i * DT, H) for i, s in enumerate(signals)]


def shot_frames(frames):
    return [i for i, f in enumerate(frames) if f.shot is not None]


def test_fist_fires_once_per_close():
    frames = run(
        PlayerActions(CFG), [sig()] * 15 + [sig(hand=FIST)] * 10 + [sig()] * 15 + [sig(hand=FIST)] * 10
    )
    assert shot_frames(frames) == [15, 40]


def test_fist_needs_open_hand_first_and_steady_aim():
    frames = run(PlayerActions(CFG), [sig(hand=FIST)] * 15 + [sig()] * 6 + [sig(hand=FIST)])
    assert shot_frames(frames) == [21]
    frames = run(PlayerActions(CFG), [sig(aim=False)] * 5 + [sig()] * 3 + [sig(hand=FIST)])
    assert shot_frames(frames) == []


def test_fist_hysteresis_and_lost_hand():
    wobble = [sig(hand=h) for h in (1.4, 1.3, None, 1.45, 1.35)]
    assert shot_frames(run(PlayerActions(CFG), [sig()] * 15 + wobble)) == []


def test_cooldown_blocks_rapid_second_shot():
    flick_fist = [sig(hand=FIST), sig()]
    frames = run(PlayerActions(CFG), [sig()] * 15 + flick_fist * 3 + [sig()] * 15 + flick_fist)
    assert shot_frames(frames) == [15, 36]


def test_shot_uses_aim_from_before_the_trigger():
    jerk = [sig(8), sig(16), sig(16, hand=FIST)]
    shot = next(f.shot for f in run(PlayerActions(CFG), [sig()] * 15 + jerk) if f.shot is not None)
    assert np.degrees(np.arctan2(-shot.direction[1], shot.direction[0])) < 8.5


def test_duck_when_head_goes_below_the_wall():
    actions = PlayerActions(CFG)
    down = [sig(head_y=y) for y in (400, 500, 545)]
    near_edge = [sig(head_y=y) for y in (535, 530, 538)]
    frames = run(actions, [sig(head_y=300)] * 5 + down + near_edge)
    assert not frames[6].ducked
    assert frames[7].ducked
    assert all(f.ducked for f in frames[8:])
    assert not frames[-1].aiming
    assert not run(actions, [sig(head_y=525)], t0=10)[-1].ducked


def test_duck_is_absolute_not_relative_to_standing_height():
    assert run(PlayerActions(CFG), [sig(head_y=600)] * 3)[-1].ducked
    assert not any(f.ducked for f in run(PlayerActions(CFG), [sig(head_y=520)] * 90))


def test_reload_needs_staying_ducked_and_fires_once_per_duck():
    hold = int(CFG.reload_s / DT)
    sequence = [sig()] * 5 + [sig(head_y=600)] * (hold + 10) + [sig()] * 5 + [sig(head_y=600)] * (hold // 2)
    frames = run(PlayerActions(CFG), sequence)
    reloads = [i for i, f in enumerate(frames) if f.reload]
    assert len(reloads) == 1 and abs(reloads[0] - (5 + hold)) <= 1
    assert 0.4 < frames[-1].reload_progress < 0.6


def test_rapid_fist_spam_is_blocked():
    spam = [sig(hand=FIST), sig(), sig(), sig(hand=FIST), sig()] * 6
    frames = run(PlayerActions(CFG), [sig()] * 15 + spam)
    fired = shot_frames(frames)
    assert fired[0] == 15
    assert all(b - a >= CFG.fire_cooldown_s / DT for a, b in zip(fired, fired[1:], strict=False))


def test_hand_must_stay_open_before_next_shot():
    blink = [sig()] * 15 + [sig(hand=FIST)] * 20 + [sig()] * 2 + [sig(hand=FIST)]
    assert shot_frames(run(PlayerActions(CFG), blink)) == [15]


def test_can_shoot_right_after_standing_up_with_arm_ready():
    ducked = [sig(head_y=600)] * 30
    frames = run(PlayerActions(CFG), [sig()] * 15 + ducked + [sig(), sig(hand=FIST)])
    assert shot_frames(frames) == [46]


def test_ignored_fists_report_why():
    frames = run(
        PlayerActions(CFG), [sig()] * 15 + [sig(hand=FIST), sig(), sig(), sig(), sig(), sig(), sig(hand=FIST)]
    )
    assert frames[15].shot is not None
    assert frames[21].ignored_fist == "cooldown"
    frames = run(PlayerActions(CFG), [sig()] * 3 + [sig(hand=FIST)])
    assert frames[3].ignored_fist == "aim settling"
    frames = run(PlayerActions(CFG), [sig(aim=False)] * 5 + [sig(aim=False, hand=FIST)])
    assert frames[5].ignored_fist == "not aiming"


def test_rapid_shots_lose_accuracy_and_recover():
    def fire_burst(gap_frames):
        actions = PlayerActions(CFG, seed=1)
        sequence = [sig()] * 15
        for _ in range(3):
            sequence += [sig(hand=FIST)] + [sig()] * gap_frames
        frames = run(actions, sequence)
        shots = [f.shot for f in frames if f.shot is not None]
        return [abs(np.degrees(np.arctan2(s.direction[1], s.direction[0]))) for s in shots], frames

    burst, frames = fire_burst(int(CFG.fire_cooldown_s / DT) + 1)
    assert len(burst) == 3 and burst[0] < 1e-3
    assert max(f.spread_deg for f in frames) > 5
    spaced, _ = fire_burst(int(2 * CFG.recoil_recovery_s / DT))
    assert len(spaced) == 3 and max(spaced) < 1e-3
