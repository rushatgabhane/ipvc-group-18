import numpy as np

from highnoon.core import evaluation as ev


def test_runs_finds_true_stretches():
    assert ev.runs(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]


def test_noise_level_measures_jitter_and_ignores_smooth_motion():
    rng = np.random.default_rng(0)
    t = np.arange(600) / 30
    motion = 200 * np.sin(t)  # a slow sway
    noisy = motion + rng.normal(0, 3.0, len(t))
    valid = np.ones(len(t), bool)
    assert 2.5 < ev.noise_level(noisy, valid) < 3.5
    assert ev.noise_level(motion, valid) < 0.1


def test_filter_lag_finds_a_known_delay():
    rng = np.random.default_rng(1)
    raw = np.cumsum(rng.normal(0, 5, 900))
    delayed = np.concatenate([np.full(4, raw[0]), raw[:-4]])
    assert np.isclose(ev.filter_lag_s(raw, delayed, np.ones(900, bool), 1 / 30), 4 / 30)


def test_gaps_report_duration_and_how_long_reidentification_took():
    t = np.arange(20) * 0.1
    visible = np.array([1] * 5 + [0] * 10 + [1] * 5, bool)
    waiting = np.zeros(20, bool)
    waiting[12:15] = True  # someone was in view without an id for the last 0.3 s of the gap
    (gap,) = ev.gaps(t, visible, waiting, player=1)
    assert gap.recovered and np.isclose(gap.duration_s, 1.0) and np.isclose(gap.reid_wait_s, 0.3)


def test_position_jumps_flag_a_teleporting_id():
    t = np.arange(5) * 0.033
    x = np.array([300, 305, 900, 905, 910.0])
    assert ev.position_jumps(t, x, np.ones(5, bool), threshold=200) == [t[2]]


def test_disagreements_ignore_single_bad_frames():
    t = np.arange(30) / 30
    agree = np.ones(30, bool)
    agree[5] = False
    agree[10:25] = False
    found = ev.disagreements(t, agree, np.ones(30, bool), player=1, min_s=0.3)
    assert len(found) == 1 and np.isclose(found[0].start_t, 10 / 30)
