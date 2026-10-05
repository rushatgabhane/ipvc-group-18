import numpy as np

from highnoon.core.filters import OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    x = np.array([[10.0, 20.0]])
    np.testing.assert_allclose(f(x, 0.0), x)


def test_reduces_jitter_on_static_signal():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw, out = [], []
    for i in range(300):
        x = np.array([[100.0, 100.0]]) + rng.normal(0, 2.0, (1, 2))
        raw.append(x)
        out.append(f(x, i / 30))
    raw_std = np.std(np.concatenate(raw[30:]), axis=0)
    out_std = np.std(np.concatenate(out[30:]), axis=0)
    assert np.all(out_std < raw_std * 0.5)


def test_beta_reduces_lag_on_fast_motion():
    def lag(beta):
        f = OneEuroFilter(min_cutoff=1.0, beta=beta)
        for i in range(60):  # ramp at 600 px/s
            y = f(np.array([[i * 20.0, 0.0]]), i / 30)
        return 59 * 20.0 - y[0, 0]

    assert lag(0.05) < lag(0.0) * 0.5


def test_mask_freezes_unreliable_points():
    f = OneEuroFilter()
    f(np.array([[0.0, 0.0], [0.0, 0.0]]), 0.0)
    out = f(np.array([[50.0, 50.0], [50.0, 50.0]]), 1 / 30, mask=np.array([True, False]))
    assert out[0, 0] > 0
    np.testing.assert_allclose(out[1], [0.0, 0.0])
