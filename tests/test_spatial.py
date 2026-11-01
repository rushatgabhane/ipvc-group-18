import numpy as np

from highnoon.tracking.spatial import TORSO_M, focal_px, locate


def test_a_torso_twice_as_small_is_twice_as_far():
    near = locate(np.array([640, 360]), 200.0, (1280, 720), 70.0)
    far = locate(np.array([640, 360]), 100.0, (1280, 720), 70.0)
    assert np.isclose(far[2], 2 * near[2])
    assert np.allclose(near[:2], 0.0)


def test_a_point_projects_back_to_where_it_was_seen():
    f = focal_px(1280, 70.0)
    x, y, z = locate(np.array([1000, 200]), 150.0, (1280, 720), 70.0)
    assert np.isclose(640 + f * x / z, 1000) and np.isclose(360 + f * y / z, 200)
    assert np.isclose(f * TORSO_M / z, 150.0)
