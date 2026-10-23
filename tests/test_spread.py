import numpy as np

from highnoon.contracts import Frame, HandObservation, Perception, PoseObservation
from highnoon.vision.spread import StageSpread

WIDTH, HEIGHT = 400, 100


def pose_at(x: float) -> PoseObservation:
    keypoints = np.full((33, 2), (x, 50.0), np.float32)
    return PoseObservation(keypoints, np.ones(33, np.float32))


def person_mask(x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[:, x0:x1] = 1.0
    return mask


def test_players_are_pushed_apart_with_their_pixels_and_hands():
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[:, 160:180] = (255, 0, 0)
    image[:, 220:240] = (0, 0, 255)
    masks = [person_mask(160, 180), person_mask(220, 260)]
    hand = HandObservation(np.full((21, 2), (205.0, 50.0), np.float32), 1.0, 1)
    perception = Perception(
        Frame(1, image, 0.0, 0), [pose_at(170), pose_at(230)], [], np.maximum(*masks), 0.0, {}, [hand], masks
    )
    stage = StageSpread(0.35)
    for _ in range(40):
        spread = stage.apply(perception)
    assert [float(p.keypoints[0, 0]) for p in spread.poses] == [130.0, 270.0]
    assert tuple(spread.frame.image[50, 125]) == (255, 0, 0)
    assert tuple(spread.frame.image[50, 265]) == (0, 0, 255)
    assert spread.mask[50, 200] == 0.0 and spread.mask[50, 125] == 1.0
    assert float(spread.hands[0].landmarks[0, 0]) == 245.0


def test_players_already_far_apart_are_not_moved():
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    masks = [person_mask(40, 60), person_mask(340, 360)]
    perception = Perception(
        Frame(1, image, 0.0, 0), [pose_at(50), pose_at(350)], [], None, 0.0, {}, [], masks
    )
    stage = StageSpread(0.35)
    for _ in range(10):
        spread = stage.apply(perception)
    assert spread is perception
