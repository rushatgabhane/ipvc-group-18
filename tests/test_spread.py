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
    stage = StageSpread(2.0)
    for _ in range(60):
        spread = stage.apply(perception)
    assert [float(p.keypoints[0, 0]) for p in spread.poses] == [140.0, 260.0]
    assert tuple(spread.frame.image[50, 135]) == (255, 0, 0)
    assert tuple(spread.frame.image[50, 255]) == (0, 0, 255)
    assert spread.mask[50, 200] == 0.0 and spread.mask[50, 135] == 1.0
    assert float(spread.hands[0].landmarks[0, 0]) == 235.0


def test_players_near_the_edges_are_not_pushed_off_screen():
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    masks = [person_mask(50, 70), person_mask(330, 350)]
    perception = Perception(
        Frame(1, image, 0.0, 0), [pose_at(60), pose_at(340)], [], None, 0.0, {}, [], masks
    )
    stage = StageSpread(2.0)
    for _ in range(60):
        spread = stage.apply(perception)
    assert [round(float(p.keypoints[0, 0])) for p in spread.poses] == [48, 352]
