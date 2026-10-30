import math

import numpy as np

from highnoon.config import DisplayConfig, GameConfig
from highnoon.contracts import FaceObservation, Frame, Perception, Player
from highnoon.game.actions import PlayerActions
from highnoon.game.entities import PlayerState
from highnoon.game.game import Game
from highnoon.render.face_damage import UV
from highnoon.render.renderer import Renderer

WIDTH, HEIGHT = 1280, 720
WALL_TOP = int(HEIGHT * (1 - GameConfig().cover_height))


def make_face(top: float) -> FaceObservation:
    landmarks = np.zeros((478, 2), np.float32)
    landmarks[:468] = UV * 200 + (540, top)
    landmarks[468], landmarks[473] = landmarks[33], landmarks[263]
    points = landmarks[:468]
    return FaceObservation(landmarks, np.concatenate([points.min(axis=0), points.max(axis=0)]))


def picture(face: FaceObservation | None) -> np.ndarray:
    cfg = GameConfig()
    game = Game(cfg, (WIDTH, HEIGHT))
    game.states[1] = PlayerState(PlayerActions(cfg), 10, cfg.max_ammo, last_hit_t=-math.inf)
    renderer = Renderer(DisplayConfig())
    renderer.world.draw = renderer.hud.draw = lambda *args: None  # only the face effects are under test
    image = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    perception = Perception(Frame(0, image, 0.0, 0), [], [], None, 0.0)
    return renderer.render(perception, [Player(id=1, face=face, visible=True)], game, None)


def test_a_face_above_the_wall_gets_its_effects():
    assert np.any(picture(make_face(WALL_TOP - 300.0)) != picture(None))


def test_a_face_behind_the_wall_shows_nothing():
    # Low enough that the top of the hat, which rises above the face, is behind the wall too.
    assert np.array_equal(picture(make_face(WALL_TOP + 130.0)), picture(None))


def test_a_face_crossing_the_wall_is_cut_off_at_its_top_edge():
    with_face, without = picture(make_face(WALL_TOP - 100.0)), picture(None)
    assert np.any(with_face[:WALL_TOP] != without[:WALL_TOP])
    assert np.array_equal(with_face[WALL_TOP:], without[WALL_TOP:])
