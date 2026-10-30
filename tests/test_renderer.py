import math

import numpy as np

from highnoon.config import DisplayConfig, GameConfig
from highnoon.contracts import FaceObservation, Frame, Perception, Player
from highnoon.game.actions import PlayerActions
from highnoon.game.entities import PlayerState
from highnoon.game.game import Game
from highnoon.render import renderer as renderer_module
from highnoon.render.face_damage import UV
from highnoon.render.face_fx import FaceEffectRenderer
from highnoon.render.renderer import Renderer, inset_box

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


def hat_rise() -> int:
    """How far the hat reaches above the top of the face."""
    canvas = np.zeros((2000, 2000, 3), np.uint8)
    FaceEffectRenderer().draw(canvas, make_face(1000.0))
    return 1000 - int(np.flatnonzero(canvas.any(axis=(1, 2))).min())


def test_a_face_behind_the_wall_shows_nothing():
    # Low enough that the top of the hat, which rises above the face, is behind the wall too.
    top = WALL_TOP + hat_rise() + 1
    assert top < HEIGHT, "the face must still be on screen for this test to mean anything"
    assert np.array_equal(picture(make_face(top)), picture(None))


def test_a_face_crossing_the_wall_is_cut_off_at_its_top_edge():
    with_face, without = picture(make_face(WALL_TOP - 100.0)), picture(None)
    assert np.any(with_face[:WALL_TOP] != without[:WALL_TOP])
    assert np.array_equal(with_face[WALL_TOP:], without[WALL_TOP:])


CAMERA_COLOR = (10, 200, 30)


def camera_picture(show_debug: bool = False) -> np.ndarray:
    cfg = GameConfig()
    game = Game(cfg, (WIDTH, HEIGHT))
    renderer = Renderer(DisplayConfig(show_debug=show_debug))
    renderer.world.draw = renderer.hud.draw = lambda *args: None
    image = np.full((HEIGHT, WIDTH, 3), CAMERA_COLOR, np.uint8)
    perception = Perception(Frame(0, image, 0.0, 0), [], [], None, 0.0)
    return renderer.render(perception, [], game, None)


def test_the_inset_is_small_and_on_the_wall_between_the_player_cards():
    x0, y0, x1, y1 = inset_box(WIDTH, HEIGHT, WALL_TOP)
    card_w, card_margin = 310, 16
    assert x0 > card_margin + card_w and x1 < WIDTH - card_margin - card_w
    assert (x0 + x1) // 2 == WIDTH // 2
    assert WALL_TOP < y0 and y1 < HEIGHT
    assert (x1 - x0) * (y1 - y0) < 0.03 * WIDTH * HEIGHT


def test_the_inset_shows_the_raw_camera_over_the_wall():
    x0, y0, x1, y1 = inset_box(WIDTH, HEIGHT, WALL_TOP)
    image = camera_picture()
    assert np.all(image[y0:y1, x0:x1] == CAMERA_COLOR)
    assert not np.all(image[y0, x0 - 10] == CAMERA_COLOR)


def test_the_inset_makes_way_for_the_debug_stats(monkeypatch):
    monkeypatch.setattr(renderer_module, "draw_stats", lambda *args: None)
    monkeypatch.setattr(renderer_module, "draw_signal_debug", lambda *args: None)
    x0, y0, x1, y1 = inset_box(WIDTH, HEIGHT, WALL_TOP)
    assert not np.all(camera_picture(show_debug=True)[y0:y1, x0:x1] == CAMERA_COLOR)


def test_hat_is_drawn_for_a_face_without_eye_landmarks():
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    face = make_face(300.0)
    FaceEffectRenderer().draw(canvas, FaceObservation(face.landmarks[:10], face.bbox))
    assert canvas.any()
