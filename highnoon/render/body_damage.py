from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.contracts import Player
from highnoon.render.face_damage import HIT_FLASH_S, damage_stage

# Damage for the rest of the player. The torso and the hands carry marks defined in a small frame
# spanned by their keypoints; the whole person mask gets a tint on top. Marks are drawn into soft
# masks and blended into the camera image (stains darken the cloth instead of covering it).
SHOULDERS, HIPS = (11, 12), (23, 24)
MIN_VISIBILITY = 0.5
MIN_TORSO_PX = 40
MIN_PALM_PX = 8
ROI_PAD = 0.3
# Same soft threshold the camera layer uses to cut the person out of the background.
MASK_LOW, MASK_HIGH = 0.45, 0.9
MIN_FEATHER_PX = 0.8
TORSO_LINE_DIV, HAND_LINE_DIV = 45, 8  # mark line width = half the torso/palm width / this

DUST_TINT, DUST_ALPHA = (110, 140, 165), 0.08
DARK_TINT, DARK_ALPHA = (20, 15, 15), 0.12
DESATURATE = (0.0, 0.0, 0.15, 0.35)  # by damage stage
FLASH, FLASH_ALPHA = (60, 60, 255), 0.3

SOOT, BLOOD = (75, 80, 92), (25, 25, 140)
HOLE_RING, HOLE_CORE = (40, 45, 60), (8, 8, 10)
CLOTH, SEAM = (215, 225, 235), (165, 175, 190)
# (layer, colour, strength, multiply, softness) in drawing order. Multiplied layers darken what is
# underneath so the texture of the cloth shows through; the others cover it. Softness scales the blur
# of the edge: smudges are softer than holes and cuts.
COMPOSITE = (
    ("soot", SOOT, 0.55, True, 4.0),
    ("cloth", CLOTH, 0.92, False, 1.0),
    ("seam", SEAM, 0.5, True, 1.0),
    ("ring", HOLE_RING, 0.7, True, 1.0),
    ("core", HOLE_CORE, 0.95, True, 1.0),
    ("blood", BLOOD, 0.55, True, 1.5),
    ("blood_core", BLOOD, 0.5, True, 2.5),
)
BLOB_JITTER = np.array([1.0, 0.8, 1.1, 0.9, 1.15, 0.85, 1.0, 0.75, 1.1, 0.95, 1.2, 0.8, 1.0, 0.9])
_RNG = np.random.default_rng(7)


def ellipse(cx: float, cy: float, rx: float, ry: float, jitter=None, n: int = 24) -> np.ndarray:
    angles = np.linspace(0, math.tau, n if jitter is None else len(jitter), endpoint=False)
    scale = 1.0 if jitter is None else jitter
    return np.stack([cx + rx * scale * np.cos(angles), cy + ry * scale * np.sin(angles)], axis=1)


def ragged(cx: float, cy: float, rx: float, ry: float, jitter: float = 0.35, n: int = 11) -> np.ndarray:
    """An ellipse with an uneven edge, like torn cloth."""
    return ellipse(cx, cy, rx, ry, 1.0 + _RNG.uniform(-jitter, jitter, n))


def strip(cx: float, cy: float, length: float, width: float, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return np.array([[cx, cy] + along * length / 2 * i + across * width / 2 * j for i, j in corners])


def to_image(shape: np.ndarray, frame) -> np.ndarray:
    origin, u, v = frame
    return np.round(origin + shape[:, :1] * u + shape[:, 1:] * v).astype(np.int32)


# Torso frame: origin between the shoulders, u = half the shoulder width (left to right shoulder),
# v = shoulders to hips (downwards).
HOLES = [(0.35, 0.5), (-0.4, 0.3), (0.1, 0.75), (-0.15, 0.55)]
HOLES_SHOWN = (0, 1, 3, 4)
HOLE_RINGS = [ragged(u, v, 0.11, 0.05) for u, v in HOLES]
HOLE_CORES = [ragged(u, v, 0.055, 0.024, jitter=0.2) for u, v in HOLES]
HOLE_SPLITS = [
    [np.array([[u, v], [u + 0.17 * math.cos(a), v + 0.075 * math.sin(a)]]) for a in _RNG.uniform(0, math.tau, 3)]
    for u, v in HOLES
]
TORSO_DUST = [ellipse(-0.5, 0.55, 0.35, 0.1, BLOB_JITTER), ellipse(0.5, 0.2, 0.25, 0.07, BLOB_JITTER[::-1])]
STAIN = ellipse(0.25, 0.62, 0.3, 0.1, BLOB_JITTER)
STAIN_CORE = ellipse(0.25, 0.64, 0.18, 0.06, BLOB_JITTER[::-1])
DRIPS = [np.array([[u, 0.7], [u, v]]) for u, v in ((0.15, 0.92), (0.3, 0.85))]

# Hand frame: origin in the palm, u = half the knuckle row (index to pinky), v = wrist to middle knuckle.
WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, PINKY_KNUCKLE = 0, 5, 9, 17
HAND_SOOT = ellipse(0.1, -0.15, 0.7, 0.15, BLOB_JITTER)
BANDAGE = strip(0.0, 0.05, 2.2, 0.28, 0)
BANDAGE_WRAP = strip(0.0, 0.4, 2.0, 0.2, 6)
BANDAGE_SEAMS = [np.array([[u, -0.09], [u + 0.12, 0.19]]) for u in (-0.8, -0.3, 0.2, 0.7)]
HAND_BLOOD = ellipse(0.4, 0.05, 0.22, 0.06, n=10)
HAND_BLOOD_CORE = ellipse(0.4, 0.05, 0.12, 0.035, n=10)


class Marks:
    """One soft-edged mask per kind of mark, covering the region being drawn."""

    def __init__(self, shape: tuple[int, int]):
        self.layers = {name: np.zeros(shape, np.uint8) for name, *_ in COMPOSITE}

    def fill(self, name: str, shape: np.ndarray, frame) -> None:
        cv2.fillPoly(self.layers[name], [to_image(shape, frame)], 255, cv2.LINE_AA)

    def stroke(self, name: str, shape: np.ndarray, frame, width: int, closed: bool = False) -> None:
        cv2.polylines(self.layers[name], [to_image(shape, frame)], closed, 255, width, cv2.LINE_AA)


def player_columns(players: list[Player]) -> dict[int, tuple[float, float]]:
    """The slice of the frame each visible player owns, split halfway between neighbours, so the
    merged person mask can be tinted per player."""
    seen = sorted(
        ((float(player.pose.center()[0]), player.id) for player in players if player.visible and player.pose),
    )
    edges = [-np.inf] + [(a[0] + b[0]) / 2 for a, b in zip(seen, seen[1:])] + [np.inf]
    return {pid: (edges[i], edges[i + 1]) for i, (_, pid) in enumerate(seen)}


def draw_body_damage(
    image: np.ndarray,
    player: Player,
    hp_fraction: float,
    hit_age: float,
    mask: np.ndarray | None = None,
    columns: tuple[float, float] = (-np.inf, np.inf),
    floor_y: int | None = None,
) -> None:
    """Draws damage on the torso, hands and silhouette of one player. `mask` is the person mask at
    image size, `columns` the x range this player owns and `floor_y` the row where the cover wall
    starts (nothing is drawn below it)."""
    stage = damage_stage(hp_fraction)
    flash = max(0.0, 1.0 - hit_age / HIT_FLASH_S)
    if (stage == 0 and flash == 0.0) or player.pose is None:
        return
    keypoints = player.smoothed if player.smoothed is not None else player.pose.keypoints
    visible = player.pose.visibility > MIN_VISIBILITY
    torso = torso_frame(keypoints, visible)
    hands = [hand_frame(hand.landmarks) for hand in player.hands.values()]
    hands = [frame for frame in hands if frame is not None]

    roi_points = [keypoints[list(SHOULDERS + HIPS)]] + [frame[3] for frame in hands]
    roi = region(image, np.concatenate(roi_points), floor_y)
    if roi is None:
        return
    (x0, y0, x1, y1), view = roi
    if mask is not None:
        person = person_alpha(mask[y0:y1, x0:x1], columns, x0)
        tint_silhouette(view, person, stage, flash)

    if stage == 0:
        return
    parts = [(frame[3], frame[:3], draw_hand, HAND_LINE_DIV) for frame in hands]
    if torso is not None:
        parts.append((keypoints[list(SHOULDERS + HIPS)], torso, draw_torso, TORSO_LINE_DIV))
    for points, (origin, u, v), draw, line_div in parts:
        sub = region(image, points, floor_y)
        if sub is None:
            continue
        (sx0, sy0, _, _), part = sub
        line = max(1, round(float(np.linalg.norm(u)) / line_div))
        marks = Marks(part.shape[:2])
        draw(marks, (origin - np.array([sx0, sy0], np.float32), u, v), stage, line)
        composite(part, marks, max(MIN_FEATHER_PX, line * 0.5))


def torso_frame(keypoints: np.ndarray, visible: np.ndarray):
    if not visible[list(SHOULDERS + HIPS)].all():
        return None
    left, right = keypoints[list(SHOULDERS)]
    shoulders = (left + right) / 2
    hips = keypoints[list(HIPS)].mean(axis=0)
    u, v = (right - left) / 2, hips - shoulders
    if float(np.linalg.norm(u)) * 2 < MIN_TORSO_PX:
        return None
    return shoulders, u, v


def hand_frame(landmarks: np.ndarray):
    if len(landmarks) <= PINKY_KNUCKLE:
        return None
    points = np.asarray(landmarks, np.float32)[:, :2]
    wrist, index, middle, pinky = points[[WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, PINKY_KNUCKLE]]
    u = (index - pinky) / 2
    if float(np.linalg.norm(u)) * 2 < MIN_PALM_PX:
        return None
    return (wrist + index + middle + pinky) / 4, u, middle - wrist, points


def region(image: np.ndarray, points: np.ndarray, floor_y: int | None):
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    pad = max(x1 - x0, y1 - y0) * ROI_PAD
    height, width = image.shape[:2]
    bottom = height if floor_y is None else min(height, floor_y)
    rx0, ry0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    rx1, ry1 = min(width, int(x1 + pad) + 1), min(bottom, int(y1 + pad) + 1)
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    return (rx0, ry0, rx1, ry1), image[ry0:ry1, rx0:rx1]


def person_alpha(mask: np.ndarray, columns: tuple[float, float], x_offset: int) -> np.ndarray:
    gain = 1.0 / (MASK_HIGH - MASK_LOW)
    person = cv2.addWeighted(mask, gain, mask, 0.0, -MASK_LOW * gain)
    cv2.threshold(person, 1.0, 1.0, cv2.THRESH_TRUNC, dst=person)
    cv2.threshold(person, 0.0, 0.0, cv2.THRESH_TOZERO, dst=person)
    xs = np.arange(person.shape[1]) + x_offset
    person *= ((xs >= columns[0]) & (xs < columns[1]))[None, :].astype(np.float32)
    return person


def tint_silhouette(view: np.ndarray, person: np.ndarray, stage: int, flash: float) -> None:
    """Drains the colour from the player as HP drops, darkens them at the lowest stage, and flashes
    red on a fresh hit. Only the player's own pixels change, never the background."""
    if not person.any():
        return
    if DESATURATE[stage]:
        desaturate(view, person * DESATURATE[stage])
    layers = [(FLASH, FLASH_ALPHA * flash)] if flash > 0.0 else []
    if stage >= 2:
        layers.append((DUST_TINT, DUST_ALPHA))
    if stage >= 3:
        layers.append((DARK_TINT, DARK_ALPHA))
    if layers:
        tint(view, person, *combine(layers))


def combine(layers: list[tuple[tuple[int, int, int], float]]) -> tuple[tuple[float, ...], float]:
    """Several constant-colour tints applied in turn, as one colour and one strength."""
    color, clear = np.zeros(3), 1.0
    for tone, alpha in layers:
        color = color * (1 - alpha) + np.array(tone) * alpha
        clear *= 1 - alpha
    return tuple(color / (1 - clear)), 1 - clear


def desaturate(view: np.ndarray, alpha: np.ndarray) -> None:
    gray = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.blendLinear(view, gray, 1.0 - alpha, alpha, dst=view)


def tint(view: np.ndarray, person: np.ndarray, color, strength: float = 1.0) -> None:
    alpha = person * strength
    solid = np.empty_like(view)
    solid[:] = color
    cv2.blendLinear(view, solid, 1.0 - alpha, alpha, dst=view)


def composite(view: np.ndarray, marks: Marks, feather: float) -> None:
    height, width = view.shape[:2]
    for name, color, strength, multiply, softness in COMPOSITE:
        layer = marks.layers[name]
        sigma = feather * softness
        pad = int(sigma * 3) + 1
        x, y, w, h = cv2.boundingRect(layer)
        if w == 0 or h == 0:
            continue
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
        patch = view[y0:y1, x0:x1]
        alpha = cv2.GaussianBlur(layer[y0:y1, x0:x1], (0, 0), sigma).astype(np.float32)
        alpha *= strength / 255.0
        solid = np.empty_like(patch)
        solid[:] = color
        # Multiplying by the colour is the same as blending towards (pixel x colour).
        target = cv2.multiply(patch, solid, scale=1 / 255) if multiply else solid
        cv2.blendLinear(patch, target, 1.0 - alpha, alpha, dst=patch)


def draw_torso(marks: Marks, frame, stage: int, line: int) -> None:
    for patch in TORSO_DUST:
        marks.fill("soot", patch, frame)
    if stage >= 3:
        marks.fill("blood", STAIN, frame)
        marks.fill("blood_core", STAIN_CORE, frame)
        for drip in DRIPS:
            marks.stroke("blood", drip, frame, line + 1)
    for i in range(HOLES_SHOWN[stage]):
        marks.fill("ring", HOLE_RINGS[i], frame)
        for split in HOLE_SPLITS[i]:
            marks.stroke("ring", split, frame, 1)
        marks.fill("core", HOLE_CORES[i], frame)


def draw_hand(marks: Marks, frame, stage: int, line: int) -> None:
    marks.fill("soot", HAND_SOOT, frame)
    if stage >= 2:
        marks.fill("cloth", BANDAGE, frame)
        marks.stroke("ring", BANDAGE, frame, line, closed=True)
        for seam in BANDAGE_SEAMS:
            marks.stroke("seam", seam, frame, 1)
    if stage >= 3:
        marks.fill("cloth", BANDAGE_WRAP, frame)
        marks.stroke("ring", BANDAGE_WRAP, frame, line, closed=True)
        marks.fill("blood", HAND_BLOOD, frame)
        marks.fill("blood_core", HAND_BLOOD_CORE, frame)
