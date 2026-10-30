from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.contracts import Player
from highnoon.render.face_damage import HIT_FLASH_S, damage_stage

# Cartoon damage for the rest of the player. The torso and the hands carry marks defined in a small
# frame spanned by their keypoints; the whole person mask gets a tint on top.
SHOULDERS, HIPS = (11, 12), (23, 24)
MIN_VISIBILITY = 0.5
MIN_TORSO_PX = 40
MIN_PALM_PX = 8
ROI_PAD = 0.3
# Same soft threshold the camera layer uses to cut the person out of the background.
MASK_LOW, MASK_HIGH = 0.45, 0.9
RIM_KERNEL = np.ones((7, 7), np.uint8)

MARK_ALPHA = 0.9
FLASH = (60, 60, 255)
SICK = (150, 205, 170)
SOOT = (55, 60, 70)
BLOOD, SCAR_EDGE = (40, 40, 215), (25, 20, 60)
PLASTER, PLASTER_EDGE = (150, 200, 235), (40, 70, 110)
BLOB_JITTER = np.array([1.0, 0.8, 1.1, 0.9, 1.15, 0.85, 1.0, 0.75, 1.1, 0.95, 1.2, 0.8, 1.0, 0.9])


def ellipse(cx: float, cy: float, rx: float, ry: float, jitter=None, n: int = 24) -> np.ndarray:
    angles = np.linspace(0, math.tau, n if jitter is None else len(jitter), endpoint=False)
    scale = 1.0 if jitter is None else jitter
    return np.stack([cx + rx * scale * np.cos(angles), cy + ry * scale * np.sin(angles)], axis=1)


def strip(cx: float, cy: float, length: float, width: float, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return np.array([[cx, cy] + along * length / 2 * i + across * width / 2 * j for i, j in corners])


def to_image(shape: np.ndarray, frame) -> np.ndarray:
    origin, u, v = frame
    return np.round(origin + shape[:, :1] * u + shape[:, 1:] * v).astype(np.int32)


def fill(overlay: np.ndarray, shape: np.ndarray, frame, color, edge=None, line: int = 0) -> None:
    polygon = to_image(shape, frame)
    cv2.fillPoly(overlay, [polygon], color, cv2.LINE_AA)
    if edge is not None:
        cv2.polylines(overlay, [polygon], True, edge, line, cv2.LINE_AA)


def stroke(overlay: np.ndarray, shape: np.ndarray, frame, color, line: int) -> None:
    cv2.polylines(overlay, [to_image(shape, frame)], False, color, line, cv2.LINE_AA)


DUST = (110, 140, 165)
DUST_ALPHA = 0.12
SICK_ALPHA = 0.22
FLASH_ALPHA = 0.4
RIM_ALPHA = 0.6
SKIN = (140, 170, 215)

# Torso frame: origin between the shoulders, u = half the shoulder width (left to right shoulder),
# v = shoulders to hips (downwards).
HOLES = [(0.35, 0.5), (-0.4, 0.3), (0.1, 0.75), (-0.15, 0.55)]
HOLES_SHOWN = (0, 1, 3, 4)
HOLE_RING = [ellipse(u, v, 0.09, 0.04, n=10) for u, v in HOLES]
HOLE_CORE = [ellipse(u, v, 0.05, 0.022, n=10) for u, v in HOLES]
TORSO_DUST = [ellipse(-0.5, 0.55, 0.35, 0.1, BLOB_JITTER), ellipse(0.5, 0.2, 0.25, 0.07, BLOB_JITTER[::-1])]
TEAR = np.array(
    [(-0.85, 0.12), (-0.6, 0.2), (-0.5, 0.14), (-0.3, 0.3), (-0.45, 0.34), (-0.4, 0.5), (-0.75, 0.42)]
)
STAIN = ellipse(0.25, 0.62, 0.3, 0.1, BLOB_JITTER)
DRIPS = [np.array([[u, 0.7], [u, v]]) for u, v in ((0.15, 0.92), (0.3, 0.85))]

# Hand frame: origin in the palm, u = half the knuckle row (index to pinky), v = wrist to middle knuckle.
WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, PINKY_KNUCKLE = 0, 5, 9, 17
HAND_SOOT = ellipse(0.1, -0.15, 0.7, 0.15, BLOB_JITTER)
BANDAGE = strip(0.0, 0.05, 2.2, 0.28, 0)
BANDAGE_WRAP = strip(0.0, 0.4, 2.0, 0.2, 6)
HAND_BLOOD = ellipse(0.4, 0.05, 0.22, 0.06, n=10)


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
    offset = np.array([x0, y0], np.float32)
    overlay = view.copy()
    if torso is not None:
        origin, u, v = torso
        draw_torso(overlay, (origin - offset, u, v), stage, max(1, round(float(np.linalg.norm(u)) / 45)))
    for origin, u, v, _ in hands:
        draw_hand(overlay, (origin - offset, u, v), stage, max(1, round(float(np.linalg.norm(u)) / 8)))
    cv2.addWeighted(overlay, MARK_ALPHA, view, 1 - MARK_ALPHA, 0, dst=view)


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
    person = np.clip((mask - MASK_LOW) / (MASK_HIGH - MASK_LOW), 0.0, 1.0).astype(np.float32)
    xs = np.arange(person.shape[1]) + x_offset
    person *= ((xs >= columns[0]) & (xs < columns[1]))[None, :]
    return person


def tint_silhouette(view: np.ndarray, person: np.ndarray, stage: int, flash: float) -> None:
    if not person.any():
        return
    if stage == 2:
        tint(view, person * DUST_ALPHA, DUST)
    if stage >= 3:
        tint(view, person * SICK_ALPHA, SICK)
        rim = np.clip(person - cv2.erode(person, RIM_KERNEL), 0.0, 1.0)
        tint(view, rim * RIM_ALPHA, BLOOD)
    if flash > 0.0:
        tint(view, person * (FLASH_ALPHA * flash), FLASH)


def tint(view: np.ndarray, alpha: np.ndarray, color) -> None:
    solid = np.empty_like(view)
    solid[:] = color
    view[:] = cv2.blendLinear(view, solid, 1.0 - alpha, alpha)


def draw_torso(overlay: np.ndarray, frame, stage: int, line: int) -> None:
    for patch in TORSO_DUST:
        fill(overlay, patch, frame, SOOT)
    if stage >= 2:
        fill(overlay, TEAR, frame, SKIN, SCAR_EDGE, line)
    if stage >= 3:
        fill(overlay, STAIN, frame, BLOOD)
        for drip in DRIPS:
            stroke(overlay, drip, frame, BLOOD, line + 1)
    for ring, core in list(zip(HOLE_RING, HOLE_CORE))[: HOLES_SHOWN[stage]]:
        fill(overlay, ring, frame, SCAR_EDGE)
        fill(overlay, core, frame, (15, 15, 15))


def draw_hand(overlay: np.ndarray, frame, stage: int, line: int) -> None:
    fill(overlay, HAND_SOOT, frame, SOOT)
    if stage >= 2:
        fill(overlay, BANDAGE, frame, PLASTER, PLASTER_EDGE, line)
    if stage >= 3:
        fill(overlay, BANDAGE_WRAP, frame, PLASTER, PLASTER_EDGE, line)
        fill(overlay, HAND_BLOOD, frame, BLOOD)
