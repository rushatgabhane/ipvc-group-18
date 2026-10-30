from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.contracts import FaceObservation

# Cartoon damage drawn on the face as HP drops. Every mark is defined in a small frame spanned by
# nearby landmarks, so it follows the face when the head moves, tilts or turns.
STAGE_HP = (0.75, 0.5, 0.25)
HIT_FLASH_S = 0.25
MIN_FACE_PX = 24
ROI_PAD = 0.25
MARK_ALPHA = 0.9
SICK_ALPHA = 0.22
FLASH_ALPHA = 0.45
TURNED_AWAY_RATIO = 0.45
FACE_MESH_POINTS = 468

# (outer eye corner, inner eye corner, lower lid, cheek) for each side of the face
EYE_SIDES = ((33, 133, 145, 50), (263, 362, 374, 280))
NOSE = (6, 133, 362, 1)  # bridge, both inner eye corners, tip

SOOT = (55, 60, 70)
BRUISE, BRUISE_EDGE = (160, 60, 135), (90, 25, 75)
SHINER, SHINER_EDGE = (70, 25, 60), (20, 10, 20)
BLOOD, SCAR_EDGE = (40, 40, 215), (25, 20, 60)
PLASTER, PLASTER_EDGE, PLASTER_PAD = (150, 200, 235), (40, 70, 110), (110, 160, 205)
SICK = (150, 205, 170)
FLASH = (60, 60, 255)

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


_ARC = np.linspace(0, math.pi, 16)
# Eye frame: origin between the eye corners, u = half the eye width pointing outwards,
# v = lower lid to cheek (downwards).
CHEEK_SOOT = ellipse(0.7, 1.0, 1.1, 0.4, BLOB_JITTER)
CHEEK_SPECKS = [ellipse(x, y, 0.12, 0.05, n=8) for x, y in ((1.9, 0.8), (-0.2, 1.3), (1.6, 1.4))]
JAW_SOOT = ellipse(0.1, 2.4, 0.7, 0.24, BLOB_JITTER[::-1])
BRUISE_SHAPE = np.concatenate(
    [
        np.stack([1.35 * np.cos(_ARC), 0.12 + 0.5 * np.sin(_ARC)], axis=1),
        np.stack([1.1 * np.cos(_ARC), 0.12 + 0.18 * np.sin(_ARC)], axis=1)[::-1],
    ]
)
SHINER_OUTER = ellipse(0.0, -0.05, 1.8, 0.6)
SHINER_HOLE = ellipse(0.0, 0.0, 1.15, 0.22)
CUT = np.array([[-0.2, 0.9], [0.3, 1.2], [0.8, 0.9], [1.3, 1.2], [1.8, 0.9]])
STITCHES = [np.array([[u, 0.8], [u, 1.3]]) for u in (0.3, 0.8, 1.3)]
# Nose frame: origin on the bridge, u = half the gap between the inner eye corners, v = bridge to tip.
# The forehead is left alone because the hat covers most of it.
PLASTER_STRIPS = [strip(0.0, 0.4, 1.6, 0.3, angle) for angle in (25, -25)]
PLASTER_CENTER = ellipse(0.0, 0.4, 0.16, 0.07, n=10)


def damage_stage(hp_fraction: float) -> int:
    return sum(hp_fraction <= level for level in STAGE_HP)


def draw_face_damage(
    image: np.ndarray, face: FaceObservation, player_id: int, hp_fraction: float, hit_age: float
) -> None:
    stage = damage_stage(hp_fraction)
    flash = max(0.0, 1.0 - hit_age / HIT_FLASH_S)
    if (stage == 0 and flash == 0.0) or len(face.landmarks) < FACE_MESH_POINTS:
        return
    points = np.asarray(face.landmarks, np.float32)[:FACE_MESH_POINTS, :2]
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    face_w = float(x1 - x0)
    if face_w < MIN_FACE_PX:
        return
    height, width = image.shape[:2]
    pad = face_w * ROI_PAD
    rx0, ry0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    rx1, ry1 = min(width, int(x1 + pad) + 1), min(height, int(y1 + pad) + 1)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    roi = image[ry0:ry1, rx0:rx1]
    local = points - np.array([rx0, ry0], np.float32)
    if stage >= 3:
        tint_face(roi, local, SICK, SICK_ALPHA)
    if flash > 0.0:
        tint_face(roi, local, FLASH, FLASH_ALPHA * flash)
    if stage:
        overlay = roi.copy()
        draw_marks(overlay, local, stage, player_id, max(2, round(face_w / 70)))
        cv2.addWeighted(overlay, MARK_ALPHA, roi, 1 - MARK_ALPHA, 0, dst=roi)


def tint_face(roi: np.ndarray, points: np.ndarray, color, alpha: float) -> None:
    hull = cv2.convexHull(np.round(points).astype(np.int32))
    overlay = roi.copy()
    cv2.fillConvexPoly(overlay, hull, color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_marks(overlay: np.ndarray, points: np.ndarray, stage: int, player_id: int, line: int) -> None:
    frames = [eye_frame(points, side) for side in EYE_SIDES]
    widest = max(float(np.linalg.norm(frame[1])) for frame in frames)
    shown = [float(np.linalg.norm(frame[1])) >= TURNED_AWAY_RATIO * widest for frame in frames]
    hurt = 0 if player_id % 2 else 1
    other = 1 - hurt

    if shown[other]:
        fill(overlay, CHEEK_SOOT, frames[other], SOOT)
        for speck in CHEEK_SPECKS:
            fill(overlay, speck, frames[other], SOOT)
    if shown[hurt]:
        fill(overlay, JAW_SOOT, frames[hurt], SOOT)
    if stage >= 2 and shown[hurt]:
        fill(overlay, BRUISE_SHAPE, frames[hurt], BRUISE, BRUISE_EDGE, line)
        stroke(overlay, CUT, frames[hurt], SCAR_EDGE, line + 3)
        stroke(overlay, CUT, frames[hurt], BLOOD, line)
        for stitch in STITCHES:
            stroke(overlay, stitch, frames[hurt], SCAR_EDGE, max(1, line - 1))
    if stage >= 3:
        if shown[other]:
            outer = to_image(SHINER_OUTER, frames[other])
            hole = to_image(SHINER_HOLE, frames[other])
            cv2.fillPoly(overlay, [outer, hole], SHINER, cv2.LINE_AA)
            cv2.polylines(overlay, [outer], True, SHINER_EDGE, line, cv2.LINE_AA)
        nose = nose_frame(points)
        for piece in PLASTER_STRIPS:
            fill(overlay, piece, nose, PLASTER, PLASTER_EDGE, line)
        fill(overlay, PLASTER_CENTER, nose, PLASTER_PAD)


def eye_frame(points: np.ndarray, side: tuple[int, int, int, int]):
    outer, inner, lid, cheek = points[list(side)]
    return (outer + inner) / 2, (outer - inner) / 2, cheek - lid


def nose_frame(points: np.ndarray):
    bridge, inner_a, inner_b, tip = points[list(NOSE)]
    return bridge, (inner_b - inner_a) / 2, tip - bridge


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
