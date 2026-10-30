from __future__ import annotations

from functools import cache, lru_cache

import cv2
import numpy as np

from highnoon.contracts import FaceObservation
from highnoon.render.assets import ASSET_DIR

# Injuries painted once into a flat texture of the face (MediaPipe's canonical UV layout), then warped
# onto the tracked mesh triangle by triangle. The texture multiplies the camera image, so the skin
# keeps its own pores, shadows and lighting instead of being covered by a sticker.
STAGE_HP = (0.75, 0.5, 0.25)
HIT_FLASH_S = 0.25
MIN_FACE_PX = 24
ROI_PAD = 0.05
FACE_MESH_POINTS = 468

# The warp runs at this face width at most and is scaled up to fit, so its cost does not grow with
# the face. The marks are soft, so nothing is lost.
WORK_FACE_PX = 110
PALLOR = 0.2
FLASH_TINT = np.array([0.55, 0.55, 1.0], np.float32)
FLASH_STRENGTH = 0.4
BASE_SIZE = 512
TEXTURE_SIZE = 256
SKIN_FEATHER = 0.035

with np.load(ASSET_DIR / "face_mesh.npz") as _mesh:
    UV, TRIANGLES = _mesh["uv"], _mesh["triangles"]

# The last warp per player. The game draws at 60 Hz but the landmarks only change per camera frame.
_warps: dict[int, tuple] = {}

# Landmarks the injuries are anchored to. Player 1's bruise is on the left of the image; Player 2 gets
# the texture mirrored, which works because the UV layout is symmetric.
HURT_LID, HURT_CHEEK = 145, 50
OTHER_CHEEK, OTHER_OUTER_EYE, OTHER_INNER_EYE = 280, 263, 362
HURT_JAW, HURT_NOSTRIL, LIP_LOW = 172, 98, 14

# Multipliers for (blue, green, red); 1 leaves the skin unchanged.
GRIME = (0.66, 0.7, 0.74)
GRAZE = (0.6, 0.58, 0.76)
BRUISE_RIM = (0.72, 0.9, 0.93)
BRUISE_MID = (0.72, 0.58, 0.68)
BRUISE_CORE = (0.62, 0.44, 0.5)
SHINER_MID = (0.6, 0.45, 0.5)
SHINER_CORE = (0.42, 0.28, 0.33)
SWELLING = (0.8, 0.78, 0.96)
BLOOD = (0.2, 0.17, 0.66)
WOUND = (0.12, 0.08, 0.3)


def damage_stage(hp_fraction: float) -> int:
    return sum(hp_fraction <= level for level in STAGE_HP)


def preload_face_damage() -> None:
    """Paints every texture up front, so the first hit of a game does not stall a frame."""
    for stage in range(len(STAGE_HP) + 1):
        for mirrored in (False, True):
            injury_texture(stage, mirrored)


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
    last = _warps.get(player_id)
    if last is not None and last[0] is face.landmarks and last[1] == roi.shape:
        map_u, map_v = last[2], last[3]
    else:
        scale = min(1.0, WORK_FACE_PX / face_w)
        work_h, work_w = max(1, round(roi.shape[0] * scale)), max(1, round(roi.shape[1] * scale))
        map_u, map_v = uv_maps((points - np.array([rx0, ry0], np.float32)) * scale, (work_h, work_w))
        _warps[player_id] = (face.landmarks, roi.shape, map_u, map_v)
    texture = cv2.remap(
        injury_texture(stage, player_id % 2 == 0),
        map_u * (TEXTURE_SIZE - 1),
        map_v * (TEXTURE_SIZE - 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )
    if flash > 0.0:
        skin = texture[:, :, 3:].astype(np.float32) / 255
        tinted = texture[:, :, :3] * (1 - skin * FLASH_STRENGTH * flash * (1 - FLASH_TINT))
        texture[:, :, :3] = tinted.astype(np.uint8)
    full_size = (roi.shape[1], roi.shape[0])
    if stage >= 3:
        skin = cv2.resize(texture[:, :, 3], full_size, interpolation=cv2.INTER_LINEAR)
        roi[:] = drain_colour(roi, (skin[:, :, None] * PALLOR).astype(np.uint16))
    shade = cv2.resize(texture[:, :, :3], full_size, interpolation=cv2.INTER_LINEAR)
    cv2.multiply(roi, shade, dst=roi, scale=1 / 255)


def drain_colour(roi: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Blends towards gray, with amount in 0..255."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)[:, :, None].astype(np.uint16)
    return ((roi * (255 - amount) + gray * amount) // 255).astype(np.uint8)


def uv_maps(points: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """For every ROI pixel, where in the texture it comes from. Pixels off the mesh get -1."""
    corners = points[TRIANGLES]
    origin = corners[:, 0]
    a, b = corners[:, 1] - origin, corners[:, 2] - origin
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    # The mesh is wound consistently, so a flipped or vanishing triangle is turned away from the camera.
    facing = np.flatnonzero(det > 1e-3)
    owner = np.zeros(shape, np.int16)
    pixels = np.round(corners).astype(np.int32)
    for index in facing.tolist():
        cv2.fillConvexPoly(owner, pixels[index], index + 1)
    rows, cols = np.nonzero(owner)
    tri = owner[rows, cols] - 1

    # Each triangle's affine map from screen to texture.
    uv_corners = UV[TRIANGLES[tri]]
    du, dv = uv_corners[:, 1] - uv_corners[:, 0], uv_corners[:, 2] - uv_corners[:, 0]
    dx, dy = cols - origin[tri, 0], rows - origin[tri, 1]
    s = (dx * b[tri, 1] - dy * b[tri, 0]) / det[tri]
    r = (dy * a[tri, 0] - dx * a[tri, 1]) / det[tri]
    mapped = uv_corners[:, 0] + s[:, None] * du + r[:, None] * dv

    map_u, map_v = np.full(shape, -1.0, np.float32), np.full(shape, -1.0, np.float32)
    map_u[rows, cols], map_v[rows, cols] = mapped[:, 0], mapped[:, 1]
    return map_u, map_v


@cache
def injury_texture(stage: int, mirrored: bool) -> np.ndarray:
    """BGR multipliers plus a skin mask in alpha, all stored as 0..255."""
    skin = skin_mask()
    faded = 255 - (255 - paint_stage(stage).astype(np.float32)) * (skin[:, :, None] / 255)
    texture = np.dstack([faded.astype(np.uint8), skin])
    if mirrored:
        texture = texture[:, ::-1]
    return cv2.resize(texture, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_AREA)


@cache
def paint_stage(stage: int) -> np.ndarray:
    if stage == 0:
        return np.full((BASE_SIZE, BASE_SIZE, 3), 255, np.uint8)
    shade = paint_stage(stage - 1).astype(np.float32) / 255
    (paint_grazes, paint_bruise_and_cut, paint_black_eye_and_bleed)[stage - 1](shade)
    return (np.clip(shade, 0, 1) * 255).astype(np.uint8)


def paint_grazes(shade: np.ndarray) -> None:
    blot(shade, UV[OTHER_CHEEK] + (-0.04, 0.03), (0.14, 0.06), GRAZE, grain=0.5, seed=1, streaks=True)
    jaw = UV[HURT_JAW] * 0.65 + UV[HURT_CHEEK] * 0.35
    blot(shade, jaw, (0.12, 0.07), GRIME, grain=0.6, seed=3)
    blot(shade, UV[HURT_CHEEK] + (0.01, 0.1), (0.07, 0.04), GRIME, grain=0.7, seed=4)


def paint_bruise_and_cut(shade: np.ndarray) -> None:
    under_eye = UV[HURT_LID] + (0.0, 0.04)
    blot(shade, under_eye, (0.17, 0.11), BRUISE_RIM, grain=0.3, seed=5)
    blot(shade, under_eye, (0.135, 0.08), BRUISE_MID, grain=0.35, seed=6)
    blot(shade, under_eye - (0.0, 0.012), (0.1, 0.05), BRUISE_CORE, grain=0.35, seed=7)

    # A straight cut along the cheekbone, in a patch of swollen skin.
    start = UV[HURT_CHEEK] + (0.0, -0.01)
    cut = [start, start + (0.025, 0.012), start + (0.05, 0.016), start + (0.07, 0.026)]
    blot(shade, start + (0.035, 0.012), (0.09, 0.05), SWELLING, grain=0.2, seed=8)
    line(shade, cut, 0.0032, BLOOD, seed=9, taper=True)
    line(shade, cut, 0.0012, WOUND, seed=10, taper=True)


def paint_black_eye_and_bleed(shade: np.ndarray) -> None:
    centre = (UV[OTHER_OUTER_EYE] + UV[OTHER_INNER_EYE]) / 2
    blot(shade, centre, (0.19, 0.14), BRUISE_RIM, grain=0.3, seed=15)
    blot(shade, centre, (0.155, 0.105), SHINER_MID, grain=0.3, seed=16)
    blot(shade, centre, (0.125, 0.075), SHINER_CORE, grain=0.3, seed=17)

    # Nosebleed from one nostril, across the lips and down the chin.
    nostril = UV[HURT_NOSTRIL] + (0.014, 0.012)
    blot(shade, nostril + (0.0, 0.012), (0.024, 0.024), BLOOD, grain=0.3, seed=20)
    line(shade, drip(nostril, 0.26, seed=19), 0.0075, BLOOD, seed=21, taper=True)

    # Split lower lip.
    split = UV[LIP_LOW] + (-0.035, 0.008)
    tear = [split, split + (0.004, 0.012), split + (0.0, 0.026)]
    line(shade, tear, 0.006, BLOOD, seed=24)
    line(shade, tear, 0.0024, WOUND, seed=25)


def blot(shade: np.ndarray, centre, radius, tone, grain: float, seed: int, streaks: bool = False) -> None:
    """A soft patch with an uneven edge and mottled inside, like real discolouration."""
    ys, xs = np.mgrid[0:BASE_SIZE, 0:BASE_SIZE].astype(np.float32) / BASE_SIZE
    distance = np.hypot((xs - centre[0]) / radius[0], (ys - centre[1]) / radius[1])
    mask = np.clip((1.0 - (distance + (fractal_noise(seed) - 0.5) * grain)) / 0.9, 0, 1)
    mask = mask * mask * (3 - 2 * mask) * (1 - grain * 0.5 * fractal_noise(seed + 100))
    if streaks:
        mask = mask * scrape_lines(seed)
    apply_tone(shade, mask, tone)


def scrape_lines(seed: int) -> np.ndarray:
    """Fine parallel scratches, stretched along the direction of the scrape."""
    rows = np.random.default_rng(seed).random((96, 6), dtype=np.float32)
    return np.clip(1.8 * cv2.resize(rows, (BASE_SIZE, BASE_SIZE), interpolation=cv2.INTER_CUBIC) - 0.3, 0, 1)


def line(shade: np.ndarray, path, width: float, tone, seed: int, taper: bool = False) -> None:
    """A stroke along a path through the texture, softened so it sits in the skin."""
    path = np.round(np.asarray(path, np.float32) * BASE_SIZE).astype(int)
    mask = np.zeros((BASE_SIZE, BASE_SIZE), np.float32)
    weight = width * BASE_SIZE
    for index, (start, end) in enumerate(zip(path[:-1], path[1:], strict=True)):
        scale = 1 - 0.6 * index / max(1, len(path) - 2) if taper else 1.0
        thickness = max(1, round(2 * weight * scale))
        cv2.line(mask, tuple(start.tolist()), tuple(end.tolist()), scale, thickness, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), max(0.8, weight * 0.45))
    apply_tone(shade, np.clip(mask * 1.3, 0, 1) * (0.75 + 0.25 * fractal_noise(seed)), tone)


def drip(start, length: float, seed: int, steps: int = 8) -> list:
    """Blood running down the face, drifting gently rather than zigzagging."""
    noise = np.random.default_rng(seed).normal(0, 0.008, steps)
    drift = np.convolve(noise, np.ones(3) / 3, mode="same")
    return [np.asarray(start) + (drift[i] - drift[0], length * i / (steps - 1)) for i in range(steps)]


def apply_tone(shade: np.ndarray, mask: np.ndarray, tone) -> None:
    shade *= 1 - mask[:, :, None] * (1 - np.array(tone, np.float32))


@cache
def fractal_noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros((BASE_SIZE, BASE_SIZE), np.float32)
    amplitude, weight = 1.0, 0.0
    for cells in (6, 12, 24, 48, 96):
        layer = rng.random((cells, cells), dtype=np.float32)
        total += amplitude * cv2.resize(layer, (BASE_SIZE, BASE_SIZE), interpolation=cv2.INTER_CUBIC)
        weight += amplitude
        amplitude *= 0.55
    total /= weight
    return np.clip((total - total.min()) / np.ptp(total), 0, 1)


@lru_cache(maxsize=1)
def skin_mask() -> np.ndarray:
    """255 on the skin, fading out towards the edge of the mesh so no outline shows."""
    mask = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    uv_pixels = np.round(UV * (BASE_SIZE - 1)).astype(np.int32)
    for triangle in TRIANGLES:
        cv2.fillConvexPoly(mask, uv_pixels[triangle], 255)
    feather = int(SKIN_FEATHER * BASE_SIZE)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1)))
    return cv2.GaussianBlur(inner, (0, 0), feather / 1.5)
