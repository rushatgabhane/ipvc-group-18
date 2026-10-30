from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np

from highnoon.contracts import FaceObservation
from highnoon.render.assets import ASSET_DIR, SPRITE_DIR

STAGE_HP = (0.75, 0.5, 0.25)
HIT_FLASH_S = 0.25
MIN_FACE_PX = 24
ROI_PAD = 0.05
FACE_MESH_POINTS = 468
WORK_FACE_PX = 110
PALLOR = 0.2
FLASH_TINT = np.array([0.55, 0.55, 1.0], np.float32)
FLASH_STRENGTH = 0.4
BASE_SIZE = 512
TEXTURE_SIZE = 256
SKIN_FEATHER = 0.025
FACE_TINT_SHRINK = 0.92
FACE_TINT_FEATHER = 0.05
PAINTED_LEVEL = 3
WHITE = (255, 255, 255)
OCCLUDERS = 250
TURNED = 0.12
NOSE_TIP = 1
FACE_SIDES = (234, 454)

DECAL_DIR = SPRITE_DIR / "face"

with np.load(ASSET_DIR / "face_mesh.npz") as _mesh:
    UV, TRIANGLES = _mesh["uv"], _mesh["triangles"]
NEAR = np.arange(len(TRIANGLES)) >= len(TRIANGLES) - OCCLUDERS


@dataclass(frozen=True)
class Decal:
    image: str
    anchor: tuple[int, ...]
    offset: tuple[float, float]
    size: tuple[float, float]
    stage: int
    strength: float = 1.0
    angle_deg: float = 0.0


# In UV units. Player 2 gets the texture mirrored.
DECALS = (
    Decal("blood_spray.png", (280,), (0.0, 0.03), (0.17, 0.12), 1),
    Decal("shiner.png", (33, 133), (0.0, 0.012), (0.22, 0.18), 2),
    Decal("blood_spray.png", (50,), (0.02, -0.01), (0.1, 0.07), 2, 1.0, -20),
    Decal("blood_spray.png", (98,), (0.005, 0.045), (0.13, 0.05), 3, 1.6, 115),
)


def damage_stage(hp_fraction: float) -> int:
    return sum(hp_fraction <= level for level in STAGE_HP)


def preload_face_damage() -> None:
    for stage in range(len(STAGE_HP) + 1):
        for mirrored in (False, True):
            injury_texture(stage, mirrored)


@dataclass(slots=True)
class Warp:
    landmarks: np.ndarray
    shape: tuple[int, ...]
    key: tuple[int, bool]
    points: np.ndarray
    work_shape: tuple[int, int]
    shade: np.ndarray | None
    _skin: np.ndarray | None = None
    _pallor: np.ndarray | None = None

    def skin(self) -> np.ndarray:
        if self._skin is None:
            self._skin = face_weight(self.points, self.work_shape)
        return self._skin

    def pallor(self) -> np.ndarray:
        if self._pallor is None:
            full_size = (self.shape[1], self.shape[0])
            self._pallor = cv2.resize(self.skin(), full_size, interpolation=cv2.INTER_LINEAR) * PALLOR
        return self._pallor


# Reused for the redraws between camera frames.
_warps: dict[int, Warp] = {}


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
    key = (stage, player_id % 2 == 0)

    warp = _warps.get(player_id)
    if warp is None or warp.landmarks is not face.landmarks or warp.shape != roi.shape or warp.key != key:
        local = points - np.array([rx0, ry0], np.float32)
        warp = _warps[player_id] = build_warp(face.landmarks, local, roi.shape, key, face_w)

    if warp.shade is not None:
        cv2.multiply(roi, warp.shade, dst=roi, scale=1 / 255)
    if flash > 0.0:
        redden = 1.0 - (warp.skin() * (FLASH_STRENGTH * flash))[:, :, None] * (1.0 - FLASH_TINT)
        tint = cv2.resize((redden * 255).astype(np.uint8), (roi.shape[1], roi.shape[0]))
        cv2.multiply(roi, tint, dst=roi, scale=1 / 255)
    if stage >= 3:
        weight = warp.pallor()
        gray = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        roi[:] = cv2.blendLinear(gray, roi, weight, 1.0 - weight)


def build_warp(
    landmarks: np.ndarray, points: np.ndarray, shape: tuple[int, ...], key: tuple[int, bool], face_w: float
) -> Warp:
    scale = min(1.0, WORK_FACE_PX / face_w)
    work_shape = (max(1, round(shape[0] * scale)), max(1, round(shape[1] * scale)))
    points = points * scale
    texture, painted = injury_texture(*key)
    map_u, map_v = uv_maps(points, work_shape, painted)
    shade = None
    if map_u is not None:
        warped = cv2.remap(
            texture, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE
        )
        shade = cv2.resize(warped, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return Warp(landmarks, shape, key, points, work_shape, shade)


def uv_maps(
    points: np.ndarray, shape: tuple[int, int], painted: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    corners = points[TRIANGLES]
    origin = corners[:, 0]
    a, b = corners[:, 1] - origin, corners[:, 2] - origin
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    # Triangles are stored farthest first and flipped ones face away. On a turned head the nearest ones
    # (the nose and the middle of the face) cover painted ones behind them, so they are drawn over them.
    facing = np.flatnonzero(det > 1e-3)
    flags = painted[facing]
    if not flags.any():
        return None, None
    if is_turned(points):
        facing = facing[np.argmax(flags) :]
        facing = facing[painted[facing] | NEAR[facing]]
    else:
        facing = facing[flags]
    ids = np.where(painted[facing], facing + 1, 0).tolist()
    owner = np.zeros(shape, np.int16)
    fill = cv2.fillConvexPoly
    for triangle, value in zip(np.round(corners[facing]).astype(np.int32), ids, strict=True):
        fill(owner, triangle, value)
    rows, cols = np.nonzero(owner)
    tri = owner[rows, cols] - 1

    uv_corners = UV[TRIANGLES[tri]] * (TEXTURE_SIZE - 1)
    du, dv = uv_corners[:, 1] - uv_corners[:, 0], uv_corners[:, 2] - uv_corners[:, 0]
    dx, dy = cols - origin[tri, 0], rows - origin[tri, 1]
    s = (dx * b[tri, 1] - dy * b[tri, 0]) / det[tri]
    r = (dy * a[tri, 0] - dx * a[tri, 1]) / det[tri]
    mapped = uv_corners[:, 0] + s[:, None] * du + r[:, None] * dv

    map_u, map_v = np.full(shape, -1.0, np.float32), np.full(shape, -1.0, np.float32)
    map_u[rows, cols], map_v[rows, cols] = mapped[:, 0], mapped[:, 1]
    return map_u, map_v


def is_turned(points: np.ndarray) -> bool:
    left, right = points[FACE_SIDES[0], 0], points[FACE_SIDES[1], 0]
    return abs(points[NOSE_TIP, 0] - (left + right) / 2) > TURNED * abs(right - left)


def face_weight(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    face_w = float(np.ptp(points[:, 0]))
    centre = points.mean(axis=0)
    hull = cv2.convexHull(np.round(centre + (points - centre) * FACE_TINT_SHRINK).astype(np.int32))
    mask = np.zeros(shape, np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, face_w * FACE_TINT_FEATHER))


@cache
def injury_texture(stage: int, mirrored: bool) -> tuple[np.ndarray, np.ndarray]:
    texture = cv2.resize(paint_stage(stage), (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_AREA)
    if mirrored:
        texture = np.ascontiguousarray(texture[:, ::-1])
    marked = (texture.min(axis=2) < 255 - PAINTED_LEVEL).astype(np.uint8)
    marked = cv2.dilate(marked, np.ones((3, 3), np.uint8))
    painted = np.zeros(len(TRIANGLES) + 1, bool)
    painted[np.unique(uv_owner()[marked > 0])] = True
    return texture, painted[1:]


@cache
def paint_stage(stage: int) -> np.ndarray:
    shade = np.ones((BASE_SIZE, BASE_SIZE, 3), np.float32)
    for decal in DECALS:
        if decal.stage <= stage:
            stamp(shade, decal)
    skin = skin_mask()[:, :, None].astype(np.float32) / 255
    return ((1 - (1 - np.clip(shade, 0, 1)) * skin) * 255).astype(np.uint8)


def stamp(shade: np.ndarray, decal: Decal) -> None:
    image = load_decal(decal.image)
    height, width = image.shape[:2]
    centre = (UV[list(decal.anchor)].mean(axis=0) + decal.offset) * BASE_SIZE
    angle = np.radians(decal.angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    linear = rotation * (np.array(decal.size) * BASE_SIZE / (width, height))
    matrix = np.hstack([linear, (centre - linear @ (width / 2, height / 2))[:, None]]).astype(np.float32)
    warped = (
        cv2.warpAffine(
            image, matrix, (BASE_SIZE, BASE_SIZE), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255, 0)
        ).astype(np.float32)
        / 255
    )
    shade *= 1 - warped[:, :, 3:] * decal.strength * (1 - warped[:, :, :3])


@cache
def load_decal(name: str) -> np.ndarray:
    image = cv2.imread(str(DECAL_DIR / name), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise FileNotFoundError(f"{DECAL_DIR / name} missing - run: python tools/make_face_decals.py")
    return image


@cache
def uv_owner() -> np.ndarray:
    owner = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE), np.int16)
    uv_pixels = np.round(UV * (TEXTURE_SIZE - 1)).astype(np.int32)
    for index, triangle in enumerate(TRIANGLES):
        cv2.fillConvexPoly(owner, uv_pixels[triangle], index + 1)
    return owner


@cache
def skin_mask() -> np.ndarray:
    mask = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    uv_pixels = np.round(UV * (BASE_SIZE - 1)).astype(np.int32)
    for triangle in TRIANGLES:
        cv2.fillConvexPoly(mask, uv_pixels[triangle], 255)
    feather = int(SKIN_FEATHER * BASE_SIZE)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1)))
    return cv2.GaussianBlur(inner, (0, 0), feather / 1.5)
