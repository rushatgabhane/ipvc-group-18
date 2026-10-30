from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
FONT_DIR = ASSET_DIR / "fonts"
SPRITE_DIR = ASSET_DIR / "sprites"
OUTLINE_PX = 2
TITLE_FONT = FONT_DIR / "Rye-Regular.ttf"
HUD_FONT = FONT_DIR / "BebasNeue-Regular.ttf"
SUPERSAMPLE = 4
ROTATION_STEP_DEG = 4
HORIZON = 0.62

POWERUP_COLORS = {"rocket": (30, 120, 240), "grenade": (60, 150, 70), "bomb": (40, 40, 200)}
POWERUP_LABELS = {"rocket": "ROCKET LAUNCHER", "grenade": "GRENADE"}


@dataclass(frozen=True)
class WeaponArt:
    file: str
    grip: tuple[float, float]
    muzzle: tuple[float, float]


WEAPON_ART = {
    "gun": WeaponArt("revolver.png", (10, 3), (59, 2)),
    "rocket": WeaponArt("rocket_launcher.png", (42, 10), (168, 10)),
    "grenade": WeaponArt("grenade.png", (6, 6), (12, 6)),
}


@dataclass(frozen=True, eq=False, slots=True)
class Sprite:
    bgra: np.ndarray
    bgr: np.ndarray
    weight: np.ndarray
    inverse: np.ndarray

    @staticmethod
    def from_bgra(bgra: np.ndarray) -> Sprite:
        weight = bgra[:, :, 3].astype(np.float32) * (1.0 / 255.0)
        return Sprite(bgra, np.ascontiguousarray(bgra[:, :, :3]), weight, 1.0 - weight)

    @property
    def width(self) -> int:
        return self.bgra.shape[1]

    @property
    def height(self) -> int:
        return self.bgra.shape[0]


def blit(image: np.ndarray, sprite: Sprite, x: int, y: int, alpha: float = 1.0) -> None:
    image_h, image_w = image.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sprite.width, image_w), min(y + sprite.height, image_h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    region = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    weight = sprite.weight[region]
    inverse = sprite.inverse[region]
    if alpha < 1.0:
        weight = weight * alpha
        inverse = 1.0 - weight
    roi = image[y0:y1, x0:x1]
    roi[:] = cv2.blendLinear(sprite.bgr[region], roi, weight, inverse)


def blit_center(image: np.ndarray, sprite: Sprite, cx: float, cy: float, alpha: float = 1.0) -> None:
    blit(image, sprite, int(cx - sprite.width / 2), int(cy - sprite.height / 2), alpha)


def scaled(sprite: Sprite, scale: float) -> Sprite:
    if abs(scale - 1.0) < 1e-3:
        return sprite
    size = (max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale)))
    return Sprite.from_bgra(cv2.resize(sprite.bgra, size, interpolation=cv2.INTER_LINEAR))


@lru_cache(maxsize=8)
def sprite_art(file: str) -> np.ndarray:
    return cv2.imread(str(SPRITE_DIR / file), cv2.IMREAD_UNCHANGED)


@lru_cache(maxsize=16)
def held_weapon(kind: str, color: tuple[int, int, int], reach_px: int) -> tuple[Sprite, Sprite]:
    spec = WEAPON_ART[kind]
    art = sprite_art(spec.file)
    zoom = reach_px / float(np.linalg.norm(np.subtract(spec.muzzle, spec.grip)))
    weapon = cv2.resize(art, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    weapon = cv2.copyMakeBorder(weapon, *[OUTLINE_PX] * 4, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    right = Sprite.from_bgra(outlined(centered_on(weapon, np.array(spec.grip) * zoom + OUTLINE_PX), color))
    return right, Sprite.from_bgra(np.ascontiguousarray(right.bgra[::-1]))


def centered_on(bgra: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    height, width = bgra.shape[:2]
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    half = int(np.ceil(np.linalg.norm(corners - pivot, axis=1).max()))
    canvas = np.zeros((2 * half, 2 * half, 4), np.uint8)
    x0, y0 = half - int(round(pivot[0])), half - int(round(pivot[1]))
    canvas[y0 : y0 + height, x0 : x0 + width] = bgra
    return canvas


def art_sprite(file: str, width: int) -> Sprite:
    art = sprite_art(file)
    zoom = width / art.shape[1]
    scaled_art = cv2.resize(art, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    return Sprite.from_bgra(centered_on(scaled_art, np.array(scaled_art.shape[1::-1]) / 2))


def outlined(bgra: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    size = 2 * OUTLINE_PX + 1
    ring = cv2.dilate(bgra[:, :, 3], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    weight = bgra[:, :, 3:].astype(np.float32) / 255
    out = np.empty_like(bgra)
    out[:, :, :3] = (bgra[:, :, :3] * weight + np.array(color, np.float32) * (1 - weight)).astype(np.uint8)
    out[:, :, 3] = ring
    return out


def rotated(sprite: Sprite, angle_deg: float) -> Sprite:
    return _rotated(sprite, int(round(angle_deg / ROTATION_STEP_DEG)) % (360 // ROTATION_STEP_DEG))


@lru_cache(maxsize=512)
def _rotated(sprite: Sprite, step: int) -> Sprite:
    center = (sprite.width / 2, sprite.height / 2)
    matrix = cv2.getRotationMatrix2D(center, step * ROTATION_STEP_DEG, 1.0)
    size = (sprite.width, sprite.height)
    return Sprite.from_bgra(
        cv2.warpAffine(sprite.bgra, matrix, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    )


def supersampled(width: int, height: int) -> tuple[np.ndarray, int, int]:
    return (
        np.zeros((height * SUPERSAMPLE, width * SUPERSAMPLE, 4), np.uint8),
        width * SUPERSAMPLE,
        height * SUPERSAMPLE,
    )


def downsample(canvas: np.ndarray, width: int, height: int) -> Sprite:
    return Sprite.from_bgra(cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA))


@lru_cache(maxsize=16)
def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


class TextCache:
    def __init__(self, capacity: int = 512):
        self._sprites: OrderedDict[tuple, Sprite] = OrderedDict()
        self.capacity = capacity

    def get(
        self,
        text: str,
        size: int,
        color: tuple[int, int, int],
        font: Path = HUD_FONT,
        stroke: int = 0,
        stroke_color: tuple[int, int, int] = (0, 0, 0),
        shadow: int = 0,
    ) -> Sprite:
        key = (text, size, color, font, stroke, stroke_color, shadow)
        sprite = self._sprites.get(key)
        if sprite is not None:
            self._sprites.move_to_end(key)
            return sprite
        sprite = render_text(text, load_font(font, size), color, stroke, stroke_color, shadow)
        self._sprites[key] = sprite
        if len(self._sprites) > self.capacity:
            self._sprites.popitem(last=False)
        return sprite


def render_text(text, font, bgr, stroke, stroke_bgr, shadow) -> Sprite:
    rgb, stroke_rgb = tuple(reversed(bgr)), tuple(reversed(stroke_bgr))
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    pad = stroke + shadow + 2
    canvas = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    origin = (pad - left, pad - top)
    if shadow:
        shadow_origin = (origin[0] + shadow, origin[1] + shadow)
        shadow_fill = (0, 0, 0, 150)
        draw.text(
            shadow_origin, text, font=font, fill=shadow_fill, stroke_width=stroke, stroke_fill=shadow_fill
        )
    draw.text(
        origin, text, font=font, fill=rgb + (255,), stroke_width=stroke, stroke_fill=stroke_rgb + (255,)
    )
    return Sprite.from_bgra(np.ascontiguousarray(np.array(canvas)[:, :, [2, 1, 0, 3]]))


def rounded_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, color, thickness: int = -1):
    r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    corners = ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90))
    if thickness < 0:
        cv2.rectangle(image, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(image, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy, _ in corners:
            cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    cv2.line(image, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, angle in corners:
        cv2.ellipse(image, (cx, cy), (r, r), angle, 0, 90, color, thickness, cv2.LINE_AA)


@lru_cache(maxsize=64)
def panel_sprite(width: int, height: int, radius: int, opacity: float, color: tuple[int, int, int]) -> Sprite:
    mask = np.zeros((height, width), np.uint8)
    rounded_rect(mask, 0, 0, width - 1, height - 1, radius, 255)
    bgra = np.zeros((height, width, 4), np.uint8)
    bgra[:, :, :3] = color
    bgra[:, :, 3] = (mask * opacity).astype(np.uint8)
    return Sprite.from_bgra(bgra)


def panel(image, x0: int, y0: int, x1: int, y1: int, r: int = 12, opacity: float = 0.6, color=(20, 16, 14)):
    blit(image, panel_sprite(x1 - x0, y1 - y0, r, opacity, color), x0, y0)


@lru_cache(maxsize=8)
def glow_sprite(size: int, color: tuple[int, int, int]) -> Sprite:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = (size - 1) / 2
    distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
    bgra = np.zeros((size, size, 4), np.uint8)
    bgra[:, :, :3] = color
    bgra[:, :, 3] = (np.clip(1 - distance, 0, 1) ** 1.6 * 255).astype(np.uint8)
    return Sprite.from_bgra(bgra)


@lru_cache(maxsize=4)
def bullet_icon(height: int, filled: bool) -> Sprite:
    width = int(height * 0.38)
    canvas, w, h = supersampled(width, height)
    brass, tip = ((60, 170, 225), (90, 120, 160)) if filled else ((70, 70, 70), (60, 60, 60))
    alpha = 255 if filled else 140
    base = tuple(int(c * 0.7) for c in brass)
    cv2.rectangle(canvas, (0, int(h * 0.38)), (w - 1, h - 1), brass + (alpha,), -1)
    cv2.ellipse(canvas, (w // 2, int(h * 0.38)), (w // 2, int(h * 0.36)), 0, 180, 360, tip + (alpha,), -1)
    cv2.rectangle(canvas, (0, int(h * 0.86)), (w - 1, h - 1), base + (alpha,), -1)
    return downsample(canvas, width, height)


@lru_cache(maxsize=8)
def powerup_sprite(size: int, kind: str) -> Sprite:
    canvas, s, _ = supersampled(size, size)
    c, r = s // 2, s // 2 - 2 * SUPERSAMPLE
    color = POWERUP_COLORS[kind]
    cv2.circle(canvas, (c, c), r, tuple(int(v * 0.55) for v in color) + (255,), -1, cv2.LINE_AA)
    cv2.circle(canvas, (c, c), int(r * 0.86), color + (255,), -1, cv2.LINE_AA)
    shine = (int(r * 0.7), int(r * 0.7))
    cv2.ellipse(canvas, (c, c), shine, 0, 200, 290, (255, 255, 255, 140), 3 * SUPERSAMPLE, cv2.LINE_AA)
    icon_size = r * 0.42
    {"rocket": draw_rocket_icon, "grenade": draw_grenade_icon, "bomb": draw_bomb_icon}[kind](
        canvas, c, icon_size
    )
    return downsample(canvas, size, size)


WHITE_BGRA = (255, 255, 255, 255)


def draw_rocket_icon(canvas: np.ndarray, c: int, k: float) -> None:
    size = k * 2.1
    along, across = np.array([0.7071, -0.7071]), np.array([0.7071, 0.7071])

    def polygon(points):
        return np.array([c + size * (a * along + b * across) for a, b in points]).round().astype(np.int32)

    flame = [(-0.42, -0.08), (-0.7, 0.0), (-0.42, 0.08)]
    body = [(-0.42, -0.11), (0.2, -0.11), (0.48, 0.0), (0.2, 0.11), (-0.42, 0.11)]
    cv2.fillPoly(canvas, [polygon(flame)], (60, 220, 255, 255), cv2.LINE_AA)
    cv2.fillPoly(canvas, [polygon(body)], WHITE_BGRA, cv2.LINE_AA)
    for side in (-1, 1):
        fin = [(-0.42, 0.11 * side), (-0.2, 0.11 * side), (-0.46, 0.3 * side)]
        cv2.fillPoly(canvas, [polygon(fin)], WHITE_BGRA, cv2.LINE_AA)


def draw_grenade_icon(canvas: np.ndarray, c: int, k: float) -> None:
    body_axes = (int(k * 0.72), int(k * 0.9))
    cv2.ellipse(canvas, (c, int(c + k * 0.2)), body_axes, 0, 0, 360, WHITE_BGRA, -1, cv2.LINE_AA)
    cv2.rectangle(
        canvas, (int(c - k * 0.3), int(c - k * 0.95)), (int(c + k * 0.3), int(c - k * 0.6)), WHITE_BGRA, -1
    )
    ring_center = (int(c + k * 0.55), int(c - k * 0.95))
    cv2.circle(canvas, ring_center, int(k * 0.25), WHITE_BGRA, 2 * SUPERSAMPLE, cv2.LINE_AA)


def draw_bomb_icon(canvas: np.ndarray, c: int, k: float) -> None:
    body_center = (int(c - k * 0.1), int(c + k * 0.15))
    cv2.circle(canvas, body_center, int(k * 0.8), WHITE_BGRA, -1, cv2.LINE_AA)
    cv2.rectangle(
        canvas, (int(c + k * 0.2), int(c - k * 0.75)), (int(c + k * 0.55), int(c - k * 0.45)), WHITE_BGRA, -1
    )
    fuse = np.array([[c + k * 0.4, c - k * 0.7], [c + k * 0.6, c - k * 1.0], [c + k * 0.9, c - k * 1.05]])
    cv2.polylines(canvas, [fuse.astype(np.int32)], False, WHITE_BGRA, 2 * SUPERSAMPLE, cv2.LINE_AA)
    cv2.circle(
        canvas, (int(c + k * 0.95), int(c - k * 1.05)), int(k * 0.2), (80, 230, 255, 255), -1, cv2.LINE_AA
    )


@lru_cache(maxsize=4)
def bottle_sprite(height: int) -> Sprite:
    width = int(height * 0.42)
    canvas, w, h = supersampled(width, height)
    glass, shade, shine = (60, 140, 50, 255), (35, 85, 30, 255), (190, 240, 190, 255)
    cx, neck_w, body_w = w // 2, int(w * 0.16), int(w * 0.44)
    body_top, neck_top = int(h * 0.42), int(h * 0.04)
    cv2.rectangle(canvas, (cx - neck_w, neck_top), (cx + neck_w, body_top + 8), glass, -1)
    cv2.rectangle(
        canvas, (cx - int(neck_w * 1.25), neck_top), (cx + int(neck_w * 1.25), int(h * 0.1)), shade, -1
    )
    cv2.ellipse(canvas, (cx, body_top + 8), (body_w, int(h * 0.16)), 0, 180, 360, glass, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (cx - body_w, body_top + 8), (cx + body_w, h - 2), glass, -1)
    cv2.rectangle(
        canvas, (cx - body_w, int(h * 0.62)), (cx + body_w, int(h * 0.84)), (200, 225, 235, 255), -1
    )
    cv2.rectangle(canvas, (cx + body_w // 2, body_top + 8), (cx + body_w, h - 2), shade, -1)
    cv2.line(
        canvas, (cx - body_w // 2, int(h * 0.5)), (cx - body_w // 2, int(h * 0.58)), shine, 3 * SUPERSAMPLE
    )
    return Sprite.from_bgra(outlined(downsample(canvas, width, height).bgra, (20, 25, 40)))


@lru_cache(maxsize=4)
def grenade_sprite(height: int) -> Sprite:
    return art_sprite("grenade.png", height)


@lru_cache(maxsize=4)
def rocket_sprite(length: int) -> Sprite:
    return art_sprite("rocket.png", length)


@lru_cache(maxsize=8)
def bullet_sprite(length: int, color: tuple[int, int, int]) -> Sprite:
    canvas, s, _ = supersampled(length, length)
    c, half = s // 2, int(s * 0.11)
    tail, shoulder, tip = int(s * 0.18), int(s * 0.6), int(s * 0.86)
    copper, shade, shine = (55, 105, 185, 255), (35, 65, 120, 255), (150, 200, 240, 255)
    cv2.ellipse(canvas, (shoulder, c), (tip - shoulder, half), 0, -90, 90, copper, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (tail, c - half), (shoulder, c + half), copper, -1)
    cv2.rectangle(canvas, (tail, c + half // 3), (shoulder, c + half), shade, -1)
    cv2.line(canvas, (tail + 2 * SUPERSAMPLE, c - half // 2), (shoulder, c - half // 2), shine, SUPERSAMPLE)
    cv2.rectangle(canvas, (tail, c - half), (tail + 2 * SUPERSAMPLE, c + half), shade, -1)
    sprite = downsample(canvas, length, length)
    return Sprite.from_bgra(outlined(sprite.bgra, color))


SKY_TOP, SKY_LOW = (175, 120, 55), (150, 205, 245)
SUN_X, SUN_Y = 0.8, 0.2
SUN_COLOR, SUN_GLOW = (170, 240, 255), (60, 110, 120)


def desert_background(width: int, height: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = sky_with_sun(width, height)
    horizon = int(height * HORIZON)
    draw_mesa_ridge(
        image, rng, horizon, horizon - height * 0.05, height * 0.16, (120, 150, 200), height * 0.04
    )
    draw_mesa_ridge(
        image, rng, horizon, horizon - height * 0.01, height * 0.10, (70, 105, 165), height * 0.03
    )
    image[horizon:] = sand(rng, width, height - horizon)
    for x in (int(width * 0.12), int(width * 0.33), int(width * 0.9)):
        cactus_h = int(height * rng.uniform(0.06, 0.1))
        draw_cactus(image, rng, x, horizon + int(height * 0.01), cactus_h)
    return image


def sky_with_sun(width: int, height: int) -> np.ndarray:
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    top, low = np.array(SKY_TOP, np.float32), np.array(SKY_LOW, np.float32)
    sky = (top + (low - top) * np.clip(t / 0.62, 0, 1) ** 1.3) * np.ones((1, width, 1), np.float32)
    sun_x, sun_y, sun_r = int(width * SUN_X), int(height * SUN_Y), int(height * 0.075)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    glow = np.clip(1 - np.sqrt((xx - sun_x) ** 2 + (yy - sun_y) ** 2) / (sun_r * 5), 0, 1) ** 2.2
    sky += glow[:, :, None] * np.array(SUN_GLOW, np.float32)
    image = np.clip(sky, 0, 255).astype(np.uint8)
    cv2.circle(image, (sun_x, sun_y), sun_r, SUN_COLOR, -1, cv2.LINE_AA)
    return image


def draw_mesa_ridge(image, rng, horizon: int, base_y: float, amplitude: float, color, plateau: float) -> None:
    width = image.shape[1]
    xs = np.linspace(0, width, 24)
    ys = np.round((base_y - amplitude * np.abs(rng.normal(0.5, 0.35, xs.size))) / plateau) * plateau
    points = [(0, horizon + 2)]
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        points.append((int(x), int(y)))
        if i + 1 < xs.size:
            points.append((int((x + xs[i + 1]) / 2), int(y)))
    points.append((width, horizon + 2))
    cv2.fillPoly(image, [np.array(points, np.int32)], color, cv2.LINE_AA)


def sand(rng, width: int, height: int) -> np.ndarray:
    depth = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    ground = np.array([110, 170, 220], np.float32) * (1 - 0.35 * depth) * np.ones((1, width, 1), np.float32)
    ground += rng.normal(0, 6, (height, width))[:, :, None]
    return np.clip(ground, 0, 255).astype(np.uint8)


def draw_cactus(image, rng, x: int, base: int, height: int, color=(45, 85, 60)) -> None:
    trunk = max(4, height // 7)
    cv2.rectangle(image, (x - trunk // 2, base - height), (x + trunk // 2, base), color, -1)
    cv2.circle(image, (x, base - height), trunk // 2, color, -1, cv2.LINE_AA)
    for side in (-1, 1):
        arm_y = base - int(height * rng.uniform(0.45, 0.7))
        arm_x = x + side * trunk * 2
        cv2.rectangle(
            image, (min(x, arm_x), arm_y - trunk // 2), (max(x, arm_x), arm_y + trunk // 2), color, -1
        )
        cv2.rectangle(
            image, (arm_x - trunk // 2, arm_y - height // 3), (arm_x + trunk // 2, arm_y), color, -1
        )
        cv2.circle(image, (arm_x, arm_y - height // 3), trunk // 2, color, -1, cv2.LINE_AA)


def wooden_wall(width: int, height: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wall = np.zeros((height, width, 3), np.float32)
    rail = max(10, height // 9)
    plank_width = max(60, width // 14)
    x = 0
    while x < width:
        plank = int(plank_width * rng.uniform(0.85, 1.15))
        wall[rail:, x : x + plank] = wood_plank(rng, plank, height - rail)[:, : max(0, min(plank, width - x))]
        wall[rail:, x : x + 2] *= 0.45
        for nail_y in (int(height * 0.3), int(height * 0.8)):
            cv2.circle(wall, (x + plank // 2, nail_y), 3, (60, 60, 70), -1, cv2.LINE_AA)
        x += plank
    wall[:rail] = np.array([60, 125, 185], np.float32)
    wall[:3] = np.array([120, 185, 230], np.float32)
    wall[rail : rail + 4] *= 0.5
    wall *= np.linspace(1.0, 0.7, height, dtype=np.float32)[:, None, None]
    return np.clip(wall, 0, 255).astype(np.uint8)


def wood_plank(rng, width: int, height: int) -> np.ndarray:
    base = np.array([45, 95, 150], np.float32) * rng.uniform(0.8, 1.1)
    coarse = rng.normal(0, 1, (4, max(4, width // 2))).astype(np.float32)
    streaks = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    rounded_edges = np.linspace(1.08, 0.88, width, dtype=np.float32)[None, :, None]
    return (base + (streaks * 10 + fine * 3)[:, :, None]) * rounded_edges


def vignette(width: int, height: int, strength: float = 0.35) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.sqrt(
        ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
    ) / np.sqrt(2)
    multiplier = 1 - strength * np.clip(distance - 0.35, 0, 1) ** 1.5 / 0.65**1.5
    return cv2.merge([(multiplier * 255).astype(np.uint8)] * 3)
