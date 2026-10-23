from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
TITLE_FONT = FONT_DIR / "Rye-Regular.ttf"
HUD_FONT = FONT_DIR / "BebasNeue-Regular.ttf"
SUPERSAMPLE = 4
ROTATION_STEP_DEG = 4

POWERUP_COLORS = {"heal": (70, 70, 225), "rocket": (30, 120, 240), "grenade": (60, 150, 70)}
POWERUP_LABELS = {"heal": "FULL HP", "rocket": "ROCKET LAUNCHER", "grenade": "GRENADE"}


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


def outline(canvas: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    kernel = np.ones((thickness, thickness), np.uint8)
    edge = cv2.morphologyEx(canvas[:, :, 3].copy(), cv2.MORPH_GRADIENT, kernel)
    canvas[edge > 0, :3] = color


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
    {"heal": draw_cross_icon, "rocket": draw_rocket_icon, "grenade": draw_grenade_icon}[kind](
        canvas, c, icon_size
    )
    return downsample(canvas, size, size)


WHITE_BGRA = (255, 255, 255, 255)


def draw_cross_icon(canvas: np.ndarray, c: int, k: float) -> None:
    arm = int(k * 0.36)
    cv2.rectangle(canvas, (int(c - k), c - arm), (int(c + k), c + arm), WHITE_BGRA, -1)
    cv2.rectangle(canvas, (c - arm, int(c - k)), (c + arm, int(c + k)), WHITE_BGRA, -1)


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


@lru_cache(maxsize=4)
def grenade_sprite(height: int) -> Sprite:
    canvas, s, _ = supersampled(height, height)
    c = s // 2
    rx, ry = int(s * 0.28), int(s * 0.34)
    cy = c + int(s * 0.06)
    olive, groove, light = (50, 95, 70, 255), (25, 50, 35, 255), (120, 175, 140, 170)
    cv2.ellipse(canvas, (c, cy), (rx, ry), 0, 0, 360, olive, -1, cv2.LINE_AA)
    for row in (-2, -1, 0, 1, 2):
        y = cy + row * ry // 3
        half = int(rx * np.sqrt(max(0.0, 1 - (row / 3) ** 2)))
        cv2.line(canvas, (c - half, y), (c + half, y), groove, SUPERSAMPLE)
    for column in (-1, 0, 1):
        cv2.ellipse(canvas, (c + column * rx // 2, cy), (max(1, rx // 6), ry), 0, 0, 360, groove, SUPERSAMPLE)
    cv2.ellipse(canvas, (c - rx // 3, cy - ry // 3), (rx // 4, ry // 5), -30, 0, 360, light, -1, cv2.LINE_AA)
    top = cy - ry
    fuse_top = top - int(s * 0.08)
    cv2.rectangle(canvas, (c - rx // 3, fuse_top), (c + rx // 3, top + SUPERSAMPLE), (90, 90, 95, 255), -1)
    lever = np.array(
        [
            (c + rx // 3, fuse_top),
            (c + rx // 2, top - int(s * 0.1)),
            (c + int(rx * 0.9), cy - ry // 4),
            (c + int(rx * 0.75), cy - ry // 4),
        ],
        np.int32,
    )
    cv2.fillPoly(canvas, [lever], (120, 120, 125, 255), cv2.LINE_AA)
    ring = (c - rx // 2, top - int(s * 0.06))
    cv2.circle(canvas, ring, int(s * 0.06), (170, 200, 215, 255), 2 * SUPERSAMPLE, cv2.LINE_AA)
    outline(canvas, (15, 30, 20), 2 * SUPERSAMPLE)
    return downsample(canvas, height, height)


@lru_cache(maxsize=4)
def rocket_sprite(length: int) -> Sprite:
    canvas, s, _ = supersampled(length, length)
    cy = s // 2
    half_body = int(s * 0.075)
    body, body_shadow, red, red_dark = (
        (75, 115, 95, 255),
        (45, 70, 55, 255),
        (45, 45, 205, 255),
        (30, 30, 140),
    )
    x_tail, x_nose, x_tip = int(s * 0.12), int(s * 0.72), int(s * 0.9)
    for side in (-1, 1):
        fin = [
            (x_tail, cy + side * half_body),
            (x_tail + int(s * 0.2), cy + side * half_body),
            (x_tail - int(s * 0.02), cy + side * int(s * 0.19)),
        ]
        cv2.fillPoly(canvas, [np.array(fin, np.int32)], red, cv2.LINE_AA)
    nozzle = int(half_body * 0.7)
    cv2.rectangle(canvas, (x_tail - int(s * 0.05), cy - nozzle), (x_tail, cy + nozzle), (60, 60, 60, 255), -1)
    cv2.rectangle(canvas, (x_tail, cy - half_body), (x_nose, cy + half_body), body, -1)
    nose = np.array([(x_nose, cy - half_body), (x_tip, cy), (x_nose, cy + half_body)], np.int32)
    cv2.fillPoly(canvas, [nose], red, cv2.LINE_AA)
    cv2.rectangle(
        canvas, (int(s * 0.58), cy - half_body), (int(s * 0.62), cy + half_body), (200, 225, 235, 255), -1
    )
    cv2.rectangle(canvas, (x_tail, cy + half_body // 3), (x_nose, cy + half_body), body_shadow, -1)
    highlight_y = cy - half_body // 2
    cv2.line(
        canvas, (x_tail + 8, highlight_y), (x_nose - 8, highlight_y), (170, 210, 190, 255), 2 * SUPERSAMPLE
    )
    outline(canvas, red_dark, 2 * SUPERSAMPLE)
    return downsample(canvas, length, length)


def desert_background(width: int, height: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = sky_with_sun(width, height)
    horizon = int(height * 0.62)
    draw_mesa_ridge(
        image, rng, horizon, horizon - height * 0.05, height * 0.16, (120, 150, 200), height * 0.04
    )
    draw_mesa_ridge(
        image, rng, horizon, horizon - height * 0.01, height * 0.10, (70, 105, 165), height * 0.03
    )
    image[horizon:] = sand(rng, width, height - horizon)
    for x in (int(width * 0.12), int(width * 0.33), int(width * 0.9)):
        draw_cactus(image, rng, x, horizon + int(height * 0.01), int(height * rng.uniform(0.06, 0.1)))
    return image


def sky_with_sun(width: int, height: int) -> np.ndarray:
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    top, low = np.array([175, 120, 55], np.float32), np.array([150, 205, 245], np.float32)
    sky = (top + (low - top) * np.clip(t / 0.62, 0, 1) ** 1.3) * np.ones((1, width, 1), np.float32)
    sun_x, sun_y, sun_r = int(width * 0.8), int(height * 0.2), int(height * 0.075)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    glow = np.clip(1 - np.sqrt((xx - sun_x) ** 2 + (yy - sun_y) ** 2) / (sun_r * 5), 0, 1) ** 2.2
    sky += glow[:, :, None] * np.array([60, 110, 120], np.float32)
    image = np.clip(sky, 0, 255).astype(np.uint8)
    cv2.circle(image, (sun_x, sun_y), sun_r, (170, 240, 255), -1, cv2.LINE_AA)
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


def draw_cactus(image, rng, x: int, base: int, height: int) -> None:
    trunk = max(4, height // 7)
    color = (45, 85, 60)
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
