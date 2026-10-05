"""Visual assets, generated procedurally once at startup and cached.

Nothing here runs per pixel per frame. Text is rendered with Pillow (TrueType, anti-aliased) into
small RGBA sprites, cached by (string, font, size, colour), and alpha-blitted into the frame. That
costs well under 0.1 ms for a HUD-sized sprite. Backgrounds and the wall are built once per
resolution.

Fonts (SIL Open Font License, see assets/fonts/): Rye for Western titles, Bebas Neue for the HUD.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
TITLE_FONT = FONT_DIR / "Rye-Regular.ttf"
HUD_FONT = FONT_DIR / "BebasNeue-Regular.ttf"


# ------------------------------------------------------------------------------------------ blitting
def blit(image: np.ndarray, sprite: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alpha-blend a BGRA sprite onto a BGR image at top-left (x, y), clipped to the image."""
    h, w = sprite.shape[:2]
    H, W = image.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    sp = sprite[y0 - y : y1 - y, x0 - x : x1 - x]
    a = sp[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    roi = image[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - a) + sp[:, :, :3] * a).astype(np.uint8)


def blit_center(image: np.ndarray, sprite: np.ndarray, cx: float, cy: float, alpha: float = 1.0) -> None:
    h, w = sprite.shape[:2]
    blit(image, sprite, int(cx - w / 2), int(cy - h / 2), alpha)


def scaled(sprite: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return sprite
    h, w = sprite.shape[:2]
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(sprite, size, interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------------------------------ text
@lru_cache(maxsize=16)
def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


class TextCache:
    """Renders strings to BGRA sprites once; repeated HUD text is then a cheap blit."""

    def __init__(self, capacity: int = 512):
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
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
    ) -> np.ndarray:
        key = (text, size, color, font, stroke, stroke_color, shadow)
        sprite = self._cache.get(key)
        if sprite is not None:
            self._cache.move_to_end(key)
            return sprite
        sprite = _render_text(text, _font(font, size), color, stroke, stroke_color, shadow)
        self._cache[key] = sprite
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return sprite


def _render_text(text, font, bgr, stroke, stroke_bgr, shadow) -> np.ndarray:
    rgb, stroke_rgb = tuple(reversed(bgr)), tuple(reversed(stroke_bgr))
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    pad = stroke + shadow + 2
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    origin = (pad - left, pad - top)
    if shadow:
        draw.text((origin[0] + shadow, origin[1] + shadow), text, font=font, fill=(0, 0, 0, 150),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 150))  # fmt: skip
    draw.text(
        origin, text, font=font, fill=rgb + (255,), stroke_width=stroke, stroke_fill=stroke_rgb + (255,)
    )
    rgba = np.array(img)
    return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])  # RGBA -> BGRA


# ------------------------------------------------------------------------------------------ shapes
def rounded_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, color, thickness: int = -1):
    """Filled (thickness=-1) or outlined rounded rectangle, anti-aliased corners."""
    r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(image, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(image, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0),
                              ((x0 + r, y1 - r), 90)):  # fmt: skip
            cv2.ellipse(image, (cx, cy), (r, r), ang, 0, 90, color, thickness, cv2.LINE_AA)


def panel(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int = 12, opacity: float = 0.6,
          color=(20, 16, 14)) -> None:  # fmt: skip
    """Translucent dark rounded panel (blends only the panel's own region)."""
    H, W = image.shape[:2]
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1c <= x0c or y1c <= y0c:
        return
    roi = image[y0c:y1c, x0c:x1c]
    mask = np.zeros(roi.shape[:2], np.uint8)
    rounded_rect(mask, x0 - x0c, y0 - y0c, x1 - x0c - 1, y1 - y0c - 1, r, 255)
    a = (mask.astype(np.float32) * (opacity / 255.0))[:, :, None]
    roi[:] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


# ------------------------------------------------------------------------------------------ sprites
def _soft_circle(size: int, color, falloff: float = 2.0) -> np.ndarray:
    """Radial glow sprite: colour with alpha fading from the centre."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    alpha = np.clip(1 - d, 0, 1) ** falloff
    out = np.zeros((size, size, 4), np.uint8)
    out[:, :, :3] = color
    out[:, :, 3] = (alpha * 255).astype(np.uint8)
    return out


@lru_cache(maxsize=8)
def bottle_sprite(height: int) -> np.ndarray:
    """A green glass bottle with highlight, BGRA, drawn at 4x and downsampled for smooth edges."""
    s = 4
    H, W = height * s, int(height * 0.42) * s
    img = np.zeros((H, W, 4), np.uint8)
    body_top, neck_w = int(H * 0.38), int(W * 0.34)
    cx = W // 2
    glass, dark, light = (55, 125, 45), (30, 70, 25), (170, 235, 170)
    # Neck, shoulder and body.
    cv2.rectangle(img, (cx - neck_w // 2, int(H * 0.06)), (cx + neck_w // 2, body_top), glass + (235,), -1)
    cv2.ellipse(img, (cx, body_top), (W // 2 - s, int(H * 0.1)), 0, 180, 360, glass + (235,), -1)
    cv2.rectangle(img, (s, body_top), (W - s, H - 3 * s), glass + (235,), -1)
    cv2.ellipse(img, (cx, H - 3 * s), (W // 2 - s, 3 * s), 0, 0, 180, glass + (235,), -1)
    # Cap, label, highlight, outline.
    cv2.rectangle(
        img, (cx - neck_w // 2 - s, 0), (cx + neck_w // 2 + s, int(H * 0.08)), (40, 40, 160, 255), -1
    )
    cv2.rectangle(img, (s, int(H * 0.55)), (W - s, int(H * 0.78)), (170, 215, 235, 255), -1)
    cv2.line(img, (2 * s, int(H * 0.66)), (W - 2 * s, int(H * 0.66)), (60, 90, 160, 255), 2 * s)
    cv2.line(img, (int(W * 0.25), body_top + 4 * s), (int(W * 0.25), int(H * 0.52)), light + (200,), 3 * s)
    cv2.line(
        img, (cx - neck_w // 4, int(H * 0.1)), (cx - neck_w // 4, body_top - 2 * s), light + (160,), 2 * s
    )
    alpha = img[:, :, 3].copy()
    edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((3 * s, 3 * s), np.uint8))
    img[edge > 0, :3] = dark
    small = cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA)
    return small


@lru_cache(maxsize=4)
def glow_sprite(size: int, color: tuple[int, int, int]) -> np.ndarray:
    return _soft_circle(size, color, falloff=1.6)


@lru_cache(maxsize=4)
def bullet_icon(height: int, filled: bool) -> np.ndarray:
    """Upright cartridge for the ammo counter."""
    s = 4
    H, W = height * s, int(height * 0.38) * s
    img = np.zeros((H, W, 4), np.uint8)
    brass, tip = ((60, 170, 225), (90, 120, 160)) if filled else ((70, 70, 70), (60, 60, 60))
    a = 255 if filled else 140
    cv2.rectangle(img, (0, int(H * 0.38)), (W - 1, H - 1), brass + (a,), -1)
    cv2.ellipse(img, (W // 2, int(H * 0.38)), (W // 2, int(H * 0.36)), 0, 180, 360, tip + (a,), -1)
    cv2.rectangle(img, (0, int(H * 0.86)), (W - 1, H - 1), tuple(int(c * 0.7) for c in brass) + (a,), -1)
    return cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------ scenery
def desert_background(w: int, h: int, seed: int = 7) -> np.ndarray:
    """High-noon desert: sky gradient, sun glow, two layers of mesas, cacti, textured sand. BGR."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky_top, sky_low = np.array([175, 120, 55], np.float32), np.array([150, 205, 245], np.float32)
    img = (sky_top + (sky_low - sky_top) * np.clip(t / 0.62, 0, 1) ** 1.3) * np.ones((1, w, 1), np.float32)

    # Sun with a soft glow.
    sx, sy, sr = int(w * 0.8), int(h * 0.2), int(h * 0.075)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    glow = np.clip(1 - d / (sr * 5), 0, 1) ** 2.2
    img += glow[:, :, None] * np.array([60, 110, 120], np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.circle(img, (sx, sy), sr, (170, 240, 255), -1, cv2.LINE_AA)

    horizon = int(h * 0.62)

    def ridge(base_y, amp, color, jag):
        xs = np.linspace(0, w, 24)
        ys = base_y - amp * np.abs(rng.normal(0.5, 0.35, xs.size))
        # Mesa shapes: flatten tops by snapping heights to a few plateau levels.
        ys = np.round(ys / jag) * jag
        pts = [(0, horizon + 2)]
        for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
            pts.append((int(x), int(y)))
            if i + 1 < xs.size:
                pts.append((int((x + xs[i + 1]) / 2), int(y)))
        pts.append((w, horizon + 2))
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)

    ridge(horizon - h * 0.05, h * 0.16, (120, 150, 200), h * 0.04)  # far, hazy
    ridge(horizon - h * 0.01, h * 0.10, (70, 105, 165), h * 0.03)  # near

    # Sand with subtle grain and darker toward the bottom.
    sand = np.zeros((h - horizon, w, 3), np.float32)
    g = np.linspace(0, 1, h - horizon, dtype=np.float32)[:, None, None]
    sand[:] = np.array([110, 170, 220], np.float32) * (1 - 0.35 * g)
    sand += rng.normal(0, 6, sand.shape[:2])[:, :, None]
    img[horizon:] = np.clip(sand, 0, 255).astype(np.uint8)

    # A few cactus silhouettes on the horizon.
    for cx in (int(w * 0.12), int(w * 0.33), int(w * 0.9)):
        ch = int(h * rng.uniform(0.06, 0.1))
        cw = max(4, ch // 7)
        col = (45, 85, 60)
        base = horizon + int(h * 0.01)
        cv2.rectangle(img, (cx - cw // 2, base - ch), (cx + cw // 2, base), col, -1)
        cv2.circle(img, (cx, base - ch), cw // 2, col, -1, cv2.LINE_AA)
        for side in (-1, 1):
            ay = base - int(ch * rng.uniform(0.45, 0.7))
            ax = cx + side * cw * 2
            cv2.rectangle(img, (min(cx, ax), ay - cw // 2), (max(cx, ax), ay + cw // 2), col, -1)
            cv2.rectangle(img, (ax - cw // 2, ay - ch // 3), (ax + cw // 2, ay), col, -1)
            cv2.circle(img, (ax, ay - ch // 3), cw // 2, col, -1, cv2.LINE_AA)
    return img


def wooden_wall(w: int, h: int, seed: int = 3) -> np.ndarray:
    """Plank wall texture with grain, gaps and nails, plus a lit top rail. BGR, size (h, w)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    rail = max(10, h // 9)
    plank_w = max(60, w // 14)
    x = 0
    while x < w:
        pw = int(plank_w * rng.uniform(0.85, 1.15))
        base = np.array([45, 95, 150], np.float32) * rng.uniform(0.8, 1.1)
        # Grain: long vertical streaks (low frequency down the plank, high across it) plus fine noise.
        streaks = cv2.resize(rng.normal(0, 1, (4, max(4, pw // 2))).astype(np.float32), (pw, h - rail),
                             interpolation=cv2.INTER_CUBIC)  # fmt: skip
        fine = rng.normal(0, 1, (h - rail, pw)).astype(np.float32)
        plank = base + (streaks * 10 + fine * 3)[:, :, None]
        shade = np.linspace(1.08, 0.88, pw, dtype=np.float32)[None, :, None]  # rounded plank edges
        img[rail:, x : x + pw] = (plank * shade)[:, : max(0, min(pw, w - x))]
        img[rail:, x : x + 2] *= 0.45  # gap between planks
        for ny in (int(h * 0.3), int(h * 0.8)):  # nails
            cv2.circle(img, (x + pw // 2, ny), 3, (60, 60, 70), -1, cv2.LINE_AA)
        x += pw
    # Top rail: lighter, with a highlight and a shadow line under it.
    img[:rail] = np.array([60, 125, 185], np.float32)
    img[:3] = np.array([120, 185, 230], np.float32)
    img[rail : rail + 4] *= 0.5
    # Darken toward the bottom so it sits in the scene.
    img *= np.linspace(1.0, 0.7, h, dtype=np.float32)[:, None, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def vignette(w: int, h: int, strength: float = 0.35) -> np.ndarray:
    """Per-pixel multiplier (uint8, 255 = unchanged) darkening the corners."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    m = 1 - strength * np.clip(d - 0.35, 0, 1) ** 1.5 / 0.65**1.5
    return cv2.merge([(m * 255).astype(np.uint8)] * 3)
