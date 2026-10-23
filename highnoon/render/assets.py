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


POWERUP_COLORS = {
    "heal": (70, 70, 225),
    "double": (40, 190, 250),
    "rocket": (30, 120, 240),
    "grenade": (60, 150, 70),
}
POWERUP_LABELS = {"heal": "FULL HP", "double": "2X DAMAGE", "rocket": "ROCKET LAUNCHER", "grenade": "GRENADE"}


@lru_cache(maxsize=8)
def powerup_sprite(size: int, kind: str) -> np.ndarray:
    """Round token with a white icon (cross, "2X", rocket or grenade), BGRA, drawn at 4x for smooth edges."""
    s = 4
    S = size * s
    img = np.zeros((S, S, 4), np.uint8)
    c, r = S // 2, S // 2 - 2 * s
    color = POWERUP_COLORS[kind]
    dark = tuple(int(v * 0.55) for v in color)
    cv2.circle(img, (c, c), r, dark + (255,), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(r * 0.86), color + (255,), -1, cv2.LINE_AA)
    shine = (int(r * 0.7), int(r * 0.7))
    cv2.ellipse(img, (c, c), shine, 0, 200, 290, (255, 255, 255, 140), 3 * s, cv2.LINE_AA)
    white = (255, 255, 255, 255)
    k = r * 0.42
    if kind == "heal":
        arm = int(k * 0.36)
        cv2.rectangle(img, (int(c - k), c - arm), (int(c + k), c + arm), white, -1)
        cv2.rectangle(img, (c - arm, int(c - k)), (c + arm, int(c + k)), white, -1)
    elif kind == "rocket":
        _draw_token_rocket(img, c, k * 2.1)
    elif kind == "grenade":
        cv2.ellipse(
            img, (c, int(c + k * 0.2)), (int(k * 0.72), int(k * 0.9)), 0, 0, 360, white, -1, cv2.LINE_AA
        )
        cv2.rectangle(
            img, (int(c - k * 0.3), int(c - k * 0.95)), (int(c + k * 0.3), int(c - k * 0.6)), white, -1
        )
        cv2.circle(
            img, (int(c + k * 0.55), int(c - k * 0.95)), int(k * 0.25), white, max(2, s * 2), cv2.LINE_AA
        )
    else:
        scale = r / 55.0
        (tw, th), _ = cv2.getTextSize("2X", cv2.FONT_HERSHEY_DUPLEX, scale, int(4 * s))
        cv2.putText(img, "2X", (c - tw // 2, c + th // 2), cv2.FONT_HERSHEY_DUPLEX, scale, white, int(4 * s),
                    cv2.LINE_AA)  # fmt: skip
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _draw_token_rocket(img: np.ndarray, c: int, size: float) -> None:
    """White rocket silhouette flying up-right, with a yellow flame."""
    u = np.array([0.7071, -0.7071])  # along the rocket (up-right on screen)
    v = np.array([0.7071, 0.7071])  # across it

    def poly(points):
        pts = [c + size * (a * u + b * v) for a, b in points]
        return np.array(pts, np.float32).round().astype(np.int32)

    cv2.fillPoly(img, [poly([(-0.42, -0.08), (-0.7, 0.0), (-0.42, 0.08)])], (60, 220, 255, 255), cv2.LINE_AA)
    white = (255, 255, 255, 255)
    body = [(-0.42, -0.11), (0.2, -0.11), (0.48, 0.0), (0.2, 0.11), (-0.42, 0.11)]
    cv2.fillPoly(img, [poly(body)], white, cv2.LINE_AA)
    for side in (-1, 1):
        fin = [(-0.42, 0.11 * side), (-0.2, 0.11 * side), (-0.46, 0.3 * side)]
        cv2.fillPoly(img, [poly(fin)], white, cv2.LINE_AA)


@lru_cache(maxsize=4)
def grenade_sprite(height: int) -> np.ndarray:
    """Olive "pineapple" grenade with segment grooves, lever and pin ring, on a square canvas
    (for spinning). BGRA, drawn at 4x and downsampled."""
    s = 4
    S = height * s
    img = np.zeros((S, S, 4), np.uint8)
    c = S // 2
    rx, ry = int(S * 0.28), int(S * 0.34)
    cy = c + int(S * 0.06)
    olive, dark, light = (50, 95, 70), (25, 50, 35), (120, 175, 140)
    cv2.ellipse(img, (c, cy), (rx, ry), 0, 0, 360, olive + (255,), -1, cv2.LINE_AA)
    for k in (-2, -1, 0, 1, 2):  # grooves
        y = cy + k * ry // 3
        half = int(rx * np.sqrt(max(0.0, 1 - (k / 3) ** 2)))
        cv2.line(img, (c - half, y), (c + half, y), dark + (255,), s)
    for k in (-1, 0, 1):
        cv2.ellipse(img, (c + k * rx // 2, cy), (max(1, rx // 6), ry), 0, 0, 360, dark + (255,), s)
    cv2.ellipse(
        img, (c - rx // 3, cy - ry // 3), (rx // 4, ry // 5), -30, 0, 360, light + (170,), -1, cv2.LINE_AA
    )
    # Fuse head, lever and pin ring.
    top = cy - ry
    cv2.rectangle(img, (c - rx // 3, top - int(S * 0.08)), (c + rx // 3, top + s), (90, 90, 95, 255), -1)
    lever = [(c + rx // 3, top - int(S * 0.08)), (c + rx // 2, top - int(S * 0.1)),
             (c + int(rx * 0.9), cy - ry // 4), (c + int(rx * 0.75), cy - ry // 4)]  # fmt: skip
    lever = np.array(lever, np.int32)
    cv2.fillPoly(img, [lever], (120, 120, 125, 255), cv2.LINE_AA)
    cv2.circle(
        img, (c - rx // 2, top - int(S * 0.06)), int(S * 0.06), (170, 200, 215, 255), 2 * s, cv2.LINE_AA
    )
    alpha = img[:, :, 3].copy()
    edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((2 * s, 2 * s), np.uint8))
    img[edge > 0, :3] = (15, 30, 20)
    return cv2.resize(img, (height, height), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=4)
def rocket_sprite(length: int) -> np.ndarray:
    """Olive rocket with a red nose and fins, pointing right, on a square canvas so it can be
    rotated to any heading without clipping. BGRA, drawn at 4x and downsampled."""
    s = 4
    L = length * s
    img = np.zeros((L, L, 4), np.uint8)
    cy = L // 2
    bh = int(L * 0.075)  # half the body height
    body, body_dark, red, red_dark = (75, 115, 95), (45, 70, 55), (45, 45, 205), (30, 30, 140)
    x_tail, x_nose, x_tip = int(L * 0.12), int(L * 0.72), int(L * 0.9)
    # Fins behind the body, nozzle, body, nose cone.
    for side in (-1, 1):
        fin = np.array([[x_tail, cy + side * bh], [x_tail + int(L * 0.2), cy + side * bh],
                        [x_tail - int(L * 0.02), cy + side * int(L * 0.19)]], np.int32)  # fmt: skip
        cv2.fillPoly(img, [fin], red + (255,), cv2.LINE_AA)
    cv2.rectangle(img, (x_tail - int(L * 0.05), cy - int(bh * 0.7)), (x_tail, cy + int(bh * 0.7)),
                  (60, 60, 60, 255), -1)  # fmt: skip
    cv2.rectangle(img, (x_tail, cy - bh), (x_nose, cy + bh), body + (255,), -1)
    nose = np.array([[x_nose, cy - bh], [x_tip, cy], [x_nose, cy + bh]], np.int32)
    cv2.fillPoly(img, [nose], red + (255,), cv2.LINE_AA)
    # Band, shading underneath, highlight on top, then a dark outline.
    cv2.rectangle(img, (int(L * 0.58), cy - bh), (int(L * 0.62), cy + bh), (200, 225, 235, 255), -1)
    cv2.rectangle(img, (x_tail, cy + bh // 3), (x_nose, cy + bh), body_dark + (255,), -1)
    cv2.line(img, (x_tail + 2 * s, cy - bh // 2), (x_nose - 2 * s, cy - bh // 2), (170, 210, 190, 255), 2 * s)
    alpha = img[:, :, 3].copy()
    edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((2 * s, 2 * s), np.uint8))
    img[edge > 0, :3] = red_dark
    return cv2.resize(img, (length, length), interpolation=cv2.INTER_AREA)


def rotated(sprite: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a BGRA sprite about its centre (same size; for small sprites only)."""
    h, w = sprite.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(sprite, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


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
