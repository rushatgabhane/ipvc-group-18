from __future__ import annotations

import math

import cv2
import numpy as np

from highnoon.game.game import Game
from highnoon.render import assets
from highnoon.render.assets import HUD_FONT, TITLE_FONT, TextCache, blit_center
from highnoon.render.fx import FX
from highnoon.render.palette import BLACK, CREAM, RED, SAFE_GREEN, WHITE, point

MUZZLE_FLASH_S = 0.09
EXPLOSION_S = 0.55
POWERUP_TRAIL = ((3, 0.46), (2, 0.64), (1, 0.82))
ROCKET_FLAME = ((44, 22, (20, 70, 140)), (36, 13, (40, 150, 255)), (30, 7, (180, 240, 255)))
FIREBALL = ((1.0, (20, 60, 200)), (0.75, (30, 140, 255)), (0.45, (170, 235, 255)))
MUZZLE_RINGS = ((46, 0.25), (30, 0.5), (16, 1.0))
AIM_DOTS = 6
AIM_DOT_GAP = 22
AIM_DOT_START = 18


class WorldView:
    def __init__(self, colors: dict[int, tuple[int, int, int]], text: TextCache, fx: FX):
        self.colors = colors
        self.text = text
        self.fx = fx
        self._glow: np.ndarray | None = None

    def draw(self, image: np.ndarray, game: Game, t: float, dt: float) -> None:
        scale = image.shape[0] / 720.0
        self._draw_powerup(image, game, t, scale)
        self._emit_trails(game, t, dt)
        self._draw_particles(image)
        self._draw_glow(image, game, t, scale)
        self._draw_projectiles(image, game, t)
        self._draw_shockwaves(image, game, t, scale)
        self._draw_grenade_warnings(image, game, t, scale)
        for player_id, aim in game.aims.items():
            if aim.path is not None:
                self._draw_reticle(image, aim, self.colors.get(player_id, WHITE), t, scale)

    def _draw_powerup(self, image, game: Game, t: float, scale: float) -> None:
        powerup = game.powerup
        if powerup is None:
            return
        color = assets.POWERUP_COLORS[powerup.kind]
        for steps_back, size in POWERUP_TRAIL:
            past = powerup.center(max(powerup.t_spawn, t - 0.035 * steps_back))
            cv2.circle(image, point(past), int(powerup.radius * size), color, -1, cv2.LINE_AA)
        x, y = powerup.center(t)
        blit_center(image, assets.glow_sprite(int(powerup.radius * 4), color), x, y, alpha=0.6)
        wobble = math.sin((t - powerup.t_spawn) * 3) * 25
        token = assets.rotated(assets.powerup_sprite(int(powerup.radius * 2.2), powerup.kind), wobble)
        blit_center(image, token, x, y)
        label = self.text.get(assets.POWERUP_LABELS[powerup.kind], int(24 * scale), color, HUD_FONT, stroke=2)
        blit_center(image, label, x, y - powerup.radius - 16 * scale)

    def _emit_trails(self, game: Game, t: float, dt: float) -> None:
        if dt <= 0:
            return
        particles = self.fx.particles
        for rocket in game.rockets:
            backward = -unit(rocket.heading(t))
            exhaust = rocket.pos + backward * 34
            angle = math.atan2(backward[1], backward[0])
            particles.burst(
                exhaust, 3, (150, 150, 155), (10, 60), (0.5, 0.9), (5, 9), 1.2, angle, False, 0.25
            )
            particles.burst(exhaust, 2, (40, 160, 255), (80, 200), (0.05, 0.15), (2, 4), 0.6, angle, False)
        for grenade in game.grenades:
            particles.burst(grenade.pos, 2, (100, 220, 255), (30, 120), (0.15, 0.35), (1, 3))
            particles.burst(grenade.pos, 1, (120, 120, 125), (5, 30), (0.4, 0.7), (4, 7), gravity=False)

    def _draw_particles(self, image) -> None:
        particles = self.fx.particles
        if not len(particles.life):
            return
        radii = np.maximum(1, (particles.size * (0.4 + 0.6 * particles.fade())).astype(np.int32)).tolist()
        centers = particles.pos.astype(np.int32).tolist()
        colors = particles.color.astype(np.int32).tolist()
        for center, radius, color in zip(centers, radii, colors, strict=True):
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)

    def _draw_glow(self, image, game: Game, t: float, scale: float) -> None:
        muzzles = [e for e in game.effects if e.kind == "muzzle" and t - e.t < MUZZLE_FLASH_S]
        blasts = [e for e in game.effects if e.kind == "explosion" and t - e.t < EXPLOSION_S]
        if not (game.aims or game.bullets or game.rockets or muzzles or blasts):
            return
        if self._glow is None or self._glow.shape != image.shape:
            self._glow = np.zeros_like(image)
        glow = self._glow
        glow[:] = 0
        for player_id, aim in game.aims.items():
            if aim.path is not None:
                draw_arc_preview(glow, aim, self.colors.get(player_id, WHITE), t, scale)
            else:
                draw_laser(glow, aim, self.colors.get(player_id, WHITE), t, scale)
        for bullet in game.bullets:
            draw_tracer(glow, bullet, self.colors.get(bullet.shooter, WHITE), t, scale)
        for rocket in game.rockets:
            draw_rocket_flame(glow, rocket, t, scale)
        for blast in blasts:
            draw_fireball(glow, blast, game.blast_radius(blast.detail == "grenade"), t)
        for muzzle in muzzles:
            draw_muzzle_flash(glow, muzzle, t, scale)
        cv2.add(image, glow, dst=image)

    def _draw_projectiles(self, image, game: Game, t: float) -> None:
        height = image.shape[0]
        for rocket in game.rockets:
            heading = rocket.heading(t)
            angle = -math.degrees(math.atan2(heading[1], heading[0]))
            blit_center(image, assets.rotated(assets.rocket_sprite(int(0.13 * height)), angle), *rocket.pos)
        for grenade in game.grenades:
            spin = -(t - grenade.t_throw) * 540
            blit_center(image, assets.rotated(assets.grenade_sprite(int(0.075 * height)), spin), *grenade.pos)

    def _draw_shockwaves(self, image, game: Game, t: float, scale: float) -> None:
        for effect in game.effects:
            age = t - effect.t
            if effect.kind != "explosion" or age >= EXPLOSION_S:
                continue
            progress = age / EXPLOSION_S
            radius = int(game.blast_radius(effect.detail == "grenade") * (0.3 + 1.2 * progress))
            thickness = max(1, int(10 * scale * (1 - progress)))
            cv2.circle(image, point(effect.pos), radius, CREAM, thickness, cv2.LINE_AA)
            if progress < 0.5:
                cv2.circle(
                    image, point(effect.pos), int(radius * 0.7), WHITE, max(1, thickness // 2), cv2.LINE_AA
                )

    def _draw_grenade_warnings(self, image, game: Game, t: float, scale: float) -> None:
        for grenade in game.grenades:
            state = game.states.get(grenade.target)
            safe = state is not None and state.frame.ducked
            flash_rate = 4 + 10 * grenade.progress(t)
            if not safe and int(t * flash_rate) % 2:
                continue
            color = SAFE_GREEN if safe else RED
            center = point(grenade.landing)
            radius = int((34 + 10 * math.sin(t * 14)) * scale)
            cv2.circle(image, center, radius + 2, BLACK, max(3, int(5 * scale)), cv2.LINE_AA)
            draw_crosshair(image, center, radius, color, t * 2, 0.5, 1.3, max(2, int(3 * scale)))
            label = self.text.get("SAFE" if safe else "DUCK!", int(46 * scale), color, TITLE_FONT, stroke=3)
            blit_center(image, label, center[0], center[1] - radius - 30 * scale)

    def _draw_reticle(self, image, aim, player_color, t: float, scale: float) -> None:
        color = player_color
        center = point(aim.end)
        pulse = 1 + 0.08 * math.sin(t * 12) if aim.on_target else 1.0
        radius = int((20 if aim.on_target else 14) * scale * pulse)
        cv2.circle(image, center, radius + 2, BLACK, 3, cv2.LINE_AA)
        draw_crosshair(image, center, radius, color, t * (3.0 if aim.on_target else 1.0), 0.55, 1.45, 2)
        cv2.circle(image, center, 2, color, -1, cv2.LINE_AA)
        spread_px = int(np.linalg.norm(aim.end - aim.origin) * math.tan(math.radians(aim.spread_deg)))
        if spread_px > radius:
            cv2.circle(image, center, spread_px + 1, BLACK, 3, cv2.LINE_AA)
            cv2.circle(image, center, spread_px, color, 2, cv2.LINE_AA)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def scaled_color(color, factor: float) -> tuple[float, float, float]:
    return tuple(float(c) * factor for c in color)


def draw_crosshair(
    image, center, radius: int, color, spin: float, inner: float, outer: float, thickness: int
):
    cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)
    for quarter in range(4):
        angle = spin + quarter * math.pi / 2
        dx, dy = math.cos(angle) * radius, math.sin(angle) * radius
        start = (int(center[0] + dx * inner), int(center[1] + dy * inner))
        end = (int(center[0] + dx * outer), int(center[1] + dy * outer))
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def draw_laser(glow, aim, color, t: float, scale: float) -> None:
    brightness = 1.0 if aim.on_target else 0.85
    halo, core = scaled_color(color, 0.4 * brightness), scaled_color(color, brightness)
    steps = (AIM_DOT_START + AIM_DOT_GAP * np.arange(AIM_DOTS)) * scale
    reach = float(np.linalg.norm(aim.end - aim.origin))
    dots = aim.origin + np.outer(steps[steps < reach], unit(aim.end - aim.origin))
    for dot in dots.astype(np.int32).tolist():
        cv2.circle(glow, dot, max(3, int(5 * scale)), halo, -1, cv2.LINE_AA)
        cv2.circle(glow, dot, max(2, int(2.5 * scale)), core, -1, cv2.LINE_AA)


def draw_arc_preview(glow, aim, color, t: float, scale: float) -> None:
    brightness = 1.0 if aim.on_target else 0.9
    halo, core = scaled_color(color, 0.4 * brightness), scaled_color(color, brightness)
    phase = int(t * 12) % 2
    for dot in aim.path[1 + phase : -1 : 2].astype(np.int32).tolist():
        cv2.circle(glow, dot, max(3, int(6 * scale)), halo, -1, cv2.LINE_AA)
        cv2.circle(glow, dot, max(2, int(3 * scale)), core, -1, cv2.LINE_AA)


def draw_tracer(glow, bullet, color, t: float, scale: float) -> None:
    progress = bullet.progress(t)
    head = point(bullet.origin + bullet.direction * bullet.length * progress)
    tail = point(bullet.origin + bullet.direction * bullet.length * max(0.0, progress - 0.35))
    cv2.line(glow, tail, head, scaled_color(color, 0.45), max(3, int(12 * scale)), cv2.LINE_AA)
    cv2.line(glow, tail, head, scaled_color(color, 1.0), max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.line(glow, tail, head, (255, 255, 255), 1, cv2.LINE_AA)


def draw_rocket_flame(glow, rocket, t: float, scale: float) -> None:
    backward = -unit(rocket.heading(t))
    flicker = 0.8 + 0.2 * math.sin(t * 60)
    for distance, radius, color in ROCKET_FLAME:
        cv2.circle(
            glow,
            point(rocket.pos + backward * distance * scale),
            int(radius * scale * flicker),
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_fireball(glow, blast, blast_radius: float, t: float) -> None:
    progress = (t - blast.t) / EXPLOSION_S
    fade = (1 - progress) ** 1.5
    growth = 0.35 + 0.75 * math.sqrt(progress)
    for size, color in FIREBALL:
        cv2.circle(
            glow,
            point(blast.pos),
            int(blast_radius * size * growth),
            scaled_color(color, fade),
            -1,
            cv2.LINE_AA,
        )


def draw_muzzle_flash(glow, muzzle, t: float, scale: float) -> None:
    strength = 1 - (t - muzzle.t) / MUZZLE_FLASH_S
    for radius, intensity in MUZZLE_RINGS:
        color = scaled_color((120, 220, 255), intensity * strength)
        cv2.circle(
            glow, point(muzzle.pos), int(radius * scale * (0.6 + 0.4 * strength)), color, -1, cv2.LINE_AA
        )
