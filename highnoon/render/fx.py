from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highnoon.render.assets import POWERUP_COLORS, POWERUP_LABELS

GRAVITY = 900.0
MAX_STEP_S = 0.05
SHAKE_S = 0.3
UP = -np.pi / 2
GRENADE_SCALE = 1.4
BOOM = ("BOOM!", (40, 150, 255), 120)
GRENADE_BOOM = ("BOOM!", (40, 150, 255), int(120 * GRENADE_SCALE))
GRENADE_WARNING = ("GRENADE! DUCK!", (60, 60, 240), 84)
PICKUP_SIZE = 72


def headshot_announcement(color) -> tuple[str, tuple[int, int, int], int]:
    return "HEADSHOT!", color, 84


def pickup_announcement(player_id: int, kind: str, color) -> tuple[str, tuple[int, int, int], int]:
    return f"P{player_id}: {POWERUP_LABELS[kind]}", color, PICKUP_SIZE


@dataclass
class FloatText:
    text: str
    pos: np.ndarray
    t0: float
    color: tuple[int, int, int]
    size: int
    duration: float = 0.9


@dataclass
class Announcement:
    text: str
    t0: float
    color: tuple[int, int, int]
    duration: float = 1.6
    size: int = 96


class Particles:
    FIELDS = ("pos", "vel", "life", "max_life", "color", "size", "drag")

    def __init__(self, capacity: int = 600, seed: int = 0):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.max_life = np.zeros(0, np.float32)
        self.color = np.zeros((0, 3), np.float32)
        self.size = np.zeros(0, np.float32)
        self.drag = np.zeros(0, np.float32)
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

    def burst(
        self,
        pos,
        count: int,
        color,
        speed=(150, 450),
        life=(0.3, 0.7),
        size=(2, 5),
        spread=np.pi * 2,
        direction=0.0,
        gravity=True,
        jitter=0.15,
    ) -> None:
        rng = self._rng
        angles = direction + rng.uniform(-spread / 2, spread / 2, count)
        speeds = rng.uniform(*speed, count)
        lifetimes = rng.uniform(*life, count).astype(np.float32)
        shades = rng.uniform(1 - jitter, 1 + jitter, (count, 1))
        new = {
            "pos": np.tile(np.asarray(pos, np.float32), (count, 1)),
            "vel": np.stack([np.cos(angles) * speeds, np.sin(angles) * speeds], axis=1).astype(np.float32),
            "life": lifetimes,
            "max_life": lifetimes,
            "color": np.clip(np.array(color, np.float32) * shades, 0, 255).astype(np.float32),
            "size": rng.uniform(*size, count).astype(np.float32),
            "drag": np.full(count, 0.0 if gravity else 3.0, np.float32),
        }
        for name in self.FIELDS:
            setattr(self, name, np.concatenate([getattr(self, name), new[name]])[-self.capacity :])

    def update(self, dt: float) -> None:
        if not len(self.life):
            return
        dt = min(dt, MAX_STEP_S)
        self.vel[self.drag == 0, 1] += GRAVITY * dt
        self.vel *= (1 - np.minimum(self.drag * dt, 0.9))[:, None]
        self.pos += self.vel * dt
        self.life -= dt
        alive = self.life > 0
        for name in self.FIELDS:
            setattr(self, name, getattr(self, name)[alive])

    def fade(self) -> np.ndarray:
        return np.clip(self.life / np.maximum(self.max_life, 1e-6), 0, 1)


class FX:
    def __init__(self, player_colors: dict[int, tuple[int, int, int]]):
        self.player_colors = player_colors
        self.particles = Particles()
        self.texts: list[FloatText] = []
        self.announcements: list[Announcement] = []
        self._seen: set[int] = set()
        self._shake_t = -1.0
        self._shake_amp = 0.0
        self._t_last: float | None = None
        self._rng = np.random.default_rng(1)
        self._spawners = {
            "muzzle": self._muzzle,
            "hit": self._hit,
            "headshot": self._hit,
            "blocked": self._blocked,
            "explosion": self._explosion,
            "grenade": self._grenade_thrown,
            "safe": self._safe,
            "pickup": self._pickup,
            "camping": self._camping,
        }

    def update(self, t: float, effects) -> None:
        dt = 0.0 if self._t_last is None else t - self._t_last
        self._t_last = t
        current = {id(e) for e in effects}
        for effect in effects:
            if id(effect) not in self._seen:
                self._spawners[effect.kind](effect, t)
        self._seen = current
        self.particles.update(dt)
        self.texts = [text for text in self.texts if t - text.t0 < text.duration]
        self.announcements = [a for a in self.announcements if t - a.t0 < a.duration]

    def announce(self, text: str, t: float, color, size: int = 96, duration: float = 1.6) -> None:
        self.announcements = [Announcement(text, t, color, duration, size)]

    def _announce(self, announcement: tuple, t: float, duration: float) -> None:
        text, color, size = announcement
        self.announce(text, t, color, size, duration)

    def shake_offset(self, t: float) -> tuple[int, int]:
        age = t - self._shake_t
        if age < 0 or age > SHAKE_S:
            return 0, 0
        amplitude = self._shake_amp * (1 - age / SHAKE_S)
        return int(self._rng.uniform(-amplitude, amplitude)), int(self._rng.uniform(-amplitude, amplitude))

    def _shake(self, t: float, amplitude: float) -> None:
        self._shake_t, self._shake_amp = t, amplitude

    def _float_text(self, text: str, effect, t: float, color, size: int, duration: float = 0.9) -> None:
        self.texts.append(FloatText(text, effect.pos.copy(), t, color, size, duration))

    def _muzzle(self, effect, t: float) -> None:
        self.particles.burst(effect.pos, 10, (120, 230, 255), (200, 500), (0.05, 0.15), (2, 4), gravity=False)

    def _hit(self, effect, t: float) -> None:
        headshot = effect.kind == "headshot"
        burst = self.particles.burst
        burst(effect.pos, 40 if headshot else 22, (50, 50, 230), (150, 520), (0.3, 0.7), (2, 6))
        burst(effect.pos, 12, (200, 230, 255), (80, 250), (0.1, 0.25), (2, 3), gravity=False)
        self._float_text(f"-{effect.amount}", effect, t, (80, 80, 255), 56 if headshot else 44)
        self._shake(t, 14.0 if headshot else 8.0)
        if headshot:
            shooter_color = self.player_colors.get(effect.player, (60, 60, 240))
            self._announce(headshot_announcement(shooter_color), t, duration=1.1)

    def _blocked(self, effect, t: float) -> None:
        burst = self.particles.burst
        burst(
            effect.pos, 18, (60, 120, 180), (120, 380), (0.4, 0.9), (2, 5), spread=np.pi * 0.9, direction=UP
        )
        burst(effect.pos, 8, (150, 230, 255), (80, 200), (0.05, 0.15), (2, 3), gravity=False)
        self._float_text("BLOCKED", effect, t, (180, 220, 255), 32, 0.7)

    def _explosion(self, effect, t: float) -> None:
        scale = GRENADE_SCALE if effect.detail == "grenade" else 1.0

        def burst(count, color, speed, life, size, **kwargs):
            self.particles.burst(effect.pos, int(count * scale), color, speed, life, size, **kwargs)

        burst(30, (210, 245, 255), (300, 750), (0.08, 0.22), (3, 6), gravity=False)
        burst(55, (40, 150, 255), (150, 600), (0.3, 0.7), (4, 9), gravity=False)
        burst(30, (30, 70, 225), (100, 420), (0.4, 0.8), (3, 8))
        burst(24, (120, 225, 255), (450, 900), (0.3, 0.6), (1, 3))
        burst(30, (70, 72, 78), (30, 170), (0.8, 1.6), (7, 14), gravity=False, spread=np.pi, direction=UP)
        self._shake(t, 22.0 * scale)
        self._announce(GRENADE_BOOM if effect.detail == "grenade" else BOOM, t, duration=0.9)

    def _grenade_thrown(self, effect, t: float) -> None:
        self._announce(GRENADE_WARNING, t, duration=1.3)

    def _safe(self, effect, t: float) -> None:
        self._float_text("SAFE!", effect, t, (140, 240, 160), 52)

    def _camping(self, effect, t: float) -> None:
        self.particles.burst(effect.pos, 10, (60, 60, 235), (60, 200), (0.2, 0.5), (2, 4), gravity=False)
        self._float_text(f"-{effect.amount}", effect, t, (60, 140, 255), 40)

    def _pickup(self, effect, t: float) -> None:
        player_color = self.player_colors.get(effect.player, (255, 255, 255))
        self.particles.burst(effect.pos, 40, POWERUP_COLORS[effect.detail], (150, 500), (0.3, 0.8), (2, 6))
        self.particles.burst(effect.pos, 16, (255, 255, 255), (80, 260), (0.1, 0.3), (2, 3), gravity=False)
        self._float_text(POWERUP_LABELS[effect.detail], effect, t, player_color, 48)
        self._announce(pickup_announcement(effect.player, effect.detail, player_color), t, duration=1.2)
