"""Short-lived visual effects: particles, floating numbers, announcements, screen shake.

Purely cosmetic and owned by the renderer. Game logic emits `Effect` events; the FX system
turns each new event into particles once and animates them with simple physics.
Particles are stored as numpy arrays and updated together, so the cost stays flat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 900.0  # px/s^2


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
    def __init__(self, capacity: int = 600, seed: int = 0):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)  # seconds left
        self.max_life = np.zeros(0, np.float32)
        self.color = np.zeros((0, 3), np.float32)
        self.size = np.zeros(0, np.float32)
        self.drag = np.zeros(0, np.float32)
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

    def burst(self, pos, n, color, speed=(150, 450), life=(0.3, 0.7), size=(2, 5), spread=np.pi * 2,
              direction=0.0, gravity=True, jitter=0.15):  # fmt: skip
        rng = self._rng
        ang = direction + rng.uniform(-spread / 2, spread / 2, n)
        spd = rng.uniform(*speed, n)
        vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1).astype(np.float32)
        col = np.clip(np.array(color, np.float32) * rng.uniform(1 - jitter, 1 + jitter, (n, 1)), 0, 255)
        lf = rng.uniform(*life, n).astype(np.float32)
        self.pos = np.concatenate([self.pos, np.tile(np.asarray(pos, np.float32), (n, 1))])[-self.capacity :]
        self.vel = np.concatenate([self.vel, vel])[-self.capacity :]
        self.life = np.concatenate([self.life, lf])[-self.capacity :]
        self.max_life = np.concatenate([self.max_life, lf])[-self.capacity :]
        self.color = np.concatenate([self.color, col.astype(np.float32)])[-self.capacity :]
        self.size = np.concatenate([self.size, rng.uniform(*size, n).astype(np.float32)])[-self.capacity :]
        drag = np.full(n, 0.0 if gravity else 3.0, np.float32)
        self.drag = np.concatenate([self.drag, drag])[-self.capacity :]

    def update(self, dt: float) -> None:
        if not len(self.life):
            return
        dt = min(dt, 0.05)
        falling = self.drag == 0
        self.vel[falling, 1] += GRAVITY * dt
        self.vel *= (1 - np.minimum(self.drag * dt, 0.9))[:, None]
        self.pos += self.vel * dt
        self.life -= dt
        keep = self.life > 0
        for name in ("pos", "vel", "life", "max_life", "color", "size", "drag"):
            setattr(self, name, getattr(self, name)[keep])

    def fade(self) -> np.ndarray:
        return np.clip(self.life / np.maximum(self.max_life, 1e-6), 0, 1)


class FX:
    """Turns game effects into particles/text and tracks screen shake."""

    def __init__(self):
        self.particles = Particles()
        self.texts: list[FloatText] = []
        self.announcements: list[Announcement] = []
        self._seen: set[int] = set()
        self._shake_t = -1.0
        self._shake_amp = 0.0
        self._t_last: float | None = None
        self._rng = np.random.default_rng(1)

    def update(self, t: float, effects, colors: dict) -> None:
        dt = 0.0 if self._t_last is None else t - self._t_last
        self._t_last = t
        current = set()
        for e in effects:
            key = id(e)
            current.add(key)
            if key not in self._seen:
                self._spawn(e, t, colors)
        self._seen = current  # forget effects the game has expired
        self.particles.update(dt)
        self.texts = [x for x in self.texts if t - x.t0 < x.duration]
        self.announcements = [a for a in self.announcements if t - a.t0 < a.duration]

    def announce(self, text: str, t: float, color, size: int = 96, duration: float = 1.6) -> None:
        self.announcements = [Announcement(text, t, color, duration, size)]

    def shake_offset(self, t: float) -> tuple[int, int]:
        age = t - self._shake_t
        if age < 0 or age > 0.3:
            return 0, 0
        amp = self._shake_amp * (1 - age / 0.3)
        return int(self._rng.uniform(-amp, amp)), int(self._rng.uniform(-amp, amp))

    def _spawn(self, e, t: float, colors: dict) -> None:
        p = self.particles
        pos = e.pos
        if e.kind == "muzzle":
            p.burst(pos, 10, (120, 230, 255), speed=(200, 500), life=(0.05, 0.15), size=(2, 4), gravity=False)
        elif e.kind in ("hit", "headshot"):
            head = e.kind == "headshot"
            p.burst(pos, 40 if head else 22, (50, 50, 230), speed=(150, 520), life=(0.3, 0.7), size=(2, 6))
            p.burst(pos, 12, (200, 230, 255), speed=(80, 250), life=(0.1, 0.25), size=(2, 3), gravity=False)
            dmg = "-35" if head else "-15"
            self.texts.append(FloatText(dmg, pos.copy(), t, (80, 80, 255), 56 if head else 44))
            self._shake_t, self._shake_amp = t, 14.0 if head else 8.0
            if head:
                self.announce("HEADSHOT!", t, (60, 60, 240), size=84, duration=1.1)
        elif e.kind == "blocked":
            p.burst(pos, 18, (60, 120, 180), speed=(120, 380), life=(0.4, 0.9), size=(2, 5),
                    spread=np.pi * 0.9, direction=-np.pi / 2)  # fmt: skip
            p.burst(pos, 8, (150, 230, 255), speed=(80, 200), life=(0.05, 0.15), size=(2, 3), gravity=False)
            self.texts.append(FloatText("BLOCKED", pos.copy(), t, (180, 220, 255), 32, 0.7))
        elif e.kind == "shatter":
            p.burst(pos, 30, (90, 180, 80), speed=(150, 450), life=(0.4, 0.9), size=(2, 5))
            p.burst(pos, 10, (230, 250, 230), speed=(100, 300), life=(0.2, 0.5), size=(1, 3))
            self.texts.append(FloatText("+1", pos.copy(), t, (120, 240, 140), 52))
