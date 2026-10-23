from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from highnoon.game.actions import ActionFrame, PlayerActions


def falling(origin: np.ndarray, velocity: np.ndarray, gravity: float, dt: float) -> np.ndarray:
    return origin + velocity * dt + np.array([0.0, 0.5 * gravity * dt * dt], np.float32)


def falling_path(origin: np.ndarray, velocity: np.ndarray, gravity: float, dts: np.ndarray) -> np.ndarray:
    path = origin + velocity * dts[:, None]
    path[:, 1] += 0.5 * gravity * dts * dts
    return path.astype(np.float32)


@dataclass
class PlayerState:
    actions: PlayerActions
    hp: int
    ammo: int
    frame: ActionFrame = field(default_factory=ActionFrame)
    last_hit_t: float = -math.inf
    rockets: int = 0
    grenades: int = 0
    next_camp_drain_t: float | None = None


@dataclass
class Hitbox:
    head_center: np.ndarray
    head_radius: float
    torso: np.ndarray


@dataclass
class Bullet:
    shooter: int
    origin: np.ndarray
    direction: np.ndarray
    length: float
    t_fire: float
    t_arrive: float

    def progress(self, t: float) -> float:
        return min(1.0, (t - self.t_fire) / max(self.t_arrive - self.t_fire, 1e-6))


@dataclass
class Rocket:
    shooter: int
    origin: np.ndarray
    velocity: np.ndarray
    gravity: float
    t_fire: float
    pos: np.ndarray

    def at(self, t: float) -> np.ndarray:
        return falling(self.origin, self.velocity, self.gravity, t - self.t_fire)

    def heading(self, t: float) -> np.ndarray:
        return self.velocity + np.array([0.0, self.gravity * (t - self.t_fire)], np.float32)


@dataclass
class Grenade:
    thrower: int
    target: int
    origin: np.ndarray
    t_throw: float
    t_land: float
    pos: np.ndarray
    landing: np.ndarray

    def progress(self, t: float) -> float:
        return min(1.0, (t - self.t_throw) / (self.t_land - self.t_throw))


@dataclass
class Powerup:
    kind: str
    origin: np.ndarray
    velocity: np.ndarray
    gravity: float
    radius: float
    t_spawn: float
    t_end: float

    def center(self, t: float) -> np.ndarray:
        return falling(self.origin, self.velocity, self.gravity, t - self.t_spawn)


@dataclass
class Effect:
    kind: str
    pos: np.ndarray
    t: float
    player: int
    amount: int = 0
    detail: str = ""


@dataclass
class Aim:
    origin: np.ndarray
    end: np.ndarray
    on_target: bool
    path: np.ndarray | None = None
