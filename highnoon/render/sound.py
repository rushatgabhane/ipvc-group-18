from __future__ import annotations

import math
from pathlib import Path

SOUND_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "sounds"
SOUND_FILES = (
    "gunshot",
    "rocket_launch",
    "grenade_throw",
    "explosion",
    "reload",
    "dry_fire",
    "countdown",
)
MUZZLE_SOUNDS = {"": "gunshot", "rocket": "rocket_launch", "grenade": "grenade_throw"}
EFFECT_SOUNDS = {
    "explosion": "explosion",
    "reload": "reload",
    "dry_fire": "dry_fire",
}


def effect_sound(effect) -> str | None:
    if effect.kind == "muzzle":
        return MUZZLE_SOUNDS.get(effect.detail)
    return EFFECT_SOUNDS.get(effect.kind)


class SoundBoard:
    def __init__(self):
        import pyglet.media

        self._sources = {
            name: pyglet.media.load(str(SOUND_DIR / f"{name}.wav"), streaming=False) for name in SOUND_FILES
        }
        self._seen: set[int] = set()
        self._count: int | None = None

    def update(self, game, t: float) -> None:
        current = {id(effect) for effect in game.effects}
        for effect in game.effects:
            if id(effect) not in self._seen and (name := effect_sound(effect)) is not None:
                self._sources[name].play()
        self._seen = current
        count = math.ceil(game.countdown_left(t)) if game.mode == "countdown" else None
        if count and count != self._count:
            self._sources["countdown"].play()
        self._count = count
