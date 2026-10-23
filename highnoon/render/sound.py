from __future__ import annotations

from pathlib import Path

SOUND_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "sounds"
SOUND_FILES = (
    "gunshot",
    "rocket_launch",
    "grenade_throw",
    "explosion",
    "reload",
    "dry_fire",
    "fight",
    "ko",
)
MUZZLE_SOUNDS = {"gun": "gunshot", "rocket": "rocket_launch", "grenade": "grenade_throw"}
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
        self._mode: str | None = None
        self._winner: int | None = None

    def update(self, game, t: float) -> None:
        current = {id(effect) for effect in game.effects}
        for effect in game.effects:
            if id(effect) not in self._seen and (name := effect_sound(effect)) is not None:
                self._sources[name].play()
        self._seen = current
        if game.mode == "duel" and self._mode == "countdown":
            self._sources["fight"].play()
        if game.winner is not None and self._winner is None:
            self._sources["ko"].play()
        self._mode, self._winner = game.mode, game.winner
