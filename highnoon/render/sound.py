from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

RATE = 44100
VOLUME = 0.6
SOUND_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "sounds"
RECORDED = (
    "hit",
    "headshot",
    "blocked",
    "explosion",
    "dry_fire",
    "reload",
    "pickup",
    "safe",
    "camping",
    "beep",
    "go",
    "win",
)
RECORDED_FILES = {"shot": "gunshot", **{name: name for name in RECORDED}}


def seconds(duration: float) -> np.ndarray:
    return np.arange(int(RATE * duration), dtype=np.float32) / RATE


def noise(duration: float, time_constant: float, smoothing: float, seed: int) -> np.ndarray:
    t = seconds(duration)
    white = np.random.default_rng(seed).uniform(-1, 1, t.size)
    filtered = lfilter([1 - smoothing], [1, -smoothing], white)
    return filtered / max(1e-6, float(np.abs(filtered).max())) * np.exp(-t / time_constant)


def rocket_launch() -> np.ndarray:
    t = seconds(0.9)
    return noise(0.9, 0.5, 0.95, 3) * np.minimum(1, t / 0.15)


def grenade_throw() -> np.ndarray:
    t = seconds(0.5)
    return noise(0.5, 1e9, 0.8, 4) * np.sin(np.pi * t / 0.5) ** 2 * 0.7


SYNTHESISED = {"rocket_launch": rocket_launch, "grenade_throw": grenade_throw}
MUZZLE_SOUNDS = {"": "shot", "rocket": "rocket_launch", "grenade": "grenade_throw"}
EFFECT_SOUNDS = set(RECORDED)


def wav_bytes(samples: np.ndarray) -> io.BytesIO:
    peak = max(1.0, float(np.abs(samples).max()))
    pcm = (np.clip(samples / peak, -1, 1) * VOLUME * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    buffer.seek(0)
    return buffer


def effect_sound(effect) -> str | None:
    if effect.kind == "muzzle":
        return MUZZLE_SOUNDS.get(effect.detail, "shot")
    return effect.kind if effect.kind in EFFECT_SOUNDS else None


class SoundBoard:
    def __init__(self):
        import pyglet.media

        self._sources = {
            name: pyglet.media.load(str(SOUND_DIR / f"{file}.wav"), streaming=False)
            for name, file in RECORDED_FILES.items()
        }
        self._sources.update(
            {
                name: pyglet.media.load(f"{name}.wav", file=wav_bytes(make()), streaming=False)
                for name, make in SYNTHESISED.items()
            }
        )
        self._seen: set[int] = set()
        self._countdown_number: int | None = None
        self._mode: str | None = None
        self._winner: int | None = None

    def play(self, name: str) -> None:
        self._sources[name].play()

    def update(self, game, t: float) -> None:
        current = {id(effect) for effect in game.effects}
        for effect in game.effects:
            if id(effect) not in self._seen and (name := effect_sound(effect)) is not None:
                self.play(name)
        self._seen = current
        self._update_countdown(game, t)
        if game.winner is not None and self._winner is None:
            self.play("win")
        self._winner = game.winner

    def _update_countdown(self, game, t: float) -> None:
        number = math.ceil(game.countdown_left(t)) if game.mode == "countdown" else None
        if number is not None and number > 0 and number != self._countdown_number:
            self.play("beep")
        if game.mode == "duel" and self._mode == "countdown":
            self.play("go")
        self._countdown_number, self._mode = number, game.mode
