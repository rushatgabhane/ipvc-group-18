from __future__ import annotations

import io
import math
import wave

import numpy as np
from scipy.signal import lfilter

RATE = 44100
VOLUME = 0.6


def seconds(duration: float) -> np.ndarray:
    return np.arange(int(RATE * duration), dtype=np.float32) / RATE


def decay(t: np.ndarray, time_constant: float) -> np.ndarray:
    return np.exp(-t / time_constant)


def tone(freq: float, duration: float, time_constant: float) -> np.ndarray:
    t = seconds(duration)
    return np.sin(2 * np.pi * freq * t) * decay(t, time_constant)


def sweep(start_hz: float, end_hz: float, duration: float, time_constant: float) -> np.ndarray:
    t = seconds(duration)
    freq = start_hz * (end_hz / start_hz) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    return np.sin(phase) * decay(t, time_constant)


def noise(duration: float, time_constant: float, smoothing: float, seed: int) -> np.ndarray:
    t = seconds(duration)
    white = np.random.default_rng(seed).uniform(-1, 1, t.size)
    filtered = lfilter([1 - smoothing], [1, -smoothing], white)
    return filtered / max(1e-6, float(np.abs(filtered).max())) * decay(t, time_constant)


def square(freq: float, duration: float, time_constant: float) -> np.ndarray:
    return np.sign(tone(freq, duration, 1e9)) * decay(seconds(duration), time_constant)


def mix(*layers: tuple[np.ndarray, float]) -> np.ndarray:
    out = np.zeros(max(layer.size for layer, _ in layers), np.float32)
    for layer, gain in layers:
        out[: layer.size] += gain * layer
    return out


def sequence(*notes: tuple[np.ndarray, float]) -> np.ndarray:
    total = max(int(start * RATE) + note.size for note, start in notes)
    out = np.zeros(total, np.float32)
    for note, start in notes:
        offset = int(start * RATE)
        out[offset : offset + note.size] += note
    return out


def gun_shot() -> np.ndarray:
    return mix(
        (noise(0.35, 0.04, 0.55, 1), 0.9), (tone(85, 0.3, 0.07), 0.8), (noise(0.02, 0.004, 0.0, 2), 0.5)
    )


def rocket_launch() -> np.ndarray:
    whoosh = noise(0.7, 0.3, 0.9, 3) * np.linspace(0.3, 1, seconds(0.7).size)
    return mix((whoosh, 1.0), (sweep(140, 420, 0.7, 0.35), 0.4), (tone(70, 0.25, 0.06), 0.7))


def grenade_throw() -> np.ndarray:
    return mix((sweep(1900, 650, 0.9, 0.6), 0.45), (noise(0.15, 0.05, 0.7, 4), 0.3))


def hit() -> np.ndarray:
    return mix((tone(130, 0.25, 0.05), 1.0), (noise(0.12, 0.025, 0.8, 5), 0.6))


def headshot() -> np.ndarray:
    bell = mix((tone(1320, 0.6, 0.18), 0.5), (tone(1980, 0.6, 0.12), 0.3), (tone(2640, 0.4, 0.07), 0.2))
    return sequence((hit(), 0.0), (bell, 0.03))


def ricochet() -> np.ndarray:
    return mix((sweep(2600, 900, 0.3, 0.12), 0.5), (noise(0.03, 0.008, 0.2, 6), 0.5))


def explosion() -> np.ndarray:
    rumble = noise(1.4, 0.35, 0.985, 7)
    crack = noise(0.15, 0.03, 0.5, 8)
    return mix((rumble, 1.0), (crack, 0.7), (tone(48, 1.0, 0.3), 0.8))


def dry_fire() -> np.ndarray:
    return mix((noise(0.03, 0.004, 0.1, 9), 0.6), (tone(3200, 0.03, 0.006), 0.4))


def reload() -> np.ndarray:
    click = mix((noise(0.04, 0.006, 0.3, 10), 0.7), (tone(950, 0.06, 0.015), 0.5))
    slide = mix((noise(0.06, 0.01, 0.5, 11), 0.7), (tone(620, 0.08, 0.02), 0.5))
    return sequence((click, 0.0), (slide, 0.13))


def pickup() -> np.ndarray:
    notes = [(square(freq, 0.12, 0.05) * 0.35, i * 0.07) for i, freq in enumerate((660, 880, 1320))]
    return sequence(*notes)


def safe() -> np.ndarray:
    return sequence((tone(660, 0.15, 0.06), 0.0), (tone(990, 0.25, 0.09), 0.09))


def camping() -> np.ndarray:
    return square(110, 0.18, 0.08) * 0.35


def beep() -> np.ndarray:
    return tone(880, 0.2, 0.08) * 0.6


def go() -> np.ndarray:
    return mix((tone(1760, 0.5, 0.18), 0.5), (tone(880, 0.5, 0.18), 0.3))


def fanfare() -> np.ndarray:
    notes = [(square(freq, 0.35, 0.15) * 0.3, i * 0.13) for i, freq in enumerate((523, 659, 784, 1047))]
    return sequence(*notes, (square(1047, 0.8, 0.35) * 0.3, 0.52))


SOUNDS = {
    "shot": gun_shot,
    "rocket_launch": rocket_launch,
    "grenade_throw": grenade_throw,
    "hit": hit,
    "headshot": headshot,
    "blocked": ricochet,
    "explosion": explosion,
    "dry_fire": dry_fire,
    "reload": reload,
    "pickup": pickup,
    "safe": safe,
    "camping": camping,
    "beep": beep,
    "go": go,
    "win": fanfare,
}
MUZZLE_SOUNDS = {"": "shot", "rocket": "rocket_launch", "grenade": "grenade_throw"}


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
    return effect.kind if effect.kind in SOUNDS else None


class SoundBoard:
    def __init__(self):
        import pyglet.media

        self._sources = {
            name: pyglet.media.load(f"{name}.wav", file=wav_bytes(make()), streaming=False)
            for name, make in SOUNDS.items()
        }
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
