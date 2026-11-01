from __future__ import annotations

import random
import threading
import traceback


class ModuleGuard:
    """Runs a module's work so an exception costs that module one call, not the whole application.

    Modules run every frame, so each distinct error is printed once with its traceback. `faults` makes
    named modules fail on purpose (each call fails with the given probability) to test and demonstrate
    that the rest of the game keeps going: `--fail-module hand:0.5`.
    """

    def __init__(self, faults: dict[str, float] | None = None, seed: int | None = None):
        self.faults = dict(faults or {})
        self._rng = random.Random(seed)
        self.failures: dict[str, int] = {}
        self._seen: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()

    def call(self, name: str, fn, *args, fallback=None):
        try:
            if name in self.faults and self._rng.random() < self.faults[name]:
                raise InjectedFault(f"{name} failed on purpose (--fail-module)")
            return fn(*args)
        except Exception as error:
            self.report(name, error)
            return fallback

    def report(self, name: str, error: Exception) -> None:
        key = (name, type(error).__name__, str(error))
        with self._lock:
            self.failures[name] = self.failures.get(name, 0) + 1
            first = key not in self._seen
            self._seen.add(key)
        if first:
            details = "".join(traceback.format_exception(error))
            print(f"[highnoon] {name} failed, continuing without it:\n{details}", flush=True)

    def summary(self) -> str:
        with self._lock:
            return ", ".join(f"{name}: {count}" for name, count in self.failures.items())


class InjectedFault(RuntimeError):
    pass


def parse_faults(specs: list[str]) -> dict[str, float]:
    """`["hand", "face:0.3"]` -> {"hand": 1.0, "face": 0.3}."""
    faults = {}
    for spec in specs:
        name, _, probability = spec.partition(":")
        faults[name] = float(probability) if probability else 1.0
    return faults
