from __future__ import annotations

import threading
import traceback


class ModuleGuard:
    """Runs a module's work so an exception costs that module one call, not the whole application.

    Modules run every frame, so each distinct error is printed once with its traceback.
    """

    def __init__(self):
        self.failures: dict[str, int] = {}
        self._seen: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()

    def call(self, name: str, fn, *args, fallback=None):
        try:
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
