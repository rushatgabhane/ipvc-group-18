"""Main loop.

Threads:
  capture     camera -> newest frame (core/capture.py)
  perception  newest frame -> pose + face inference (vision/perception.py)
  main        identity -> filtering -> game -> render -> display (this file)

The main thread always processes the newest perception result. If it is slower than
inference, frames are dropped rather than queued, which keeps latency bounded.
Window/event handling (Cocoa via pyglet or HighGUI) must run on the main thread on macOS.
"""

from __future__ import annotations

import time

from highnoon.config import Config
from highnoon.core.capture import open_source
from highnoon.core.profiler import Profiler
from highnoon.game.game import Game
from highnoon.render.display import open_display
from highnoon.render.renderer import Renderer
from highnoon.tracking.identity import IdentityTracker
from highnoon.vision.motion import MotionFilter
from highnoon.vision.perception import Perceiver, PerceptionWorker

KEY_HELP = "q/esc quit | r restart round | d debug overlay | b background | f fullscreen"


def run(cfg: Config) -> Profiler:
    profiler = Profiler(csv_path=cfg.metrics_csv)
    source = open_source(cfg.capture)
    worker = PerceptionWorker(source, Perceiver(cfg.perception))
    tracker = IdentityTracker(cfg.tracking, cfg.perception.max_players)
    motion = MotionFilter(cfg.filters)
    renderer = Renderer(cfg.display)
    game = Game(cfg.game, (cfg.capture.width, cfg.capture.height))

    display = None
    if cfg.display.enabled:
        size = (cfg.capture.width, cfg.capture.height)
        display = open_display(cfg.display.backend, cfg.display.window_name, size, cfg.display.vsync)
        print(KEY_HELP)

    source.start()
    worker.start()
    last_id = 0
    frames = 0
    try:
        while True:
            perception = worker.get(last_id, timeout=1.0)
            if perception is None:
                if worker.finished:
                    break
                continue
            frame = perception.frame
            # Frames skipped because the pipeline was busy; >0 means we are below camera rate.
            profiler.add({"dropped": float(max(0, frame.id - last_id - 1)) if last_id else 0.0})
            last_id = frame.id
            profiler.add(perception.timings)
            profiler.add({"wait": (time.perf_counter() - perception.t_done) * 1000})

            with profiler.stage("tracking"):
                players = tracker.update(perception)
            with profiler.stage("filter"):
                motion.update(players, frame.t_capture, tracker.reassigned)
            with profiler.stage("game"):
                game.width, game.height = frame.size
                game.update(players, frame.t_capture, tracker.reassigned)
            with profiler.stage("render"):
                output = renderer.render(perception, players, game, profiler)

            if display is not None:
                with profiler.stage("display"):
                    keys = display.show(output)
                if "q" in keys or "esc" in keys:
                    break
                if "d" in keys:
                    cfg.display.show_debug = not cfg.display.show_debug
                if "b" in keys:
                    cfg.display.background = not cfg.display.background
                if "r" in keys:
                    game.reset_round()
                if "f" in keys:
                    display.toggle_fullscreen()

            profiler.end_frame(frame.id, frame.t_capture)
            frames += 1
            if cfg.max_frames is not None and frames >= cfg.max_frames:
                break
    finally:
        worker.stop()
        source.stop()
        profiler.close()
        if display is not None:
            display.close()
    return profiler
