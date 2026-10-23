from __future__ import annotations

import time

from highnoon.config import Config
from highnoon.core.capture import open_source
from highnoon.core.profiler import Profiler
from highnoon.core.recorder import SessionRecorder
from highnoon.core.signal_log import SignalLog
from highnoon.game.game import Game
from highnoon.render.display import Display
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
    display = open_display(cfg)
    signal_log = SignalLog(cfg.signals_csv) if cfg.signals_csv else None
    recorder = (
        SessionRecorder(cfg.record_video, cfg.capture.fps, cfg.capture.mirror) if cfg.record_video else None
    )

    source.start()
    worker.start()
    last_frame_id = 0
    frames_shown = 0
    try:
        while True:
            perception = worker.get(last_frame_id, timeout=1.0)
            if perception is None:
                if worker.finished:
                    break
                continue
            frame = perception.frame
            record_pipeline_timings(profiler, perception, last_frame_id)
            last_frame_id = frame.id

            with profiler.stage("tracking"):
                players = tracker.update(perception)
            with profiler.stage("filter"):
                motion.update(players, frame.t_capture, tracker.reassigned)
            with profiler.stage("game"):
                game.width, game.height = frame.size
                game.update(players, frame.t_capture, tracker.reassigned)
            if recorder is not None:
                recorder.add(frame.image)
            if signal_log is not None:
                signal_log.record(frame.id, frame.t_capture, players, game, len(perception.hands))
            with profiler.stage("render"):
                output = renderer.render(perception, players, game, profiler)

            if display is not None:
                with profiler.stage("display"):
                    keys = display.show(output, renderer.shake)
                if handle_keys(keys, cfg, game, display):
                    break

            profiler.end_frame(frame.id, frame.t_capture)
            frames_shown += 1
            if cfg.max_frames is not None and frames_shown >= cfg.max_frames:
                break
    finally:
        worker.stop()
        source.stop()
        profiler.close()
        if signal_log is not None:
            signal_log.close()
        if recorder is not None:
            recorder.close()
            print(f"recorded {recorder.path} ({recorder.dropped} frames dropped)")
        if display is not None:
            display.close()
    return profiler


def open_display(cfg: Config) -> Display | None:
    if not cfg.display.enabled:
        return None
    print(KEY_HELP)
    size = (cfg.capture.width, cfg.capture.height)
    return Display(cfg.display.window_name, size, cfg.display.vsync)


def record_pipeline_timings(profiler: Profiler, perception, last_frame_id: int) -> None:
    skipped = perception.frame.id - last_frame_id - 1 if last_frame_id else 0
    profiler.add({"dropped": float(max(0, skipped))})
    profiler.add(perception.timings)
    profiler.add({"wait": (time.perf_counter() - perception.t_done) * 1000})


def handle_keys(keys: list[str], cfg: Config, game: Game, display: Display) -> bool:
    if "q" in keys or "esc" in keys:
        return True
    if "d" in keys:
        cfg.display.show_debug = not cfg.display.show_debug
    if "b" in keys:
        cfg.display.background = not cfg.display.background
    if "r" in keys:
        game.reset_round()
    if "f" in keys:
        display.toggle_fullscreen()
    return False
