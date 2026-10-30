from __future__ import annotations

import time

from highnoon.config import Config
from highnoon.core.capture import open_source
from highnoon.core.match_log import MatchLog
from highnoon.core.profiler import Profiler
from highnoon.core.recorder import SessionRecorder
from highnoon.core.signal_log import SignalLog
from highnoon.game.game import Game
from highnoon.render.display import Display
from highnoon.render.renderer import Renderer
from highnoon.render.sound import SoundBoard
from highnoon.tracking.identity import IdentityTracker
from highnoon.vision.motion import MotionFilter
from highnoon.vision.perception import Perceiver, PerceptionWorker

MAX_EXTRAPOLATE_S = 0.1
PERIOD_FOLLOW = 0.1
LATE_FRAME = 1.5
KEY_HELP = "q/esc quit | r restart round | h simulate a hit | d debug overlay | b background | f fullscreen"


def run(cfg: Config) -> Profiler:
    profiler = Profiler(csv_path=cfg.metrics_csv)
    source = open_source(cfg.capture)
    worker = PerceptionWorker(source, Perceiver(cfg.perception))
    tracker = IdentityTracker(cfg.tracking, cfg.perception.max_players)
    motion = MotionFilter(cfg.filters)
    renderer = Renderer(cfg.display)
    game = Game(cfg.game, (cfg.capture.width, cfg.capture.height))
    display = open_display(cfg)
    sounds = SoundBoard() if display is not None and cfg.display.sound else None
    signal_log = SignalLog(cfg.signals_csv) if cfg.signals_csv else None
    match_log = MatchLog(cfg.games_csv) if cfg.games_csv else None
    recorder = (
        SessionRecorder(cfg.record_video, cfg.capture.fps, cfg.capture.mirror) if cfg.record_video else None
    )

    source.start()
    worker.start()
    tick_s = (
        1.0 / cfg.display.render_hz
        if display is not None and cfg.capture.realtime and cfg.display.render_hz
        else None
    )
    last_frame_id = 0
    frames_shown = 0
    perception = players = None
    t_game = arrived = 0.0
    frame_period = 1.0 / cfg.capture.fps
    tweens = 0
    next_tick = time.perf_counter()
    try:
        while True:
            per_frame = max(1, round(frame_period / tick_s)) - 1 if tick_s else 0
            due = next_tick if tweens < per_frame else arrived + LATE_FRAME * frame_period * (tweens + 1)
            wait = 1.0 if tick_s is None or players is None else max(0.0, due - time.perf_counter())
            fresh = worker.get(last_frame_id, timeout=wait)
            if fresh is None and worker.finished:
                break
            if fresh is None and (tick_s is None or players is None):
                continue
            if fresh is not None:
                perception = fresh
                tweens = 0
                frame = perception.frame
                record_pipeline_timings(profiler, perception, last_frame_id)
                last_frame_id = frame.id
                now = time.perf_counter()
                if arrived:
                    frame_period += PERIOD_FOLLOW * (min(now - arrived, MAX_EXTRAPOLATE_S) - frame_period)
                arrived = now
                t_game = max(t_game, frame.t_capture)
                with profiler.stage("tracking"):
                    players = tracker.update(perception)
                with profiler.stage("filter"):
                    motion.update(players, frame.t_capture, tracker.reassigned)
                with profiler.stage("game"):
                    game.width, game.height = frame.size
                    game.update(players, t_game, tracker.reassigned)
                if recorder is not None:
                    recorder.add(frame.image)
                if signal_log is not None:
                    signal_log.record(frame.id, t_game, players, game, len(perception.hands))
                with profiler.stage("camera"):
                    renderer.set_camera(perception, players, game)
            else:
                tweens += 1
                since_frame = min(time.perf_counter() - arrived, MAX_EXTRAPOLATE_S)
                t_game = max(t_game, perception.frame.t_capture + since_frame)
                game.update(players, t_game, set())
            if sounds is not None:
                sounds.update(game, t_game)
            with profiler.stage("render" if fresh is not None else "tween"):
                output = renderer.draw(perception, players, game, profiler, t_game)

            if display is not None:
                with profiler.stage("display" if fresh is not None else "tween_display"):
                    keys = display.show(output, renderer.shake)
                if handle_keys(keys, cfg, game, display):
                    break
            if tick_s is not None:
                between = frame_period / max(1, round(frame_period / tick_s))
                next_tick = arrived + between if fresh is not None else next_tick + between
            if fresh is None:
                continue

            profiler.end_frame(frame.id, frame.t_capture)
            if match_log is not None:
                match_log.update(game, players, t_game, profiler.last_frame)
            frames_shown += 1
            if cfg.max_frames is not None and frames_shown >= cfg.max_frames:
                break
    finally:
        worker.stop()
        source.stop()
        profiler.close()
        if signal_log is not None:
            signal_log.close()
        if match_log is not None:
            match_log.close()
            print(f"{match_log.rounds} games appended to {match_log.path}")
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
    if "h" in keys:
        for state in game.states.values():
            state.hp = max(1, state.hp - game.cfg.max_hp // 4)
            state.last_hit_t = game.now
    if "f" in keys:
        display.toggle_fullscreen()
    return False
