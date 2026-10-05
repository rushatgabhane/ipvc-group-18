"""Central configuration. Every tunable lives here so experiments are reproducible.

Defaults are chosen from measurements on an M1 Pro (see docs/PERFORMANCE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass
class CaptureConfig:
    source: str = "0"  # camera index ("0") or path to a video file
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True  # flip horizontally so players see a mirror image
    realtime: bool = True  # video files: pace to their native fps (False = as fast as possible)
    loop: bool = False  # video files: restart at the end


@dataclass
class PerceptionConfig:
    # Measured: lite 25 ms vs full 33 ms per frame. Lite keeps us inside a 33 ms (30 fps) budget.
    pose_model: str = "pose_landmarker_lite"
    face_model: str = "face_landmarker"
    hand_model: str = "hand_landmarker"
    max_players: int = 2
    # Segmentation mask comes from the pose model for ~0.3 ms extra, so T5 gets it nearly free.
    segmentation: bool = True
    run_face: bool = True
    # Hands (for the fist trigger) cost ~15 ms per hand, so they run on their own thread and the
    # pipeline uses the newest finished result instead of waiting (see vision/perception.py).
    run_hands: bool = True
    max_hands: int = 2  # one trigger hand per player; each extra hand costs another ~15 ms
    hands_async: bool = True  # False = run inline (deterministic, for clip analysis)
    # Pose and face run on separate threads: measured 36.7 ms sequential -> 25.7 ms parallel.
    parallel: bool = True
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    @property
    def pose_model_path(self) -> Path:
        return MODELS_DIR / f"{self.pose_model}.task"

    @property
    def face_model_path(self) -> Path:
        return MODELS_DIR / f"{self.face_model}.task"

    @property
    def hand_model_path(self) -> Path:
        return MODELS_DIR / f"{self.hand_model}.task"


@dataclass
class FilterConfig:
    """One Euro filter parameters (Casiez et al., CHI 2012), in pixel units at capture resolution."""

    min_cutoff: float = 1.0  # Hz; lower = smoother when still, more lag
    beta: float = 0.05  # speed coefficient; higher = less lag during fast motion
    d_cutoff: float = 1.0  # Hz; cutoff for the derivative estimate
    min_visibility: float = 0.5  # keypoints below this are treated as missing


@dataclass
class TrackingConfig:
    max_match_distance: float = 0.25  # fraction of frame width a player may move between frames
    lost_timeout_s: float = 1.5  # keep a lost player's identity this long before releasing it


@dataclass
class GameConfig:
    """T4 rules and action thresholds. Distances are in torso lengths (scale-invariant)."""

    # Fire trigger: "fist" = close the aiming hand (needs hand tracking), "flick" = recoil flick.
    trigger: str = "fist"
    fist_closed: float = 1.25  # hand openness below this = closed ...
    fist_open: float = 1.55  # ... and above this = open again (hysteresis; must reopen between shots)
    aim_lookback_s: float = 0.1  # shoot along the aim from this long before the trigger
    # Flick trigger: a quick upward "recoil" flick of the aiming arm.
    flick_rise_deg: float = 20.0  # elevation gain that counts as a flick ...
    flick_window_s: float = 0.15  # ... within this time window
    armed_after_s: float = 0.3  # arm must aim steadily this long first (raising the arm is not a shot)
    fire_cooldown_s: float = 0.35
    # Duck: shoulders drop relative to the player's own standing height. Hysteresis avoids flicker.
    duck_enter: float = 0.35
    duck_exit: float = 0.20
    # Reload: lower both arms and hold.
    reload_hold_s: float = 0.7
    max_ammo: int = 6
    max_hp: int = 100
    body_damage: int = 15
    head_damage: int = 35
    # Bullets take time to arrive so the target can react (duck) after seeing the shot.
    bullet_travel_s: float = 0.25
    cover_top: float = 0.45  # crate top, torso lengths below the standing shoulder line
    round_end_s: float = 4.0
    practice_targets: int = 3


@dataclass
class DisplayConfig:
    enabled: bool = True
    backend: str = "gl"  # "gl" = pyglet/OpenGL (~2.3 ms/frame) or "cv" (~16 ms/frame on macOS)
    vsync: bool = False  # vsync removes tearing but blocks up to one refresh (~16 ms at 60 Hz)
    window_name: str = "HIGH NOON"
    show_debug: bool = True
    background: bool = True  # replace the real background using the segmentation mask


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    game: GameConfig = field(default_factory=GameConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    metrics_csv: Path | None = None  # per-frame timings for the report
    max_frames: int | None = None  # stop after N displayed frames (benchmarks)
