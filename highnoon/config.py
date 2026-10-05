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
    max_players: int = 2
    # Segmentation mask comes from the pose model for ~0.3 ms extra, so T5 gets it nearly free.
    segmentation: bool = True
    run_face: bool = True
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
class DisplayConfig:
    enabled: bool = True
    backend: str = "pygame"  # "pygame" (~2.9 ms/frame) or "cv" (~16 ms/frame on macOS)
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
    display: DisplayConfig = field(default_factory=DisplayConfig)
    metrics_csv: Path | None = None  # per-frame timings for the report
    max_frames: int | None = None  # stop after N displayed frames (benchmarks)
