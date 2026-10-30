from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass
class CaptureConfig:
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    realtime: bool = True
    loop: bool = False


@dataclass
class PerceptionConfig:
    max_players: int = 2
    max_hands: int = 4
    delegate: str = "auto"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    pose_model_path: Path = MODELS_DIR / "pose_landmarker_lite.task"
    face_model_path: Path = MODELS_DIR / "face_landmarker.task"
    hand_model_path: Path = MODELS_DIR / "hand_landmarker.task"


@dataclass
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0
    min_visibility: float = 0.5


@dataclass
class TrackingConfig:
    max_match_distance: float = 0.25
    lost_timeout_s: float = 1.5
    reid_max_distance: float = 0.5
    reid_margin: float = 0.05
    reid_confirm_s: float = 0.15
    reid_blend: float = 0.01
    enrol_max_samples: int = 90


@dataclass
class GameConfig:
    fist_closed: float = 1.25
    fist_open: float = 1.45
    aim_lookback_s: float = 0.1
    armed_after_s: float = 0.3
    fire_cooldown_s: float = 0.6
    recoil_per_shot_deg: float = 10.0
    recoil_recovery_s: float = 1.2
    recoil_max_deg: float = 20.0
    cover_height: float = 0.25
    duck_margin: float = 0.015
    max_ammo: int = 3
    reload_s: float = 0.8
    camp_grace_s: float = 7.0
    camp_drain_fraction: float = 0.05
    camp_drain_interval_s: float = 1.0
    camp_warning_s: float = 5.0
    max_hp: int = 100
    body_damage: int = 10
    head_damage: int = 25
    bullet_travel_s: float = 0.6
    hand_reach: float = 0.06
    gun_reach: float = 0.27
    rocket_reach: float = 0.38
    grenade_reach: float = 0.1
    round_end_s: float = 5.0
    countdown_s: float = 5.0
    come_back_after_s: float = 1.0
    leave_timeout_s: float = 5.0
    powerup_interval_s: float = 15.0
    powerup_fall_s: float = 5.0
    powerup_radius: float = 0.08
    powerup_hit_scale: float = 1.3
    powerup_center_spread: float = 0.05
    rocket_speed: float = 1.6
    rocket_gravity: float = 1.3
    rocket_blast_radius: float = 0.16
    rocket_damage: int = 50
    rocket_min_damage: int = 20
    grenade_flight_s: float = 1.4
    grenade_arc: float = 0.45
    practice: bool = False
    practice_powerup_interval_s: float = 10.0
    bomb_damage: int = 25
    bottle_count: int = 4
    bottle_radius: float = 0.06
    bottle_respawn_s: float = 0.6


@dataclass
class DisplayConfig:
    enabled: bool = True
    vsync: bool = False
    window_name: str = "HIGH NOON"
    show_debug: bool = False
    sound: bool = True
    render_hz: int = 60
    background: bool = True
    grade: bool = True


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    game: GameConfig = field(default_factory=GameConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    metrics_csv: Path | None = None
    signals_csv: Path | None = None
    games_csv: Path | None = Path("results/games.csv")
    record_video: Path | None = None
    max_frames: int | None = None
