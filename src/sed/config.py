"""Configuration schema and project layout.

Loads `configs/default.toml`, merges in `projects/<id>/project.toml`
and any `configs/projects/<id>.toml` overlay, returns a typed `Config` object.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
PROJECTS_DIR = REPO_ROOT / "projects"
LIBRARY_DIR = REPO_ROOT / "data_library"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class AudioCfg(BaseModel):
    input_sample_rate: int = 48000
    sample_rate: int = 32000
    channels: int = 1
    record_channels: int = 1
    record_device: str = "default"
    blocksize: int = 1024
    input_gain_db: float = 0.0
    inference_channel: int = 0


class FeaturesCfg(BaseModel):
    n_fft: int = 1024
    win_length: int = 800
    hop_length: int = 320
    n_mels: int = 128
    fmin: int = 0
    fmax: int = 16000
    window_seconds: float = 1.0


class OnsetCfg(BaseModel):
    algo: str = "superflux"
    hop_ms: int = 32
    median_window_s: float = 5.0
    mad_k: float = 4.0
    preroll_ms: int = 300
    postroll_ms: int = 700
    refractory_ms: int = 150
    min_flux: float = 0.05
    clip_seconds: float = 1.0


class ModelCfg(BaseModel):
    backbone: str = "mn10_as"
    pretrained: bool = True
    classes: list[str] = Field(default_factory=lambda: ["_negative"])


class TrainingCfg(BaseModel):
    batch_size: int = 64
    epochs: int = 60
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 10
    seed: int = 42
    mixup_alpha: float = 0.3
    balanced_sampler: bool = False
    loss: str = "asymmetric_focal"
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 2.0
    negative_weight: float = 1.5
    num_workers: int = 4
    precision: str = "16-mixed"


class AugmentationCfg(BaseModel):
    # Each augmentation can be disabled by setting its probability/amount to 0.
    # See configs/default.toml [augmentation] for tuning guidance.
    domain_mix_prob: float = 0.95
    snr_db_range: tuple[float, float] = (-2.0, 25.0)
    per_class_min_snr_db: dict[str, float] = Field(default_factory=dict)
    ambience_pool: str = "data_library/ambience"
    external_negative_mix_prob: float = 0.20
    external_negative_pool: str = "data_library/negatives_external"
    masking_check: bool = True
    masking_min_db: float = 3.0
    time_shift_ms: int = 200                      # 0 disables shift
    pitch_shift_prob: float = 0.4                 # 0 disables pitch shift
    pitch_semitones: tuple[float, float] = (-2.0, 2.0)
    time_stretch_prob: float = 0.2                # 0 disables time stretch
    time_stretch: tuple[float, float] = (0.95, 1.05)
    gaussian_noise_amp: tuple[float, float] = (0.001, 0.01)   # (0,0) disables
    specaug_freq_masks: int = 2                   # 0 disables freq masks
    specaug_freq_width: int = 24
    specaug_time_masks: int = 2                   # 0 disables time masks
    specaug_time_width: int = 25


class InferenceCfg(BaseModel):
    hop_ms: int = 100
    multi_mic_mode: str = "max"               # "max" | "single" | "mean"
    smoothing_median_window: int = 3
    smoothing_ema_alpha: float = 0.4
    default_threshold: float = 0.6
    hysteresis_low: float = 0.3
    consecutive_frames: int = 1
    trigger_refractory_ms: int = 300
    margin_to_second_best: float = 0.08
    negative_max: float = 0.5
    inference_threads: int = 2


class ArtnetCfg(BaseModel):
    enabled: bool = True
    target_ip: str = "255.255.255.255"
    universe: int = 0
    fps: int = 30


class ZmqCfg(BaseModel):
    publish_endpoint: str = "ipc:///tmp/sed-events.sock"


class LoggingCfg(BaseModel):
    level: str = "INFO"
    log_detections: bool = True


class TargetCfg(BaseModel):
    """One target class → ArtNet output mapping.

    For each trigger event on a target class, the lighting controller will
    set DMX channel `channel` on the configured universe to `value` (0-255)
    and run the `effect` envelope (fade-in / hold / fade-out timings).

    Example: 3 glass classes share channel 1 by using distinct values:
      clink_wine_thin   → channel=1, value=85
      clink_wine_thick  → channel=1, value=170
      clink_champagne   → channel=1, value=255
    The downstream lighting controller maps value → cue/scene.
    """
    enabled: bool = True
    channel: int = 1                       # 1-indexed DMX channel
    value: int = 255                       # DMX value 0-255 sent on trigger
    effect: str = "default"


class ProjectMeta(BaseModel):
    id: str = "default"
    name: str = "Unnamed Project"
    created_at: str = ""
    notes: str = ""


class Config(BaseModel):
    project: ProjectMeta = Field(default_factory=ProjectMeta)
    audio: AudioCfg = Field(default_factory=AudioCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    onset: OnsetCfg = Field(default_factory=OnsetCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    augmentation: AugmentationCfg = Field(default_factory=AugmentationCfg)
    inference: InferenceCfg = Field(default_factory=InferenceCfg)
    artnet: ArtnetCfg = Field(default_factory=ArtnetCfg)
    zmq: ZmqCfg = Field(default_factory=ZmqCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    targets: dict[str, TargetCfg] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base. Lists are replaced, dicts merged."""
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(project_id: str | None = None) -> Config:
    """Load default config + optional project overlay."""
    raw = _read_toml(DEFAULT_CONFIG)
    if project_id:
        # 1. configs/projects/<id>.toml (committed-to-repo overlay)
        repo_overlay = REPO_ROOT / "configs" / "projects" / f"{project_id}.toml"
        raw = _deep_merge(raw, _read_toml(repo_overlay))
        # 2. projects/<id>/project.toml (live, per-deployment overlay)
        live_overlay = PROJECTS_DIR / project_id / "project.toml"
        raw = _deep_merge(raw, _read_toml(live_overlay))
    return Config.model_validate(raw)


# ---------------------------------------------------------------------------
# Project layout helpers
# ---------------------------------------------------------------------------


class ProjectLayout:
    """Filesystem layout for a single project / venue."""

    def __init__(self, project_id: str):
        self.id = project_id
        self.root = PROJECTS_DIR / project_id

    @property
    def project_toml(self) -> Path:
        return self.root / "project.toml"

    @property
    def recordings_dir(self) -> Path:
        return self.root / "recordings"

    def positives_dir(self, class_name: str) -> Path:
        return self.recordings_dir / "positives" / class_name

    def hardneg_dir(self, class_name: str) -> Path:
        return self.recordings_dir / "hardneg" / class_name

    @property
    def ambience_dir(self) -> Path:
        return self.recordings_dir / "ambience"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def create(self) -> None:
        for sub in [
            self.recordings_dir,
            self.recordings_dir / "positives",
            self.recordings_dir / "hardneg",
            self.ambience_dir,
            self.dataset_dir,
            self.models_dir,
            self.reports_dir,
        ]:
            sub.mkdir(parents=True, exist_ok=True)


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Config snapshots — for reproducibility
# ---------------------------------------------------------------------------
#
# A snapshot is the FULL effective config (default + project overlay merged)
# serialized to JSON next to the trained model. At training time we write
# `projects/<id>/models/training_config.json`. At inference time we *prefer*
# loading from the snapshot over the live config, falling back to live when
# no snapshot exists (e.g. before the first training).
#
# Why: today the live `default.toml` is consulted at every command. If you
# edit a default after training, the running inference picks up the new
# value silently. The snapshot is the contract — it captures exactly what
# the model was trained against so behaviour is stable across edits.
#
# We use JSON not TOML for the snapshot because (a) it's already in the
# stdlib for both reading and writing, (b) Pydantic dumps cleanly to JSON,
# and (c) snapshots are machine-generated artefacts not meant to be hand-
# edited the way project.toml is.


def dump_config_snapshot(cfg: Config, path: Path) -> Path:
    """Write the effective config to `path` as JSON. Atomic via temp+rename."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2)
    tmp.replace(path)
    return path


def load_config_snapshot(path: Path) -> Config:
    """Read a config snapshot back into a Config object."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return Config.model_validate(json.load(f))


def load_config_for_inference(project_id: str) -> Config:
    """Like `load_config(project_id)` but prefers the training_config.json
    snapshot if present. This is what the runtime should call so that
    edits to default.toml don't silently change inference behaviour after
    a model was trained.

    Falls back to live config when no snapshot exists (e.g. before first
    training, or after deleting models/).
    """
    layout = ProjectLayout(project_id)
    snapshot = layout.models_dir / "training_config.json"
    if snapshot.exists():
        cfg = load_config_snapshot(snapshot)
        # Allow venue-specific overrides that should NOT be frozen at
        # training time to be re-applied from project.toml. These are
        # operationally tunable (no model-shape implications).
        live = _read_toml(layout.root / "project.toml")
        for tunable_block in ("artnet", "targets", "thresholds", "logging"):
            if tunable_block in live:
                merged = _deep_merge(
                    cfg.model_dump(mode="json"), {tunable_block: live[tunable_block]}
                )
                cfg = Config.model_validate(merged)
        if "audio" in live:
            audio = {
                k: v
                for k, v in live["audio"].items()
                if k in {
                    "input_sample_rate",
                    "record_channels",
                    "record_device",
                    "blocksize",
                    "input_gain_db",
                    "inference_channel",
                }
            }
            if audio:
                merged = _deep_merge(cfg.model_dump(mode="json"), {"audio": audio})
                cfg = Config.model_validate(merged)
        return cfg
    return load_config(project_id)
