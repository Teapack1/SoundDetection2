"""Configuration schema and project layout.

Each project has its own user-edited input config at
`projects/<id>/config.toml`. Project directories also hold generated state:
recordings, dataset snapshots, and trained model bundles under
`models/<model_id>/` (each with its own training_config.json + runtime_settings.json).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

import math

from pydantic import BaseModel, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"
LIBRARY_DIR = REPO_ROOT / "data_library"
PROJECT_CONFIG_NAME = "config.toml"
_TOML_NULL_SENTINEL = "__SED_TOML_NULL__"


EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT: dict[str, Any] = {
    "features.frontend": "efficientat",
    "model.sample_rate": 32000,
    "model.channels": 1,
    "features.n_fft": 1024,
    "features.win_length": 800,
    "features.hop_length": 320,
    "features.n_mels": 128,
    "features.fmin": 0,
    "features.fmax": 15000,
    "features.window_seconds": 1.0,
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class AudioCfg(BaseModel):
    """Sound-card / capture settings (PART 2 — machine-local, read live).

    These describe the physical audio interface on this machine, not the model.
    Raw recorded WAVs use this sample rate. Training and live inference resample
    complete clips/windows to the model's feature rate (model.sample_rate) at
    the model boundary. The model's own shape (feature sample_rate, input
    channels) lives in [model] and is frozen into each trained bundle.
    """
    record_sample_rate: int = 48000  # PortAudio capture rate of this card.
    # Ordered list of physical input-channel indexes to capture, e.g. [0, 3, 1]
    # records three channels and stores them in that order (recorded ch0 = device
    # ch0, recorded ch1 = device ch3, recorded ch2 = device ch1). Default [0] =
    # only the first input channel. Same format/semantics as
    # inference.inference_channels. A bare integer is accepted for back-compat and
    # read as a channel COUNT (N -> [0..N-1]).
    record_channels: list[int] = Field(default_factory=lambda: [0])
    blocksize: int = 1024            # PortAudio callback chunk.
    # Free-form description of the physical microphone(s) in use. Stamped onto
    # every recorded clip's sidecar JSON so the dataset later carries traceable
    # provenance ("which mic captured this?"). Default ["generic"]; override per
    # venue/project (e.g. ["Shure SM58"], or list for multi-mic e.g. ["generic", "rode"]).
    microphone_type: list[str] = Field(default_factory=lambda: ["generic"])

    @field_validator("microphone_type", mode="before")
    @classmethod
    def _coerce_microphone_type(cls, value: Any) -> Any:
        if value is None:
            return ["generic"]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return ["generic"]
            if "," in stripped or ";" in stripped:
                parts = [p.strip() for p in stripped.replace(";", ",").split(",") if p.strip()]
                return parts if parts else ["generic"]
            return [stripped]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if value else ["generic"]

    @field_validator("record_channels", mode="before")
    @classmethod
    def _coerce_record_channels(cls, value: Any) -> Any:
        # Back-compat: a bare integer historically meant a channel COUNT, so
        # N -> [0..N-1]. A list (or comma string) is an ordered index selection.
        if isinstance(value, bool):
            raise ValueError("audio.record_channels must be a list of channel indexes")
        if value is None:
            return [0]
        if isinstance(value, int):
            return list(range(value)) if value >= 1 else [0]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return [0]
            parts = [p.strip() for p in stripped.replace(";", ",").split(",") if p.strip()]
            return [int(p) for p in parts] if parts else [0]
        return value

    @model_validator(mode="after")
    def _check_audio(self) -> "AudioCfg":
        if self.record_sample_rate <= 0:
            raise ValueError("audio.record_sample_rate must be positive")
        if not self.record_channels:
            raise ValueError("audio.record_channels must list at least one channel")
        if any(int(c) < 0 for c in self.record_channels):
            raise ValueError("audio.record_channels must be zero-based channel indexes")
        channels = [int(c) for c in self.record_channels]
        if len(set(channels)) != len(channels):
            raise ValueError("audio.record_channels must not contain duplicates")
        if max(channels) > 63:
            raise ValueError("audio.record_channels indexes must be <= 63")
        self.record_channels = channels
        if self.blocksize <= 0:
            raise ValueError("audio.blocksize must be positive")
        return self


class FeaturesCfg(BaseModel):
    # Feature frontend — fixed to the pretrained EfficientAT preprocessing
    # contract. The old legacy_torchaudio_db frontend was removed; bundles
    # trained with it need an older repo revision or a retrain.
    frontend: str = "efficientat"
    n_fft: int = 1024
    win_length: int = 800
    hop_length: int = 320
    n_mels: int = 128
    fmin: int = 0
    fmax: int = 15000  # match EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT (EfficientAT 32 kHz default)
    window_seconds: float = 1.0

    @model_validator(mode="after")
    def _check_features(self) -> "FeaturesCfg":
        self.frontend = str(self.frontend or "efficientat").lower()
        if self.frontend != "efficientat":
            raise ValueError(
                "features.frontend must be 'efficientat' — the legacy_torchaudio_db "
                "frontend was removed (retrain the model or pin an older repo revision)"
            )
        if self.n_fft <= 0 or self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("features n_fft, win_length, and hop_length must be positive")
        if self.n_mels <= 0 or self.window_seconds <= 0:
            raise ValueError("features n_mels and window_seconds must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("features.win_length must be <= n_fft")
        if self.fmin < 0 or self.fmax <= self.fmin:
            raise ValueError("features must satisfy 0 <= fmin < fmax")
        return self


class OnsetCfg(BaseModel):
    algo: str = "superflux"
    hop_ms: int = 32
    median_window_s: float = 5.0
    mad_k: float = 3.0
    preroll_ms: int = 300
    # Derived from record_seconds and preroll_ms; recording code uses this value.
    postroll_ms: int = 700
    # Global delay between accepted recordings. Unlike the detector's small
    # internal anti-double-trigger refractory, this applies across all selected
    # mics: once any mic starts a recording group, no mic can start another group
    # until this cooldown expires.
    recording_cooldown_ms: int = 1500
    # Operators tune min_flux_db; code keeps the matching linear value for the
    # onset detector.
    min_flux: float = 1e-4
    min_flux_db: float = -80.0
    record_seconds: float = 1.0
    # Per-clip preroll randomisation. When True, each onset-triggered clip gets
    # a preroll drawn uniformly from `preroll_ms_range`; the postroll auto-
    # adjusts so the total saved length stays exactly record_seconds. Replaces
    # the post-processing time-shift augmentation — temporal jitter now lives
    # at recording time so the spectrogram is never wrapped or zero-padded.
    randomize_preroll: bool = True
    preroll_ms_range: tuple[int, int] = (100, 700)
    # Multichannel triggering. With "any" each recorded channel runs its own
    # onset detector and ANY of them can trigger a clip; near-simultaneous hits
    # across channels are merged and only the fired physical channels are saved.
    # "mix" runs a single detector on the channel downmix (cheaper, but a
    # loud-on-one-mic transient can be diluted). "ch0" is the legacy behaviour.
    multi_mic_trigger: str = "any"   # any | mix | ch0
    # Window for collapsing the same physical transient detected on several
    # channels into a single recording. Applies only to multi_mic_trigger="any",
    # where each selected mic runs its own onset detector.
    cross_channel_merge_ms: int = 500

    @model_validator(mode="after")
    def _check_multi_mic_trigger(self) -> "OnsetCfg":
        if self.multi_mic_trigger not in {"any", "mix", "ch0"}:
            raise ValueError(
                f"onset.multi_mic_trigger must be one of any|mix|ch0 "
                f"(got {self.multi_mic_trigger!r})"
            )
        if self.cross_channel_merge_ms < 0:
            raise ValueError("onset.cross_channel_merge_ms must be >= 0")
        if self.recording_cooldown_ms < 0:
            raise ValueError("onset.recording_cooldown_ms must be >= 0")
        return self

    @model_validator(mode="after")
    def _sync_min_flux(self) -> "OnsetCfg":
        fs = self.model_fields_set
        if "min_flux" in fs and "min_flux_db" not in fs:
            # User set linear only → derive dB from it.
            self.min_flux_db = 20.0 * math.log10(self.min_flux + 1e-12)
        elif "min_flux_db" in fs and "min_flux" not in fs:
            # User set dB only → derive linear from it.
            self.min_flux = float(10.0 ** (self.min_flux_db / 20.0))
        elif "min_flux" in fs and "min_flux_db" in fs:
            # User set both → linear wins (back-compat); resync dB to match.
            self.min_flux_db = 20.0 * math.log10(self.min_flux + 1e-12)
        # else: both at code defaults (already consistent).
        return self

    @model_validator(mode="after")
    def _check_randomize_preroll(self) -> "OnsetCfg":
        clip_ms = int(self.record_seconds * 1000)
        if self.preroll_ms >= clip_ms:
            raise ValueError(
                f"preroll_ms ({self.preroll_ms} ms) must be below "
                f"record_seconds ({clip_ms} ms)"
            )
        self.postroll_ms = clip_ms - self.preroll_ms
        if self.randomize_preroll:
            lo, hi = self.preroll_ms_range
            if lo < 0 or hi <= lo:
                raise ValueError(
                    f"preroll_ms_range must satisfy 0 <= lo < hi (got {self.preroll_ms_range})"
                )
            if hi >= clip_ms:
                raise ValueError(
                    f"preroll_ms_range upper bound ({hi} ms) must be strictly less than "
                    f"record_seconds ({self.record_seconds * 1000:.0f} ms) — otherwise postroll "
                    "would be zero or negative."
                )
        return self


class ModelCfg(BaseModel):
    backbone: str = "mn10_as"
    pretrained: bool = True
    classes: list[str] = Field(default_factory=lambda: ["_negative"])
    # Model-shape audio: what the network was trained on. Frozen into each
    # trained bundle — the capture stream is resampled to this feature rate.
    sample_rate: int = 32000      # Feature/model rate (EfficientAT: 32 kHz).
    channels: int = 1             # Model input channels (normally mono).

    @model_validator(mode="after")
    def _check_model(self) -> "ModelCfg":
        if self.sample_rate <= 0:
            raise ValueError("model.sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("model.channels must be positive")
        return self


class DatasetSourceRule(BaseModel):
    """One data-library label selection row for a model output class.

    New configs use only ``labels`` plus optional ``project_ids``/``density``.
    ``label`` and ``category`` remain accepted so older selector rows still
    parse; compose normalizes them to the same data-library query semantics.
    """
    labels: list[str] = Field(default_factory=list)
    label: str | None = None
    project_ids: list[str] | str | None = None
    project_id: list[str] | str | None = None
    density: str | None = None
    category: str | None = None


def _normalize_rule_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows = value if isinstance(value, list) else [value]
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, BaseModel):
            row = row.model_dump(exclude_none=True)
        if isinstance(row, dict):
            out.append(dict(row))
    return out


class DatasetCfg(BaseModel):
    """Dataset composition selections for one project.

    `target_classes` is the editable model class list. Include `_negative`
    when you want an explicit no-target output; remove it to train only on
    target event classes.
    """
    target_classes: list[str] = Field(default_factory=list)
    # Accepted for older configs; generated configs use target_classes.
    classes: list[str] = Field(default_factory=list)
    external_positives_pool: str = "data_library/sounds"
    external_negatives_pool: str = "data_library/sounds"
    sources: dict[str, list[DatasetSourceRule]] = Field(default_factory=dict)
    # Deprecated aliases. They still load, then normalize into `sources`.
    project: dict[str, Any] = Field(default_factory=dict)
    external: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_sources(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw_sources = data.get("sources")
        if isinstance(raw_sources, dict):
            data = dict(data)
            data["sources"] = {
                str(class_name): _normalize_rule_rows(rows)
                for class_name, rows in raw_sources.items()
            }
        return data

    @model_validator(mode="after")
    def _legacy_sources_to_primary(self) -> "DatasetCfg":
        if self.sources:
            return self
        merged: dict[str, list[DatasetSourceRule]] = {}
        for block in (self.project or {}, self.external or {}):
            for class_name, rows in block.items():
                merged.setdefault(str(class_name), []).extend(
                    DatasetSourceRule.model_validate(row)
                    for row in _normalize_rule_rows(rows)
                )
        self.sources = merged
        return self


class TrainingCfg(BaseModel):
    batch_size: int = 64
    epochs: int = 60
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 10
    seed: int = 42
    loss: str = "cross_entropy"
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 2.0
    negative_weight: float = 1.5
    # SpecAugment-style masking on training spectrograms only. The pretrained
    # EfficientAT backbone was trained WITH masking (freqm=48/timem=192 on
    # 10 s clips); fine-tuning without any hurts robustness on small venue
    # datasets. Mask widths are scaled for 1 s / ~100-frame windows.
    specaugment_enabled: bool = True
    specaugment_freq_mask: int = 24   # max masked mel bins per mask (of n_mels)
    specaugment_time_mask: int = 16   # max masked frames per mask (of ~100)
    num_workers: int = 4
    precision: str = "16-mixed"
    auto_export: bool = True
    auto_test: bool = True

    @model_validator(mode="after")
    def _check_training(self) -> "TrainingCfg":
        allowed_losses = {
            "cross_entropy", "ce", "softmax_ce",
            "asymmetric_focal",
            "bce", "bce_logits", "binary_cross_entropy",
        }
        self.loss = self.loss.strip().lower()
        if self.loss not in allowed_losses:
            raise ValueError(
                "training.loss must be one of: cross_entropy, "
                "asymmetric_focal, bce_logits"
            )
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("training batch_size and epochs must be positive")
        if self.lr_backbone <= 0 or self.lr_head <= 0:
            raise ValueError("training learning rates must be positive")
        if self.weight_decay < 0 or self.warmup_epochs < 0 or self.patience < 0:
            raise ValueError("training weight_decay, warmup_epochs and patience must be >= 0")
        if self.negative_weight <= 0:
            raise ValueError("training negative_weight must be positive")
        if self.specaugment_freq_mask < 0 or self.specaugment_time_mask < 0:
            raise ValueError("training specaugment mask widths must be >= 0")
        if self.num_workers < 0:
            raise ValueError("training num_workers must be >= 0")
        return self


class AugmentationCombination(BaseModel):
    base: str
    overlay: str
    base_filter: dict[str, Any] = Field(default_factory=dict)
    overlay_filter: dict[str, Any] = Field(default_factory=dict)
    base_filters: list[dict[str, Any]] = Field(default_factory=list)
    overlay_filters: list[dict[str, Any]] = Field(default_factory=list)
    recipe_id: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    mask_min_mean_db: float | None = None
    mask_min_p25_db: float | None = None
    mask_min_band_db: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "recipe_id" not in data and "id" in data:
            data["recipe_id"] = data["id"]
        if "base_filters" not in data and isinstance(data.get("base_filter"), dict):
            data["base_filters"] = [data["base_filter"]]
        if "overlay_filters" not in data and isinstance(data.get("overlay_filter"), dict):
            data["overlay_filters"] = [data["overlay_filter"]]
        for legacy_key in ("note", "notes"):
            if legacy_key in data and "description" not in data:
                data["description"] = data[legacy_key]
        tags = data.get("tags")
        if isinstance(tags, str):
            data["tags"] = [
                part.strip()
                for part in tags.replace(";", ",").split(",")
                if part.strip()
            ]
        return data


class AugmentationTargetCfg(BaseModel):
    """Deprecated wrapper for old [augmentation.targets.<class>] tables."""
    combinations: list[AugmentationCombination] = Field(default_factory=list)


class AugmentationCfg(BaseModel):
    """Offline overlay augmentation pipeline.

    Runs as a pre-processing step via `sed dataset augment`. Writes per-
    class WAVs + sidecars under data_library/derived/<project>/trainval/ for
    human review before training. Training consumes the outputs directly;
    there is no runtime augmentation.
    """
    enabled: bool = True
    seed: int = 42
    samples_per_source: float = 2
    pitch_shift_enabled: bool = True
    semitones_range: tuple[float, float] = (-2.0, 2.0)
    overlay_enabled: bool = True
    snr_db_range: tuple[float, float] = (0.0, 20.0)
    scale_noise_to_snr: bool = True
    prevent_clipping: bool = True
    # Transient-aware loudness reference: when > 0, target and overlay RMS are
    # measured in the same window starting at the base clip's spectral-flux
    # onset. 0 = use full-clip RMS.
    transient_window_ms: int = 100
    combinations: dict[str, list[AugmentationCombination]] = Field(default_factory=dict)
    # Deprecated alias accepted for compatibility:
    # [augmentation.targets.<class>].combinations -> [augmentation.combinations].<class>
    targets: dict[str, AugmentationTargetCfg] = Field(default_factory=dict)
    # Deprecated compatibility controls used only when `combinations` is empty.
    generate_positive_overlay: bool = True
    generate_negative_overlay: bool = True
    negative_overlay_base_labels: list[str] = Field(default_factory=list)
    mask_filter_enabled: bool = True
    mask_candidate_attempts: int = 8
    mask_min_mean_db: float = 6.0
    mask_min_p25_db: float = 3.0
    mask_min_band_db: float = -3.0
    mask_band_count: int = 10
    mask_top_frames: int = 5
    # Broadband floor: reject candidates whose base transient sits more than
    # this many dB below the overlay in the same window (effective SNR). The
    # band-limited mask gates can stay positive while the overlay swamps the
    # mix overall — this gate catches those. None = off.
    mask_min_effective_snr_db: float | None = None
    # Render a two-color base/overlay mel PNG (<clip>.mix.png) next to each
    # overlay-augmented clip so a buried base transient is visible at a glance.
    mix_map_enabled: bool = True
    per_class_min_snr_db: dict[str, float] = Field(default_factory=dict)
    per_class_negative_label: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_combinations(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        raw = data.get("combinations")
        if isinstance(raw, dict):
            data["combinations"] = {
                str(class_name): _normalize_rule_rows(rows)
                for class_name, rows in raw.items()
            }
        return data

    @model_validator(mode="after")
    def _legacy_targets_to_combinations(self) -> "AugmentationCfg":
        if self.combinations or not self.targets:
            return self
        self.combinations = {
            target_class: list(target_cfg.combinations)
            for target_class, target_cfg in self.targets.items()
        }
        return self


class InferenceCfg(BaseModel):
    hop_ms: int = 300
    multi_mic_mode: str = "max"               # "max" | "best" | "single" | "mean"
    inference_channels: list[int] = Field(default_factory=lambda: [0])
    # Onset-gated inference. When True, the runtime runs the same onset detector
    # the recorder uses, scores ONE onset-centered window per transient, and
    # decides with single-window gates — instead of a blind sliding window that
    # demands `consecutive_frames` of sustained confidence. This matches how the
    # training/frozen-test clips were built (onset-centered), so brief transients
    # (e.g. glass clinks) are no longer missed by the temporal-consensus gate.
    # The onset detector params come from the [onset] block (same as recording).
    onset_gating_enabled: bool = False
    # When False, live inference fires on the raw per-window probability the
    # moment it crosses threshold (no median / EMA / hysteresis / consecutive
    # frames). This makes the live decision identical to the frozen-test
    # evaluation, which is always per-window. The other gates (threshold,
    # negative_max, margin, refractory) still apply.
    smoothing_enabled: bool = True
    smoothing_median_window: int = 3
    smoothing_ema_alpha: float = 0.4
    default_threshold: float = 0.9
    hysteresis_low: float = 0.3
    consecutive_frames: int = 3
    trigger_refractory_ms: int = 300
    margin_to_second_best: float = 0.08
    negative_max: float = 0.5
    inference_threads: int = 2
    min_dbfs: float | None = None

    @field_validator("min_dbfs", mode="before")
    @classmethod
    def _coerce_min_dbfs(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "off", "none", "null"}:
            return None
        return value

    @field_validator("inference_channels", mode="before")
    @classmethod
    def _coerce_inference_channels(cls, value: Any) -> Any:
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return [0]
            if "," in stripped:
                return [int(part.strip()) for part in stripped.split(",") if part.strip()]
            return [int(stripped)]
        return value

    @model_validator(mode="after")
    def _check_inference(self) -> "InferenceCfg":
        if self.hop_ms <= 0:
            raise ValueError("inference.hop_ms must be positive")
        if self.multi_mic_mode not in {"max", "best", "single", "mean"}:
            raise ValueError("inference.multi_mic_mode must be max, best, single, or mean")
        if not self.inference_channels:
            raise ValueError("inference.inference_channels must list at least one channel")
        if any(int(c) < 0 for c in self.inference_channels):
            raise ValueError("inference.inference_channels must be zero-based channel indexes")
        channels = [int(c) for c in self.inference_channels]
        if len(set(channels)) != len(channels):
            raise ValueError("inference.inference_channels must not contain duplicates")
        self.inference_channels = channels
        if self.smoothing_median_window < 1 or self.consecutive_frames < 1:
            raise ValueError("inference smoothing_median_window and consecutive_frames must be >= 1")
        if not 0.0 <= self.smoothing_ema_alpha <= 1.0:
            raise ValueError("inference.smoothing_ema_alpha must be between 0 and 1")
        for name in ("default_threshold", "hysteresis_low", "negative_max"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"inference.{name} must be between 0 and 1")
        if self.margin_to_second_best < 0:
            raise ValueError("inference.margin_to_second_best must be >= 0")
        if self.trigger_refractory_ms < 0 or self.inference_threads < 1:
            raise ValueError("inference trigger_refractory_ms must be >= 0 and threads >= 1")
        return self


class MiningCfg(BaseModel):
    """Runtime detection mining.

    When enabled, live inference writes detected windows to
    projects/<id>/recordings/mined/<class>/ for later review. The folder is not
    consumed by dataset compose automatically; humans should inspect mined clips
    and promote useful false positives into negatives/hard-negative training
    sources deliberately.
    """
    enabled: bool = False
    output_dir: str = "mined"
    per_class_min_confidence: dict[str, float] = Field(default_factory=dict)
    # Per-class cooldown between mined "threshold" clips. Without it the miner
    # saves a clip on every hop a class stays above threshold (e.g. ~1.4/s at
    # hop_ms=700), flooding the mined/ folder during one sound. Trigger-kind
    # clips are already refractory-gated by the arbiter; this gates the rest.
    min_interval_ms: int = 1500


class DeployCfg(BaseModel):
    """Model selection + run-time operator controls (PART 2 — set on the edge, read live).

    Read LIVE from the edge config.toml at runtime — never baked into a model
    bundle. The sound card's capture settings live in [audio] (also PART 2);
    this block picks which model runs, which device to open, and how chatty
    `sed inference` should be.

    `active_model` names the subfolder under projects/<id>/models/ to run, e.g.
    a model id like "2026-05-21_2140-wide-snr". Empty = auto: use the only
    trained model present.
    """
    active_model: str = ""
    record_device: str = "default"   # device index or unique name substring
    verbose: str = "none"             # "none" | "info" | "live"

    @model_validator(mode="after")
    def _check_deploy(self) -> "DeployCfg":
        self.verbose = str(self.verbose or "none").lower()
        if self.verbose not in {"none", "info", "live"}:
            raise ValueError("deploy.verbose must be one of: none, info, live")
        return self


class ArtnetCfg(BaseModel):
    # Off and unconfigured by default so install and all test runs work with
    # nothing sent on the network. ArtNet is enabled as the final on-site step,
    # once target_ip is set to the receiver's specific unicast IP. Broadcasting
    # is never a default; the controller rejects broadcast targets at runtime.
    enabled: bool = False
    target_ip: str = ""
    universe: int = 0
    fps: int = 30


class ZmqCfg(BaseModel):
    publish_endpoint: str = "ipc:///tmp/sed-events.sock"


class LoggingCfg(BaseModel):
    level: str = "INFO"
    log_detections: bool = True


class InspectCfg(BaseModel):
    """Inspect-server UI knobs."""
    # Maximum cards shown per class in the inspect HTML. The dataset+augmented
    # tabs both honour this. Keep server mode bounded because augmented runs
    # can produce thousands of files per class; use static inspect with an
    # explicit 0 only when you really want a full self-contained dump.
    max_thumbs_per_class: int = 200


class TargetCfg(BaseModel):
    """One inference class → trigger / threshold / mining / ArtNet spec.

    `[targets]` is the single place that lists the classes inference cares about
    and how each behaves:
      - `enabled`   eligible to fire a trigger (and ArtNet, if configured).
      - `threshold` per-class decision threshold; falls back to legacy
                    `[thresholds].<class>` then `inference.default_threshold`
                    (for `_negative`, `inference.negative_max`).
      - `mine`      save detections of this class when mining is on (`--mine`
                    or `mining.enabled`). `_negative` may set `mine = true`.
      - `channel` / `value`  DMX channel + value sent on trigger.

    Example: 3 glass classes share channel 1 by using distinct values:
      clink_wine_thin   → channel=1, value=85
      clink_wine_thick  → channel=1, value=170
      clink_champagne   → channel=1, value=255
    The downstream lighting controller maps value → cue/scene.
    """
    enabled: bool = True
    threshold: float | None = None         # per-class decision threshold (see Config.class_threshold)
    mine: bool = False                     # save mined clips of this class when mining is on
    channel: int = 1                       # 1-indexed DMX channel
    value: int = 255                       # DMX value 0-255 sent on trigger


class ProjectMeta(BaseModel):
    id: str = "default"
    name: str = "Unnamed Project"
    created_at: str = ""
    description: str = ""


class Config(BaseModel):
    project: ProjectMeta = Field(default_factory=ProjectMeta)
    audio: AudioCfg = Field(default_factory=AudioCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    onset: OnsetCfg = Field(default_factory=OnsetCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    augmentation: AugmentationCfg = Field(default_factory=AugmentationCfg)
    inference: InferenceCfg = Field(default_factory=InferenceCfg)
    mining: MiningCfg = Field(default_factory=MiningCfg)
    deploy: DeployCfg = Field(default_factory=DeployCfg)
    artnet: ArtnetCfg = Field(default_factory=ArtnetCfg)
    zmq: ZmqCfg = Field(default_factory=ZmqCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    inspect: InspectCfg = Field(default_factory=InspectCfg)
    targets: dict[str, TargetCfg] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)

    def class_threshold(self, cls: str) -> float:
        """Effective decision threshold for one class.

        Resolution order: per-class `[targets].<cls>.threshold` → (for
        `_negative`) `inference.negative_max` → legacy `[thresholds].<cls>` →
        `inference.default_threshold`. The legacy `[thresholds]` block is kept
        as a fallback so older configs and frozen model snapshots keep working.
        """
        target = self.targets.get(cls)
        if target is not None and target.threshold is not None:
            return float(target.threshold)
        if cls == "_negative":
            return float(self.inference.negative_max)
        if cls in self.thresholds:
            return float(self.thresholds[cls])
        return float(self.inference.default_threshold)

    def effective_thresholds(self, classes: list[str]) -> dict[str, float]:
        """`class_threshold` for every class, as a dict."""
        return {cls: self.class_threshold(cls) for cls in classes}

    def mining_classes(self) -> set[str]:
        """Classes flagged for mining in `[targets]` (may include `_negative`)."""
        return {cls for cls, target in self.targets.items() if target.mine}

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_audio_shape(cls, data: Any) -> Any:
        """Back-compat for older configs and model snapshots.

        Feature `sample_rate`/`channels` used to live under [audio]; they now
        live under [model]. `load_config()` rejects them in user-edited TOML
        (with a clear message), but machine-generated snapshots are loaded
        straight via `model_validate` and must keep working — so copy them
        across here when a pre-move snapshot is loaded ([model] wins if set).
        """
        if isinstance(data, dict):
            data = dict(data)
            audio = data.get("audio")
            if isinstance(audio, dict):
                audio = dict(audio)
                model = data.get("model")
                model = dict(model) if isinstance(model, dict) else {}
                inference = data.get("inference")
                inference = dict(inference) if isinstance(inference, dict) else {}
                moved = False
                if "input_sample_rate" in audio and "record_sample_rate" not in audio:
                    audio["record_sample_rate"] = audio["input_sample_rate"]
                    moved = True
                if "inference_channel" in audio and "inference_channels" not in inference:
                    inference["inference_channels"] = [audio["inference_channel"]]
                    moved = True
                if "inference_channels" in audio and "inference_channels" not in inference:
                    inference["inference_channels"] = audio["inference_channels"]
                    moved = True
                for key in ("sample_rate", "channels"):
                    if key in audio and key not in model:
                        model[key] = audio[key]
                        moved = True
                if moved:
                    data = {**data, "audio": audio, "model": model, "inference": inference}
            onset = data.get("onset")
            if isinstance(onset, dict):
                onset = dict(onset)
                moved = False
                if "clip_seconds" in onset and "record_seconds" not in onset:
                    onset["record_seconds"] = onset["clip_seconds"]
                    moved = True
                if "randomize_window" in onset and "randomize_preroll" not in onset:
                    onset["randomize_preroll"] = onset["randomize_window"]
                    moved = True
                if "refractory_ms" in onset and "recording_cooldown_ms" not in onset:
                    onset["recording_cooldown_ms"] = onset["refractory_ms"]
                    moved = True
                if moved:
                    data = {**data, "onset": onset}
            inference = data.get("inference")
            if isinstance(inference, dict):
                inference = dict(inference)
                moved = False
                if "inference_channel" in inference and "inference_channels" not in inference:
                    inference["inference_channels"] = [inference["inference_channel"]]
                    moved = True
                if moved:
                    data = {**data, "inference": inference}
        return data

    @model_validator(mode="after")
    def _sync_dataset_classes(self) -> "Config":
        dataset_fields = self.dataset.model_fields_set
        selected: list[str] | None = None
        if "target_classes" in dataset_fields:
            selected = list(self.dataset.target_classes)
        elif "classes" in dataset_fields:
            selected = list(self.dataset.classes)
        if selected is not None:
            classes = list(dict.fromkeys(str(c) for c in selected))
            self.dataset.target_classes = classes
            self.dataset.classes = classes
            self.model.classes = classes
        elif self.model.classes:
            classes = list(dict.fromkeys(str(c) for c in self.model.classes))
            self.model.classes = classes
            self.dataset.target_classes = classes
            self.dataset.classes = classes
        return self

    @model_validator(mode="after")
    def _check_inference_channels(self) -> "Config":
        # record_channels and inference_channels are independent ordered
        # physical-channel selections (recording and live inference may even run
        # on different machines), so inference channels are NOT constrained to
        # the recorded set. Each command opens max(index)+1 physical channels.
        channels = list(self.inference.inference_channels)
        if self.inference.multi_mic_mode == "single" and len(channels) != 1:
            raise ValueError(
                "inference.multi_mic_mode = 'single' requires exactly one inference channel"
            )
        return self

    @model_validator(mode="after")
    def _check_augmentation_targets(self) -> "Config":
        if not self.augmentation.combinations:
            return self
        classes = set(self.model.classes)
        unknown = sorted(set(self.augmentation.combinations) - classes)
        if unknown:
            raise ValueError(
                "augmentation combination target(s) must exist in [dataset] target_classes: "
                + ", ".join(unknown)
            )
        return self


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)(=\s*)null(\s*(?:#.*)?$)",
        rf'\1"{_TOML_NULL_SENTINEL}"\2',
        text,
    )
    return _restore_toml_nulls(tomllib.loads(text))


def _restore_toml_nulls(value: Any) -> Any:
    if value == _TOML_NULL_SENTINEL:
        return None
    if isinstance(value, dict):
        return {k: _restore_toml_nulls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_toml_nulls(v) for v in value]
    return value


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
    """Load a project's user-edited config.toml.

    Project commands should pass ``project_id``. A missing project config is a
    clear setup error. ``project_id=None`` returns schema defaults for
    projectless utilities/tests that need a baseline config object.
    """
    if project_id is None:
        return Config()
    path = ProjectLayout(project_id).config_toml
    if not path.exists():
        raise FileNotFoundError(
            f"No project config at {path}. Run `sed project new {project_id}` "
            "to create a prefilled projects/<id>/config.toml."
        )
    raw = _read_toml(path)
    _check_no_legacy_keys(path, raw)
    return Config.model_validate(raw)


_LEGACY_KEY_RENAMES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dataset", "external_hardneg_pool"),
     "Rename to `external_negatives_pool` and point at `data_library/sounds`."),
    (("dataset", "external_ambience_pool"),
     "Remove this key. Move ambience/background audio to `data_library/sounds/<label>/`."),
    (("ambience",),
     "Remove this section. Track room tone/background as negative labels instead."),
    (("augmentation", "snr_mix"),
     "Remove this section. Use flat `[augmentation]` overlay settings instead."),
    (("augmentation", "pitch_shift"),
     "Remove this section. Use `pitch_shift_enabled` and `semitones_range` in `[augmentation]`."),
    (("augmentation", "hardneg_overlay"),
     "Remove this section. Use flat `[augmentation]` overlay settings instead."),
    (("augmentation", "combined"),
     "Remove this section. The single overlay pipeline already applies optional pitch shift."),
    (("augmentation", "hardneg_overlay", "per_class_hardneg_label"),
     "Rename to `per_class_negative_label`."),
    (("audio", "record_device"),
     "Move to `[deploy] record_device` (machine-local; set on the edge)."),
    (("audio", "sample_rate"),
     "Move to `[model] sample_rate` — it is the frozen feature rate the model "
     "was trained at. [audio] now holds only the live sound-card settings."),
    (("audio", "channels"),
     "Move to `[model] channels` — model input channels are part of the model "
     "shape, not the sound card."),
)


def _check_no_legacy_keys(path: Path, raw: dict[str, Any]) -> None:
    """Hard break: refuse to load configs that still use removed concepts."""
    problems: list[str] = []
    for key_path, fix in _LEGACY_KEY_RENAMES:
        node: Any = raw
        for part in key_path:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            parent = ".".join(key_path[:-1])
            key_ref = f"[{parent}] {key_path[-1]}" if parent else f"[{key_path[-1]}]"
            problems.append(f"  - {key_ref}: {fix}")
    for section_name in ("project", "external"):
        for class_name, rows in (raw.get("dataset", {}).get(section_name, {}) or {}).items():
            if not isinstance(rows, list):
                rows = [rows]
            for idx, row in enumerate(rows):
                if isinstance(row, dict) and row.get("category") == "ambience":
                    problems.append(
                        f"  - [dataset.{section_name}] {class_name}[{idx}]: "
                        "category='ambience' was removed. Move these clips to "
                        "`negatives/<label>/` and use category='negatives'."
                    )
    if problems:
        raise RuntimeError(
            f"Old config keys in {path}:\n" + "\n".join(problems)
        )


# ---------------------------------------------------------------------------
# Project layout helpers
# ---------------------------------------------------------------------------


class ProjectLayout:
    """Filesystem layout for a single project / venue."""

    def __init__(self, project_id: str):
        self.id = project_id
        self.root = PROJECTS_DIR / project_id

    @property
    def config_toml(self) -> Path:
        return self.root / PROJECT_CONFIG_NAME

    @property
    def recordings_dir(self) -> Path:
        return self.root / "recordings"

    def positives_dir(self, class_name: str) -> Path:
        return self.recordings_dir / "positives" / class_name

    def negatives_dir(self, class_name: str) -> Path:
        return self.recordings_dir / "negatives" / class_name

    # ---- Legacy project-local helpers ---------------------------------
    # New recording writes WAVs to data_library/. These helpers remain for
    # older tests/imports and for non-audio reference material.

    @property
    def test_dir(self) -> Path:
        return self.recordings_dir / "test"

    def test_positives_dir(self, class_name: str) -> Path:
        return self.test_dir / "positives" / class_name

    def test_negatives_dir(self, label: str) -> Path:
        return self.test_dir / "negatives" / label

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def dataset_inspect_dir(self) -> Path:
        """Where `sed dataset inspect` writes its HTML + outliers CSV."""
        return self.dataset_dir / "inspect"

    # ---- Models ------------------------------------------------------
    # Every `sed train` writes one self-contained model bundle under
    # models/<model_id>/: the config snapshot, checkpoint, exported OpenVINO
    # model, training curves, test metrics, confusion matrix, and the dataset
    # CSVs used. There is no separate runs/ directory — a model is selected for
    # deployment simply by name via [deploy] active_model (`sed models select`).

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def model_dir(self, model_id: str) -> Path:
        """Subfolder holding one trained model bundle (named after its id)."""
        return self.models_dir / model_id

    def list_models(self) -> list[str]:
        """Names of model subfolders that carry a config snapshot, sorted."""
        if not self.models_dir.exists():
            return []
        return sorted(
            p.name for p in self.models_dir.iterdir()
            if p.is_dir() and (p / "training_config.json").exists()
        )

    def create(self) -> None:
        for sub in [
            self.dataset_dir,
            self.models_dir,
        ]:
            sub.mkdir(parents=True, exist_ok=True)


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())


def project_config_template(project_id: str, name: str = "", created_at: str = "") -> str:
    """Return the clean, prefilled config.toml text for a new project."""
    safe_name = name or project_id
    return f'''# SoundEventDetection project config.
# Edit project-specific values here; omitted keys use defaults from sed.config.
# Dataset UI: uv run sed dataset inspect --project {project_id} --serve
# Inspect state: projects/{project_id}/dataset/dataset_selection.json

[project]
id = "{project_id}"
name = "{safe_name}"
created_at = "{created_at}"
description = ""

[dataset]
target_classes = ["_negative"]

[deploy]
active_model = ""
record_device = "default"
verbose = "none" # none | info | live

[audio]
record_sample_rate = 48000
record_channels = [0] # ordered physical input channels, e.g. [0, 3, 1]
microphone_type = "generic" # description of mic, or list for multi-mic setup e.g. ["generic", "rode"]

[onset]
min_flux_db = -80
record_seconds = 1.0
preroll_ms = 300
recording_cooldown_ms = 1500 # global delay between accepted recordings
randomize_preroll = true
preroll_ms_range = [100, 700]
multi_mic_trigger = "any" # any | mix | ch0 (multichannel: which channels can trigger)
cross_channel_merge_ms = 500 # only for "any": merge mic onsets into one clip

[model]
backbone = "mn10_as" # mn01 | mn10_as | mn20_as | mn40_as
pretrained = true
sample_rate = 32000

[features]
frontend = "efficientat" # pretrained EfficientAT preprocessing contract

[training]
epochs = 60
loss = "cross_entropy" # cross_entropy | asymmetric_focal | bce_logits
precision = "16-mixed"
auto_export = true
auto_test = true

[inference]
inference_channels = [0]
multi_mic_mode = "max" # max | mean | single
smoothing_enabled = true
default_threshold = 0.9
margin_to_second_best = 0.08
negative_max = 0.5
min_dbfs = "off" # "off" or dBFS number

# [targets] lists inference classes + trigger/threshold/mining/ArtNet specs;
# triggering AND mining read this block (mine=true saves clips, run with --mine).
[targets]
# cheers_loud = {{ enabled = true, threshold = 0.9, mine = true, channel = 1, value = 255 }}
# cheers_quiet = {{ enabled = true, threshold = 0.9, channel = 2, value = 180 }}
# _negative   = {{ enabled = false, mine = true, threshold = 0.5 }}

[thresholds]
# legacy fallback — prefer per-class threshold in [targets]
# cheers_loud = 0.90

[artnet]
enabled = false
target_ip = ""
universe = 0

[mining]
enabled = false
output_dir = "mined"
# per_class_min_confidence = {{ cheers_loud = 0.85 }}

[logging]
level = "INFO"
log_detections = true
'''


def write_project_config_template(
    layout: ProjectLayout,
    name: str = "",
    created_at: str = "",
    overwrite: bool = False,
) -> Path:
    """Create a prefilled projects/<id>/config.toml."""
    if layout.config_toml.exists() and not overwrite:
        return layout.config_toml
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.config_toml.write_text(
        project_config_template(layout.id, name=name, created_at=created_at),
        encoding="utf-8",
    )
    return layout.config_toml


# ---------------------------------------------------------------------------
# Config snapshots — for reproducibility
# ---------------------------------------------------------------------------
#
# A snapshot is the FULL effective project config serialized to JSON next to the
# trained model. `sed train` writes it to
# `projects/<id>/models/<model_id>/training_config.json` and the runtime reads it
# back for the selected model.
#
# Why: the live project config is consulted at workflow commands. If you edit it
# after training, inference must not silently change model-shape behaviour. The
# snapshot is the contract — it captures exactly what the model was trained
# against so behaviour is stable across edits.
#
# We use JSON not TOML for generated snapshots/settings because they are
# machine-generated artefacts, not user-edited input files.


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


# Blocks overlaid from the model bundle (may be tuned per selected run without
# retraining). ArtNet, logging, mining, deploy, and audio are NOT here: they are
# site-local operator settings read live from edge config.toml. `targets` stays
# because channel/value mappings are tied to the model's classes.
_RUNTIME_BLOCKS = ("targets", "thresholds")
_RUNTIME_INFERENCE_KEYS = {
    "hop_ms",
    "default_threshold",
    "consecutive_frames",
    "min_dbfs",
}


def runtime_settings_from_config(cfg: Config) -> dict[str, Any]:
    """Extract deployment/runtime knobs that may vary per selected model.

    Site-local operator settings are NOT here: [deploy], [audio], [artnet],
    [logging], and [mining] are read live from the edge config.toml, so they
    never get frozen into a model bundle.
    """
    data = cfg.model_dump(mode="json")
    out = {block: data[block] for block in _RUNTIME_BLOCKS if block in data}
    out["inference"] = {
        k: data.get("inference", {}).get(k)
        for k in _RUNTIME_INFERENCE_KEYS
        if k in data.get("inference", {})
    }
    return out


def load_runtime_settings(path: Path) -> dict[str, Any]:
    """Load generated runtime settings JSON. Missing file means no overlay."""
    import json
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("settings", payload)


def dump_runtime_settings(settings: dict[str, Any], path: Path) -> Path:
    """Write runtime settings JSON atomically."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"settings": settings}, f, indent=2)
    tmp.replace(path)
    return path


def dump_runtime_settings_from_config(cfg: Config, path: Path) -> Path:
    return dump_runtime_settings(runtime_settings_from_config(cfg), path)


def apply_runtime_settings(cfg: Config, settings: dict[str, Any]) -> Config:
    """Overlay generated runtime settings onto a frozen training config."""
    if not settings:
        return cfg
    allowed: dict[str, Any] = {}
    for block in _RUNTIME_BLOCKS:
        if block in settings:
            allowed[block] = settings[block]
    if isinstance(settings.get("inference"), dict):
        allowed["inference"] = {
            k: v for k, v in settings["inference"].items()
            if k in _RUNTIME_INFERENCE_KEYS
        }
    if not allowed:
        return cfg
    return Config.model_validate(_deep_merge(cfg.model_dump(mode="json"), allowed))


def is_efficientat_pretrained_config(cfg: Config) -> bool:
    """Whether this config uses an EfficientAT AudioSet-pretrained backbone."""
    return bool(cfg.model.pretrained) and str(cfg.model.backbone).startswith("mn")


def pretrained_feature_mismatches(cfg: Config) -> list[str]:
    """Return config fields that violate the selected pretrained checkpoint."""
    if not is_efficientat_pretrained_config(cfg):
        return []
    actual = {
        "features.frontend": cfg.features.frontend,
        "model.sample_rate": cfg.model.sample_rate,
        "model.channels": cfg.model.channels,
        "features.n_fft": cfg.features.n_fft,
        "features.win_length": cfg.features.win_length,
        "features.hop_length": cfg.features.hop_length,
        "features.n_mels": cfg.features.n_mels,
        "features.fmin": cfg.features.fmin,
        "features.fmax": cfg.features.fmax,
        "features.window_seconds": cfg.features.window_seconds,
    }
    return [
        f"{key}={actual[key]!r} (expected {expected!r})"
        for key, expected in EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT.items()
        if actual[key] != expected
    ]


def validate_pretrained_feature_contract(
    cfg: Config,
    *,
    phase: str = "configuration",
) -> None:
    """Fail early when a pretrained model is paired with incompatible features.

    Recording, training, export, and live inference all derive their clip length,
    feature sample rate, mel shape, and preprocessing from these fields. With
    EfficientAT AudioSet-pretrained backbones they must stay on the checkpoint's
    original contract; custom feature shapes should use ``model.pretrained=false``.
    """
    mismatches = pretrained_feature_mismatches(cfg)
    if not mismatches:
        return
    raise RuntimeError(
        f"EfficientAT AudioSet-pretrained {phase} requires the feature contract "
        "used by the pretrained checkpoint: "
        + "; ".join(mismatches)
        + ". Use the project template defaults, or set model.pretrained=false "
        "for an intentionally custom frontend."
    )


def resolve_active_model_dir(layout: ProjectLayout, active_model: str = "") -> Path:
    """Resolve which models/<id>/ subfolder the runtime should load.

    Priority:
      1. explicit ``active_model`` → models/<active_model>/
      2. exactly one trained model → use it
    Raises FileNotFoundError with actionable guidance otherwise.
    """
    models = layout.models_dir
    if active_model:
        d = models / active_model
        if (d / "training_config.json").exists():
            return d
        found = layout.list_models()
        raise FileNotFoundError(
            f"active_model '{active_model}' not found under {models}. "
            f"Models: {', '.join(found) if found else 'none'}. "
            f"Set [deploy] active_model in config.toml or run `sed models select`."
        )
    found = layout.list_models()
    if len(found) == 1:
        return models / found[0]
    if len(found) > 1:
        raise FileNotFoundError(
            f"Multiple models in {models} ({', '.join(found)}); "
            f"set [deploy] active_model in config.toml (or `sed models select <id>`)."
        )
    raise FileNotFoundError(
        f"No model in {models}. Train one with `sed train --project {layout.id}`, "
        f"then `sed models select <id>` (or copy a model bundle in on the edge)."
    )


def load_config_for_inference(project_id: str) -> Config:
    """Load the active model's config plus generated runtime settings.

    The active model subfolder comes from [deploy] active_model in the edge
    config.toml. Model-correctness settings (feature sample_rate + channels in
    [model], classes, thresholds, targets, ...) come from that frozen bundle.
    Machine-local settings ([audio] sound card, [deploy] model/device/verbosity,
    [inference] inference_channels / multi_mic_mode / onset_gating_enabled,
    [artnet] output, [logging], and [mining]) come from the edge config.toml, so
    one bundle runs unchanged on any machine.
    """
    layout = ProjectLayout(project_id)
    try:
        edge = load_config(project_id)
    except FileNotFoundError:
        edge = Config()
    model_dir = resolve_active_model_dir(layout, edge.deploy.active_model)
    cfg = load_config_snapshot(model_dir / "training_config.json")
    cfg = apply_runtime_settings(
        cfg, load_runtime_settings(model_dir / "runtime_settings.json")
    )
    validate_pretrained_feature_contract(cfg, phase="inference")
    # Machine-local/operator settings come from the edge config.toml, never the bundle:
    # [audio] (sound card: record_sample_rate, record_channels, blocksize,
    # microphone_type), [inference] inference_channels / multi_mic_mode /
    # onset_gating_enabled, [deploy] (active_model, record_device) plus verbosity,
    # [artnet], [logging], and [mining]. Model-shape ([model], [features]) stays
    # frozen.
    #
    # multi_mic_mode and onset_gating_enabled are deployment-topology / decision
    # strategy, not model shape (the model is a single-channel classifier run once
    # per mic), so they must track the edge config.toml — otherwise a tunable the
    # operator edits there is silently overridden by the value frozen at train
    # time. The calibratable knobs (thresholds, default_threshold,
    # consecutive_frames, hop_ms, min_dbfs) deliberately stay with the bundle /
    # runtime_settings.json and are NOT overlaid here.
    cfg.artnet = edge.artnet
    cfg.deploy = edge.deploy
    cfg.audio = edge.audio
    cfg.inference.inference_channels = list(edge.inference.inference_channels)
    cfg.inference.multi_mic_mode = edge.inference.multi_mic_mode
    cfg.inference.onset_gating_enabled = edge.inference.onset_gating_enabled
    cfg.logging = edge.logging
    cfg.mining = edge.mining
    # [targets] is the operator-facing trigger/threshold/mining spec. Classes the
    # operator lists in the edge config.toml win over the bundle's frozen targets
    # so enabling/disabling triggers, toggling `mine`, and overriding thresholds
    # take effect live (like [mining]). The bundle's calibrated [thresholds] block
    # still backs any class whose target sets no explicit threshold.
    if edge.targets:
        merged_targets = dict(cfg.targets)
        for cls, target in edge.targets.items():
            if cls in cfg.model.classes or cls in cfg.targets:
                merged_targets[cls] = target
        cfg.targets = merged_targets
    return Config.model_validate(cfg.model_dump(mode="json"))
