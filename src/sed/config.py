"""Configuration schema and project layout.

Each project has its own user-edited input config at
`projects/<id>/config.toml`. Project directories also hold generated state:
recordings, dataset snapshots, and trained model bundles under
`models/<model_id>/` (each with its own training_config.json + runtime_settings.json).
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

import math

from pydantic import BaseModel, Field, field_validator, model_validator

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"
LIBRARY_DIR = REPO_ROOT / "data_library"
# Global, cross-project preset store (audio-feature presets live under
# presets/audio_features/<name>.json). See src/sed/data/presets.py.
PRESETS_DIR = REPO_ROOT / "presets"
FEATURE_PRESETS_DIR = PRESETS_DIR / "audio_features"
PROJECT_CONFIG_NAME = "config.toml"
# The project id a clip carries when it was not recorded for any project.
# Not a real project directory: it is the shared pool every project's default
# source rule accepts alongside its own id (see data/source_filters.py), which
# is what makes project-less recording expand the library for everyone.
LIBRARY_PROJECT_ID = "_library"
_TOML_NULL_SENTINEL = "__SED_TOML_NULL__"


# An audio feature is built on two independent axes, and every frontend in this
# repo is describable as a point on both (see FeaturesCfg):
#   compression — how STFT power becomes the numbers (feature generation)
#   normalization — how those numbers are mapped into the model's range
FEATURE_COMPRESSIONS: tuple[str, ...] = ("none", "log", "db", "pcen")
FEATURE_SCALING_METHODS: tuple[str, ...] = ("none", "zscore", "minmax")
FEATURE_SCALING_SCOPES: tuple[str, ...] = ("global", "per_band")
# "fixed" constants come from a checkpoint contract or are typed by hand and are
# never fitted; "dataset" constants are fitted on the training split.
FEATURE_SCALING_SOURCES: tuple[str, ...] = ("fixed", "dataset")

# EfficientAT's published "fast normalization", (x + 4.5) / 5 over natural-log
# mel. It is a global z-score with fixed constants measured on AudioSet, so it
# is stored as one — there is no second, hidden normalization mechanism.
EFFICIENTAT_FAST_NORM_MEAN: float = -4.5
EFFICIENTAT_FAST_NORM_STD: float = 5.0

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
    # The AudioSet checkpoints expect natural-log mel put through their own
    # fixed affine. Fitting the constants on a dataset instead replaces it and
    # is off-contract.
    "features.compression": "log",
    "features.scaling_method": "zscore",
    "features.scaling_scope": "global",
    "features.scaling_source": "fixed",
    "features.scaling_stats": {
        "mean": [EFFICIENTAT_FAST_NORM_MEAN], "std": [EFFICIENTAT_FAST_NORM_STD],
    },
}


# Native/recommended ClinkCNN-v2 frontend. Alternative experiments keep this
# learned architecture but resolve their own fixed spectrogram geometry; each
# run bundle still persists one complete frontend contract.
CLINK_CNN_V2_FEATURE_CONTRACT: dict[str, Any] = {
    "features.frontend": "efficientat",
    "features.feature_type": "mel",
    "model.sample_rate": 44100,
    "model.channels": 1,
    "features.n_fft": 1024,
    "features.win_length": 882,
    "features.hop_length": 220,
    "features.n_mels": 128,
    "features.fmin": 80,
    "features.fmax": 20000,
    "features.window_seconds": 0.4,
    "features.stft_engine": "librosa",
    "features.colormap": "none",
    # Natural-log mel through the EfficientAT affine — this architecture trains
    # from scratch, so the affine is a starting point rather than a checkpoint
    # requirement; the Audio Features lab can replace it with fitted constants.
    "features.compression": "log",
    "features.scaling_method": "zscore",
    "features.scaling_scope": "global",
    "features.scaling_source": "fixed",
    "features.scaling_stats": {
        "mean": [EFFICIENTAT_FAST_NORM_MEAN], "std": [EFFICIENTAT_FAST_NORM_STD],
    },
}


def clink_cnn_v2_native_deviations(cfg: "Config") -> list[str]:
    """Ways this config departs from ClinkCNN-v2's *native* geometry.

    Advisory, never blocking. The encoder is fully convolutional with an
    adaptive frequency pool, so a different sample rate or window length builds
    and trains — it is simply not the geometry the architecture was designed
    around, which is worth saying once rather than refusing to run.
    """
    if str(cfg.model.backbone) != "clink-cnn-v2":
        return []
    out: list[str] = []
    if int(cfg.model.sample_rate) != 44_100:
        out.append(
            f"model.sample_rate={cfg.model.sample_rate!r} (native 44100)"
        )
    if not math.isclose(float(cfg.features.window_seconds), 0.4, rel_tol=0.0, abs_tol=1e-9):
        out.append(
            f"features.window_seconds={cfg.features.window_seconds!r} (native 0.4)"
        )
    return out


def clink_cnn_v2_feature_mismatches(cfg: "Config") -> list[str]:
    """Return *structurally* unsupported frontend properties for ClinkCNN-v2.

    ClinkCNN-v2 remains one fixed learned topology, but its convolutional trunk
    and adaptive frequency pool can consume several resolved one-channel
    spectrogram geometries. Only the properties the trunk genuinely cannot
    consume are listed here — a channel count it has no stem for, a value domain
    it was not built to read, or a resolved shape outside its pooling range.
    Sample rate and window length are *preferences*, not requirements, and are
    reported advisorily by :func:`clink_cnn_v2_native_deviations` instead, so any
    rate stays runnable. Parent transfer remains stricter and compares the
    complete frontend signature in :mod:`sed.train.transfer`.
    """
    if str(cfg.model.backbone) != "clink-cnn-v2":
        return []
    mismatches: list[str] = []
    f = cfg.features
    if str(f.frontend) != "efficientat":
        mismatches.append(
            f"features.frontend={f.frontend!r} (expected 'efficientat')"
        )
    if str(f.feature_type) not in {"mel", "linear"}:
        mismatches.append(
            f"features.feature_type={f.feature_type!r} (expected 'mel' or 'linear')"
        )
    if int(cfg.model.channels) != 1:
        mismatches.append(
            f"model.channels={cfg.model.channels!r} (expected 1)"
        )
    if str(f.colormap) != "none":
        mismatches.append(
            f"features.colormap={f.colormap!r} (expected 'none')"
        )
    if str(f.stft_engine) != "librosa":
        mismatches.append(
            f"features.stft_engine={f.stft_engine!r} (expected 'librosa')"
        )
    if bool(f.pcen):
        mismatches.append("features.pcen=True (ClinkCNN-v2 experiments use log power)")
    if bool(f.db_norm):
        mismatches.append("features.db_norm=True (legacy dB image normalization is unsupported)")
    if not bool(f.log_power):
        mismatches.append("features.log_power=False (raw power is unsupported)")
    if float(f.fmax) > float(cfg.model.sample_rate) / 2.0:
        mismatches.append(
            f"features.fmax={f.fmax!r} exceeds Nyquist {cfg.model.sample_rate / 2:g}"
        )

    # Import lazily: features imports Config helpers at runtime, while Config
    # validation must remain import-cycle free for all other backbones.
    if not mismatches:
        from sed.audio.features import feature_band_count, feature_frame_count_for_cfg

        # The frequency pyramid and temporal attentive pooling do have real
        # limits: outside these the trunk cannot produce an embedding at all.
        bands = feature_band_count(f, int(cfg.model.sample_rate))
        frames = feature_frame_count_for_cfg(cfg)
        if not 32 <= int(bands) <= 1024:
            mismatches.append(
                f"resolved frequency rows={bands} (expected 32..1024)"
            )
        if not 25 <= int(frames) <= 201:
            mismatches.append(
                f"resolved time frames={frames} (expected 25..201)"
            )
    # Empty mel filters are a feature-design cost, not a structural limit — the
    # trunk consumes the constant rows fine. `advice.feature_advice` explains the
    # n_fft/time-resolution tension behind them (code `empty_mel_filters`).
    return mismatches


def validate_clink_cnn_v2_feature_contract(
    cfg: "Config",
    *,
    phase: str = "configuration",
) -> None:
    """Reject a ClinkCNN-v2 config outside its supported frontend family."""
    mismatches = clink_cnn_v2_feature_mismatches(cfg)
    if mismatches:
        raise ValueError(
            f"ClinkCNN-v2 frontend rejected during {phase}: "
            + "; ".join(mismatches)
            + ". Use a mono 44.1 kHz, 0.4-second ClinkCNN-v2 Audio Features "
            "preset; the native recommendation is 128x81 Kaldi/HTK log-mel."
        )


# Backbones the trainer can build, with whether AudioSet-pretrained weights are
# available (the `_as` variants) and the feature contract that backbone expects.
# Drives the Audio Features lab: the model dropdown, the `pretrained` checkbox
# (enabled only when pretrained_available), and the recommended-feature proposal.
# Pretrained `_as` backbones recommend the frozen EfficientAT contract; the rest
# default to the same mel shape but train from scratch (any feature is allowed).
# feat_dim = make_divisible(160*width, 8) * 6 — the embedding width feeding the
# classifier head (independent of input shape thanks to global pooling).
BACKBONES: dict[str, dict[str, Any]] = {
    "mn01": {"width": 0.1, "pretrained_available": False, "feat_dim": 96},
    "mn04": {"width": 0.4, "pretrained_available": False, "feat_dim": 384},
    "mn05": {"width": 0.5, "pretrained_available": False, "feat_dim": 480},
    "mn04_as": {"width": 0.4, "pretrained_available": True, "feat_dim": 384},
    "mn05_as": {"width": 0.5, "pretrained_available": True, "feat_dim": 480},
    "mn10_as": {"width": 1.0, "pretrained_available": True, "feat_dim": 960},
    "mn20_as": {"width": 2.0, "pretrained_available": True, "feat_dim": 1920},
    "mn30": {"width": 3.0, "pretrained_available": False, "feat_dim": 2880},
    "mn40_as": {"width": 4.0, "pretrained_available": True, "feat_dim": 3840},
    # Dynamic MobileNet (DyMN): same MobileNetV3 tail/width scaling as mn, so
    # feat_dim matches the equal-width mn variant (verified by a real forward).
    "dymn04_as": {"width": 0.4, "pretrained_available": True, "feat_dim": 384},
    "dymn10_as": {"width": 1.0, "pretrained_available": True, "feat_dim": 960},
    "dymn20_as": {"width": 2.0, "pretrained_available": True, "feat_dim": 1920},
}

# The mn backbone reduces the input spectrogram 32× in both axes (5 stride-2
# stages, each (kernel, stride, padding)) before global average pooling. AudioSet
# checkpoints trained on 128 mel × ~1000 frames (10 s @ hop 320 / 32 kHz), 1 ch.
# Carried per-backbone in the registry payload so the lab's shape readout is
# driven by the selected architecture, not a hardcoded constant (future non-mn
# backbones can ship their own reduction).
MN_REDUCTION_LAYERS: tuple[tuple[int, int, int], ...] = (
    (3, 2, 1), (3, 2, 1), (5, 2, 2), (3, 2, 1), (5, 2, 2),
)
MN_DOWNSAMPLE = 32
MN_NATIVE_BANDS = 128
MN_NATIVE_FRAMES = 1000
MN_INPUT_CHANNELS = 1


def backbone_pretrained_available(backbone: str) -> bool:
    """Whether `backbone` has AudioSet-pretrained weights (the `_as` variants)."""
    name = str(backbone)
    if bool(BACKBONES.get(name, {}).get("pretrained_available", False)):
        return True
    try:
        from sed.train.backbones import get_spec

        return bool(get_spec(name).pretrained_available)
    except Exception:  # noqa: BLE001
        return False


def backbone_registry_payload() -> list[dict[str, Any]]:
    """Serializable backbone list for the model dropdown (Audio Features + Model
    tabs). Built-in mn* backbones first, then user-registered timm/custom specs."""
    payload: list[dict[str, Any]] = [
        {
            "name": name,
            "family": "efficientat",
            "width": meta["width"],
            "pretrained_available": meta["pretrained_available"],
            "feat_dim": meta["feat_dim"],
            "downsample": MN_DOWNSAMPLE,
            "reduction_layers": [list(l) for l in MN_REDUCTION_LAYERS],
            "native_bands": MN_NATIVE_BANDS,
            "native_frames": MN_NATIVE_FRAMES,
            "input_channels": MN_INPUT_CHANNELS,
            "export_supported": not str(name).startswith("dymn"),
            "recommended": EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT
            if meta["pretrained_available"]
            else None,
            "builtin": True,
        }
        for name, meta in BACKBONES.items()
    ]
    # Overlay the complete train-time registry (curated EfficientAT variants,
    # local KWS templates, timm quick picks, then user registrations). Best
    # effort: lightweight edge installs need not import torch to read config.
    try:
        from sed.train.backbones import all_specs

        for spec in all_specs().values():
            if spec.name in {row["name"] for row in payload}:
                continue
            payload.append({
                "name": spec.name,
                "family": spec.family,
                "width": None,
                "pretrained_available": spec.pretrained_available,
                "feat_dim": spec.feat_dim,
                "downsample": spec.downsample,
                "reduction_layers": [],
                "native_bands": spec.native_bands,
                "native_frames": spec.native_frames,
                "input_channels": spec.input_channels,
                "recommended": spec.input_contract,
                "source": spec.source,
                "description": spec.description,
                "params": [p.__dict__.copy() for p in spec.params],
                "provider": spec.provider or spec.family,
                "task_capabilities": list(spec.task_capabilities),
                "export_supported": bool(spec.export_supported),
                "builtin": bool(spec.builtin),
            })
    except Exception:  # noqa: BLE001 — registry is optional on edge installs
        pass
    return payload


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
    # Feature frontend. "efficientat" is the default log/fast-norm frontend used
    # by EfficientAT checkpoints; "ced" is the 16 kHz / 64-mel dB frontend used
    # by the CED transformer family. The old legacy_torchaudio_db frontend was removed; bundles
    # trained with it need an older repo revision or a retrain.
    frontend: str = "efficientat"
    # Feature type selected in the Audio Features lab. Both "mel" and "linear"
    # are trainable: "mel" with the efficientat normalization matches the
    # AudioSet-pretrained contract; "linear" (and PCEN / off-contract mel) needs
    # a from-scratch backbone (model.pretrained=false — the "Align model to
    # features" toggle). The feature designed here is the model's ground-truth input.
    feature_type: str = "mel"
    # Name of the saved preset this config was applied from (traceability only).
    preset: str = ""
    n_fft: int = 1024
    win_length: int = 800
    hop_length: int = 320
    n_mels: int = 128
    fmin: int = 0
    fmax: int = 15000  # match EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT (EfficientAT 32 kHz default)
    # 0.4 s: the target transients are ~0.2 s, and a longer window dilutes them
    # in background — the primary constraint in AGENTS.md. `onset.record_seconds`
    # stays longer on purpose so one library can serve projects at any window;
    # the crop is placed around the onset, not the clip centre (see
    # Config.window_lead_in_s and sed.train.datamodule.SedDataset._crop_start).
    window_seconds: float = 0.4
    # --- Feature generation: magnitude compression -----------------------
    # What maps STFT power to the values the network sees, before ANY
    # normalization. One axis with the same choices for mel and linear:
    #   none — linear power (what the STFT produces)
    #   log  — natural log, ln(power + 1e-5)
    #   db   — decibels, 10*log10(power)
    #   pcen — per-channel energy normalization (librosa.pcen); an adaptive
    #          gain control, so it is a compression that also self-normalizes
    # This is a *feature generation* choice: it decides what the numbers mean.
    # How those numbers are then mapped into the model's numeric range is the
    # separate `scaling_*` axis below. `log_power`, `db_norm` and `pcen` are the
    # older per-feature-type spellings and are kept mirrored by the validator.
    compression: str = "log"
    # Legacy spelling, linear only: log-power vs raw power (compression log/none).
    log_power: bool = True
    # Linear only: STFT engine. "librosa" is the standard frontend (pre-emphasis,
    # Hann window, centered frames). "scipy" reproduces the legacy soundDetection
    # system exactly: scipy.signal.spectrogram PSD with a Tukey(0.25) window,
    # per-segment mean detrend, no pre-emphasis, no centering, and amplitudes at
    # int16 full-scale (the legacy system read WAVs as raw int16).
    stft_engine: str = "librosa"
    # Legacy spelling of "dB compression + a fixed min-max normalization over
    # [db_min, db_max]" — the legacy soundDetection image normalization. Kept as
    # an input alias and mirrored back by the validator; the canonical form is
    # compression="db" with scaling_method="minmax", scaling_source="fixed".
    db_norm: bool = False
    # Edges of a fixed min-max normalization, in the compressed domain's own
    # units (dB for db compression). The legacy system used -20..60 dB.
    db_min: float = -20.0
    db_max: float = 60.0
    # Colormap image render (the legacy soundDetection model input). When
    # colormap != "none" the normalized spectrogram is rendered to a 3-channel
    # colormap image — transposed (time→rows) and resized to image_size×image_size,
    # exactly the legacy convertToSpec.py path — so the *feature itself* is the
    # image the model trains/infers on (3 channels), and the preview shows it.
    # "jet" reproduces OpenCV's COLORMAP_JET. Linear-feature only; pair with the
    # SmallerVGGNet model. "none" keeps the single-channel spectrogram.
    colormap: str = "none"
    image_size: int = 96
    # Legacy spelling of compression="pcen": Per-Channel Energy Normalization
    # (librosa.pcen) over the mel OR linear power spectrogram. Incompatible with
    # the AudioSet-pretrained weights (different value distribution), so it
    # requires a from-scratch backbone. Off by default.
    pcen: bool = False
    pcen_time_constant: float = 0.4
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    pcen_eps: float = 1e-6
    # Legacy spelling of a dataset-fitted global z-score over the compressed
    # values: (x - mean) / std. Mirrored by the validator onto scaling_method =
    # "zscore" / scaling_scope = "global" / scaling_source = "dataset".
    standardize: bool = False
    standardize_mean: float = 0.0
    standardize_std: float = 1.0
    # --- Value normalization ---------------------------------------------
    # The one normalization axis, applied AFTER the compression above and
    # nowhere else. It never changes what the numbers mean, only the range they
    # arrive in:
    #   none   — the compressed values reach the model unchanged
    #   zscore — (x - mean) / std
    #   minmax — clip((x - lo) / (hi - lo), 0, 1)
    # scope "global" uses one value pair over all bands; "per_band" uses one
    # pair per feature row (mel band / linear bin) — the usual SED choice at
    # high band counts. The constants come from one of two places:
    #   dataset — fitted on the live grouped TRAINING split only (validation
    #             excluded) and frozen into the model bundle
    #   fixed   — pinned by a checkpoint contract or typed by hand; nothing is
    #             fitted, so the training launch gate does not demand a fit.
    # Both EfficientAT's (x + 4.5) / 5 fast-norm and the legacy [-20, 60] dB
    # image map are ordinary "fixed" specs — z-score and min-max respectively —
    # so mel and linear features are normalized by the same mechanism.
    scaling_method: str = "none"    # none | zscore | minmax
    scaling_scope: str = "global"   # global | per_band
    scaling_source: str = "dataset" # dataset | fixed
    # zscore: {"mean": [...], "std": [...]}; minmax: {"lo": [...], "hi": [...]}.
    # One entry per statistic when global. Empty with source="dataset" = not
    # fitted yet (the training launch gate enforces fitting; extractors pass
    # values through unscaled until then).
    scaling_stats: dict[str, list[float]] = Field(default_factory=dict)
    # Dataset-fitted scaling provenance (shared by every scaling_method).
    # Existing manually specified mel contracts remain resolved by default;
    # experimental linear presets set this false until the Audio Features
    # estimator fills the statistics.
    standardize_stats_resolved: bool = True
    standardize_stats_dataset_sha256: str = ""
    standardize_stats_split_seed: int = 0
    # The grouped validation fraction also changes which clips supply the
    # train-only moments.  Keep it with the fitted provenance so changing the
    # Train-tab split setting cannot silently reuse moments from another split.
    standardize_stats_val_fraction: float = 0.0
    standardize_stats_clips: int = 0

    @model_validator(mode="after")
    def _check_features(self) -> "FeaturesCfg":
        self.frontend = str(self.frontend or "efficientat").lower()
        if self.frontend not in {"efficientat", "ced"}:
            raise ValueError(
                "features.frontend must be 'efficientat' or 'ced' — the "
                "legacy_torchaudio_db frontend was removed (retrain the model "
                "or pin an older repo revision)"
            )
        self.feature_type = str(self.feature_type or "mel").lower()
        if self.feature_type not in ("mel", "linear"):
            raise ValueError("features.feature_type must be 'mel' or 'linear'")
        self._resolve_compression()
        if self.n_fft <= 0 or self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("features n_fft, win_length, and hop_length must be positive")
        if self.window_seconds <= 0:
            raise ValueError("features.window_seconds must be positive")
        # n_mels only constrains the mel path; linear ignores it.
        if self.feature_type == "mel" and self.n_mels <= 0:
            raise ValueError("features.n_mels must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("features.win_length must be <= n_fft")
        if self.fmin < 0 or self.fmax <= self.fmin:
            raise ValueError("features must satisfy 0 <= fmin < fmax")
        if self.pcen and self.pcen_power <= 0:
            raise ValueError("features.pcen_power must be positive")
        self.stft_engine = str(self.stft_engine or "librosa").lower()
        if self.stft_engine not in ("librosa", "scipy"):
            raise ValueError("features.stft_engine must be 'librosa' or 'scipy'")
        # The scipy engine reproduces the legacy linear PSD; the mel path always
        # uses the efficientat frontend, so a mel config carrying it would
        # silently mean something else — reject it instead.
        if self.feature_type == "mel" and self.stft_engine != "librosa":
            raise ValueError("features.stft_engine='scipy' requires feature_type='linear'")
        if self.db_max <= self.db_min:
            raise ValueError("features.db_max must be greater than db_min")
        self.colormap = str(self.colormap or "none").lower()
        if self.colormap not in ("none", "jet"):
            raise ValueError("features.colormap must be 'none' or 'jet'")
        if self.colormap != "none" and self.feature_type != "linear":
            raise ValueError("features.colormap requires feature_type='linear'")
        if self.colormap != "none" and self.pcen:
            raise ValueError("features.colormap is incompatible with pcen")
        if self.colormap != "none" and self.image_size <= 0:
            raise ValueError("features.image_size must be positive when colormap is set")
        if self.colormap == "jet" and (
            self.stft_engine != "scipy"
            or not self.db_norm
            or abs(float(self.db_min) - (-20.0)) > 1e-9
            or abs(float(self.db_max) - 60.0) > 1e-9
        ):
            raise ValueError(
                "features.colormap='jet' is the legacy reproduction mode and requires "
                "linear SciPy STFT with db_norm=True and the fixed [-20, 60] dB range"
            )
        if self.standardize and self.standardize_std <= 0:
            raise ValueError("features.standardize_std must be positive")
        if self.standardize_stats_clips < 0:
            raise ValueError("features.standardize_stats_clips must be >= 0")
        if not 0.0 <= self.standardize_stats_val_fraction < 1.0:
            raise ValueError("features.standardize_stats_val_fraction must be in [0, 1)")
        self._resolve_scaling()
        self._check_scaling_stats()
        if self.frontend == "ced":
            if self.feature_type != "mel":
                raise ValueError("features.frontend='ced' requires feature_type='mel'")
            if (self.n_fft, self.win_length, self.hop_length) != (512, 512, 160):
                raise ValueError("features.frontend='ced' requires n_fft=512, win_length=512, hop_length=160")
            if self.n_mels != 64:
                raise ValueError("features.frontend='ced' requires n_mels=64")
            if self.fmin != 0 or self.fmax != 8000:
                raise ValueError("features.frontend='ced' requires fmin=0 and fmax=8000")
            if self.pcen:
                raise ValueError("features.frontend='ced' is incompatible with pcen")
            if self.colormap != "none":
                raise ValueError("features.frontend='ced' is 1-channel only; colormap must be 'none'")
        return self

    def _resolve_compression(self) -> None:
        """Settle the compression axis and keep the legacy flags mirrored.

        ``compression`` is canonical. When a config was written with only the
        old per-feature-type flags (``pcen`` / ``db_norm`` / ``log_power``) it
        is derived from them, so every stored project, preset and frozen bundle
        keeps producing byte-identical values.
        """
        if self.pcen:
            implied = "pcen"
        elif self.db_norm or self.frontend == "ced":
            # The CED checkpoints are a dB-mel contract; naming it makes the
            # frontend describable in the same vocabulary as everything else.
            implied = "db"
        elif self.feature_type == "linear" and not self.log_power:
            implied = "none"
        else:
            implied = "log"
        value = str(self.compression or "").lower()
        # The canonical field wins, except when it still holds the default while
        # the old flags name something else — that is a config written before the
        # field existed, or one whose flags were edited without it.
        if value not in FEATURE_COMPRESSIONS or (value == "log" and implied != "log"):
            value = implied
        if value not in FEATURE_COMPRESSIONS:
            raise ValueError(
                "features.compression must be one of " + ", ".join(FEATURE_COMPRESSIONS)
            )
        self.compression = value
        # Mirror back so readers of the old flags stay correct.
        self.pcen = value == "pcen"
        self.log_power = value != "none"
        if value == "pcen" and self.pcen_power <= 0:
            raise ValueError("features.pcen_power must be positive")
        if value == "pcen" and self.colormap != "none":
            raise ValueError("features.colormap is incompatible with pcen")

    def _resolve_scaling(self) -> None:
        """Settle the normalization axis and keep the legacy flags mirrored.

        Three older spellings all become ordinary ``scaling_*`` specs, so the
        pipeline has exactly one normalization stage:

        * ``db_norm``           → fixed min-max over ``[db_min, db_max]``
        * ``standardize``       → dataset z-score, global scope
        * the implicit EfficientAT ``(x + 4.5) / 5`` fast-norm, which was never
          a config field at all → fixed z-score, ``mean=-4.5``, ``std=5``

        Each mapping is numerically exact, so an existing project keeps feeding
        its model the same values while gaining an explicit description of them.
        """
        self.scaling_method = str(self.scaling_method or "none").lower()
        if self.scaling_method not in ("none", "zscore", "minmax"):
            raise ValueError("features.scaling_method must be 'none', 'zscore', or 'minmax'")
        self.scaling_scope = str(self.scaling_scope or "global").lower()
        if self.scaling_scope not in ("global", "per_band"):
            raise ValueError("features.scaling_scope must be 'global' or 'per_band'")
        self.scaling_source = str(self.scaling_source or "dataset").lower()
        if self.scaling_source not in FEATURE_SCALING_SOURCES:
            raise ValueError(
                "features.scaling_source must be one of "
                + ", ".join(FEATURE_SCALING_SOURCES)
            )

        if self.db_norm and self.standardize:
            raise ValueError(
                "features.db_norm and features.standardize are two different "
                "normalizations of the same values — choose one"
            )
        # Is the current spec still 'unset' as far as the legacy flags go? Either
        # literally none, or the implicit fast-norm a previous validation filled
        # in (so a config re-validated after an in-place edit still migrates).
        is_fast_norm = (
            self.scaling_method == "zscore"
            and self.scaling_source == "fixed"
            and self.scaling_scope == "global"
            and self.scaling_stats.get("mean") == [EFFICIENTAT_FAST_NORM_MEAN]
            and self.scaling_stats.get("std") == [EFFICIENTAT_FAST_NORM_STD]
        )
        if is_fast_norm and not (
            self.frontend == "efficientat"
            and self.feature_type == "mel"
            and self.compression == "log"
            and self.colormap == "none"
        ):
            # AudioSet's affine describes log-mel and nothing else. Switching
            # basis or compression must not silently drag its constants along —
            # they would be an arbitrary offset and gain on the new values.
            self.scaling_method, self.scaling_stats = "none", {}
            self.scaling_source = "dataset"
            is_fast_norm = False
        unset = self.scaling_method == "none" or is_fast_norm
        if self.db_norm and unset:
            self.scaling_method, self.scaling_scope = "minmax", "global"
            self.scaling_source = "fixed"
            self.scaling_stats = {
                "lo": [float(self.db_min)], "hi": [float(self.db_max)],
            }
        elif self.standardize and unset:
            self.scaling_method, self.scaling_scope = "zscore", "global"
            self.scaling_source = "dataset"
            self.scaling_stats = {
                "mean": [float(self.standardize_mean)],
                "std": [float(self.standardize_std)],
            }
        elif (
            self.scaling_method == "none"
            # Stats without a method are a mistake, not an implicit contract;
            # _check_scaling_stats reports it rather than being overwritten here.
            and not self.scaling_stats
            and self.frontend == "efficientat"
            and self.feature_type == "mel"
            and self.compression == "log"
            and self.colormap == "none"
        ):
            # The AudioSet fast-norm, made explicit. Exactly (x + 4.5) / 5.
            self.scaling_method, self.scaling_scope = "zscore", "global"
            self.scaling_source = "fixed"
            self.scaling_stats = {
                "mean": [EFFICIENTAT_FAST_NORM_MEAN],
                "std": [EFFICIENTAT_FAST_NORM_STD],
            }

        if self.scaling_source == "fixed" and not self.standardize_stats_resolved:
            # "Declared but not fitted yet" is only meaningful for constants that
            # were going to be fitted. Fixed constants are complete by
            # construction, so an unresolved spec is a dataset one.
            self.scaling_source = "dataset"
        fixed = self.scaling_source == "fixed"
        if fixed and self.scaling_method != "none" and not self.scaling_stats:
            raise ValueError(
                "features.scaling_source='fixed' needs explicit features.scaling_stats — "
                "fixed constants are never fitted"
            )

        # --- legacy mirrors -------------------------------------------------
        lo = (self.scaling_stats.get("lo") or [None])[0]
        hi = (self.scaling_stats.get("hi") or [None])[0]
        self.db_norm = bool(
            self.compression == "db"
            and self.scaling_method == "minmax"
            and self.scaling_scope == "global"
            and fixed
            and lo is not None
            and hi is not None
        )
        if self.db_norm:
            self.db_min, self.db_max = float(lo), float(hi)
        # ``standardize`` keeps its exact historical meaning: a *fitted* global
        # z-score over log values, replacing the efficientat frontend's own
        # affine. Fixed contract constants, PCEN and dB z-scores are expressible
        # only canonically, so the bool never widens beyond what old bundles and
        # presets meant by it.
        self.standardize = bool(
            self.scaling_method == "zscore"
            and self.scaling_scope == "global"
            and self.scaling_source == "dataset"
            and self.compression == "log"
            and self.frontend == "efficientat"
        )
        if self.standardize:
            if self.scaling_stats.get("mean") and self.scaling_stats.get("std"):
                self.standardize_mean = float(self.scaling_stats["mean"][0])
                self.standardize_std = float(self.scaling_stats["std"][0])
            else:
                self.scaling_stats = {
                    "mean": [float(self.standardize_mean)],
                    "std": [float(self.standardize_std)],
                }
            if self.standardize_std <= 0:
                raise ValueError("features.standardize_std must be positive")

        if self.scaling_method != "none" and self.colormap != "none" and not self.db_norm:
            raise ValueError(
                "features.colormap renders an already-[0,1] image, so it only accepts the "
                "fixed dB min-max normalization (db_norm)"
            )

    def _check_scaling_stats(self) -> None:
        """Coerce + sanity-check whatever fitted scaling statistics are present.

        Empty stats are allowed (not fitted yet — the training launch gate
        enforces fitting before a run starts); present stats must be coherent.
        Band-count agreement with the feature geometry is checked where the
        sample rate is known (``resolve_feature_scaling`` in sed.audio.features).
        """
        stats = {
            str(k): [float(v) for v in (vals or [])]
            for k, vals in (self.scaling_stats or {}).items()
        }
        self.scaling_stats = stats
        if not stats:
            return
        if self.scaling_method == "none":
            raise ValueError(
                "features.scaling_stats requires a scaling_method of 'zscore' or 'minmax'"
            )
        keys = ("mean", "std") if self.scaling_method == "zscore" else ("lo", "hi")
        extra = sorted(set(stats) - set(keys))
        if extra:
            raise ValueError(
                f"features.scaling_stats has unexpected keys {extra} for "
                f"scaling_method={self.scaling_method!r}"
            )
        a, b = stats.get(keys[0]) or [], stats.get(keys[1]) or []
        if not a or len(a) != len(b):
            raise ValueError(
                f"features.scaling_stats needs equal-length non-empty "
                f"'{keys[0]}' and '{keys[1]}' lists"
            )
        if self.scaling_scope == "global" and len(a) != 1:
            raise ValueError(
                "features.scaling_scope='global' takes exactly one fitted value per statistic"
            )
        if self.scaling_method == "zscore":
            if any(s <= 0.0 for s in b):
                raise ValueError("features.scaling_stats std values must be positive")
        elif any(hi <= lo for lo, hi in zip(a, b)):
            raise ValueError("features.scaling_stats requires hi > lo for every band")

    def renders_image(self) -> bool:
        """Whether this feature is a multi-channel colormap image (vs a 1-ch spec)."""
        return str(self.colormap or "none").lower() != "none"

    def matches_pretrained_normalization(self) -> bool:
        """True only for the mel + efficientat log/fast-norm pretrained contract."""
        return (
            self.frontend == "efficientat"
            and self.feature_type == "mel"
            and self.compression == "log"
            and self.scaling_method == "zscore"
            and self.scaling_scope == "global"
            and self.scaling_source == "fixed"
            and abs(
                float((self.scaling_stats.get("mean") or [0.0])[0])
                - EFFICIENTAT_FAST_NORM_MEAN
            ) < 1e-9
            and abs(
                float((self.scaling_stats.get("std") or [1.0])[0])
                - EFFICIENTAT_FAST_NORM_STD
            ) < 1e-9
        )

    def compression_label(self) -> str:
        """Human name of the compression stage, for UI and manifests."""
        return {
            "none": "linear power", "log": "log power",
            "db": "decibels", "pcen": "PCEN (adaptive gain)",
        }[self.compression]

    def normalization_label(self) -> str:
        """Human name of the normalization stage, for UI and manifests."""
        if self.scaling_method == "none":
            return "none"
        name = "z-score" if self.scaling_method == "zscore" else "min-max"
        scope = "per band" if self.scaling_scope == "per_band" else "global"
        return f"{name} · {scope} · {self.scaling_source}"


class OnsetCfg(BaseModel):
    algo: str = "superflux"
    hop_ms: int = 32
    # Mel bands for the trigger's spectral-flux analysis. Intentionally coarser
    # than the model feature (64 vs 128): the trigger only needs to find an
    # onset, not classify it. Exposed here so it is config-tunable rather than a
    # buried constant in the detector.
    trigger_n_mels: int = 64
    median_window_s: float = 5.0
    mad_k: float = 3.0
    preroll_ms: int = 120
    # Derived from record_seconds and preroll_ms; recording code uses this value.
    postroll_ms: int = 280
    # Global delay between accepted recordings. Unlike the detector's small
    # internal anti-double-trigger refractory, this applies across all selected
    # mics: once any mic starts a recording group, no mic can start another group
    # until this cooldown expires.
    recording_cooldown_ms: int = 600
    # Operators tune min_flux_db; code keeps the matching linear value for the
    # onset detector.
    min_flux: float = 1e-4
    min_flux_db: float = -80.0
    record_seconds: float = 0.4
    # Per-clip preroll randomisation. When True, each onset-triggered clip gets
    # a preroll drawn uniformly from `preroll_ms_range`; the postroll auto-
    # adjusts so the total saved length stays exactly record_seconds. Replaces
    # the post-processing time-shift augmentation — temporal jitter now lives
    # at recording time so the spectrogram is never wrapped or zero-padded.
    randomize_preroll: bool = True
    preroll_ms_range: tuple[int, int] = (60, 240)
    # Multichannel triggering. With "any" each recorded channel runs its own
    # Multichannel triggering. With "any" each recorded channel runs its own
    # onset detector and ANY of them can trigger a clip; near-simultaneous hits
    # across channels are merged and only the fired physical channels are saved.
    # "any_all" triggers on ANY channel onset but saves ALL recorded channels.
    # "mix" runs a single detector on the channel downmix (cheaper, but a
    # loud-on-one-mic transient can be diluted). "ch0" is the legacy behaviour.
    multi_mic_trigger: str = "any"   # any | any_all | mix | ch0
    # Window for collapsing the same physical transient detected on several
    # channels into a single recording. Applies only to multi_mic_trigger="any"
    # or "any_all", where each selected mic runs its own onset detector.
    cross_channel_merge_ms: int = 500

    @model_validator(mode="after")
    def _check_multi_mic_trigger(self) -> "OnsetCfg":
        if self.multi_mic_trigger not in {"any", "any_all", "mix", "ch0"}:
            raise ValueError(
                f"onset.multi_mic_trigger must be one of any|any_all|mix|ch0 "
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
    # A recipe makes the model choice reproducible. ``backbone`` and
    # ``pretrained`` remain the compatibility surface used by existing projects;
    # the fields below pin the more precise choice made in the Model Lab.
    recipe_id: str = ""
    weight_id: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    # v2 recipes accept only catalog/project-declared constructor parameters.
    # v1 remains readable for frozen legacy bundles.
    params_contract_version: int = 1
    task: str = "clip"  # clip | frame
    initialization: str = "upstream"  # random | upstream | parent
    parent_project: str = ""
    parent_run: str = ""
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
        self.task = str(self.task or "clip").strip().lower()
        if self.task not in {"clip", "frame"}:
            raise ValueError("model.task must be 'clip' or 'frame'")
        self.initialization = str(self.initialization or "upstream").strip().lower()
        if self.initialization not in {"random", "upstream", "parent"}:
            raise ValueError("model.initialization must be random, upstream, or parent")
        if self.initialization == "parent" and (not self.parent_project or not self.parent_run):
            raise ValueError("model.parent_project and model.parent_run are required for parent initialization")
        if not isinstance(self.params, dict):
            raise ValueError("model.params must be a table/object")
        if self.params_contract_version < 1:
            raise ValueError("model.params_contract_version must be >= 1")
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
    # Fraction of each class held out for validation (session-grouped,
    # leakage-free). Applies to every project, clip/frame alike. Not a
    # compose-time artifact: the training pipeline carves val out of
    # train.csv live, at train start, seeded by [training] seed.
    val_fraction: float = Field(default=0.15, ge=0.0, lt=0.9)
    # Optional split-derived frozen test. Compose carves a deterministic,
    # session-grouped slice from the training pool, persists it under
    # dataset/frozen_test_split.json, and reuses it so the same held-out clips
    # score every run. ``None`` means "project default": on for pretraining
    # projects, off for venue projects.
    split_test_enabled: bool | None = None
    split_test_fraction: float = Field(default=0.1, ge=0.0, lt=0.9)
    # Deprecated aliases kept for older configs and UI callers. They still
    # influence the effective split-test default when the unified keys are not
    # set.
    pretraining_test_enabled: bool = True
    pretraining_test_fraction: float = Field(default=0.1, ge=0.0, lt=0.9)
    sources: dict[str, list[DatasetSourceRule]] = Field(default_factory=dict)
    # Deprecated aliases. They still load, then normalize into `sources`.
    project: dict[str, Any] = Field(default_factory=dict)
    external: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_sources(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "split_test_fraction" not in data and "pretraining_test_fraction" in data:
            data["split_test_fraction"] = data["pretraining_test_fraction"]
        if "split_test_enabled" not in data and "pretraining_test_enabled" in data:
            data["split_test_enabled"] = data["pretraining_test_enabled"]
        raw_sources = data.get("sources")
        if isinstance(raw_sources, dict):
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
    # Number of physical microbatches accumulated before one optimizer step.
    # Effective batch = batch_size * accumulate_grad_batches.
    accumulate_grad_batches: int = 1
    epochs: int = 60
    lr_backbone: float = 1e-4
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    # Dropout on the shared classifier head (before the final Linear). This is a
    # *training* knob, not part of the model recipe: it never reaches
    # ``recipe_signature`` and is an identity at eval/export, so raising it does
    # not make old runs incomparable. Dropout inside a backbone is separate and
    # belongs to that architecture's recipe params (Model tab).
    head_dropout: float = 0.2
    # Automatic mode resolves the effective classifier dropout from the model's
    # architecture regularization and target fine-tune depth. Frozen snapshots
    # materialize a numeric/manual value so their training contract cannot drift.
    head_dropout_auto: bool = True
    # Linear LR warmup before cosine decay over the full `epochs` budget (early
    # stopping usually ends first, so the LR rarely reaches zero — expected).
    # Independent of `fine_tune_warmup_epochs`, which controls when the frozen
    # encoder unfreezes, not the LR schedule.
    warmup_epochs: int = 5
    patience: int = 10
    seed: int = 42
    # One model-agnostic class-imbalance policy. ``off`` preserves natural row
    # frequency, ``sqrt`` tempers class-frequency skew, and ``balanced`` gives
    # background and all target classes equal top-level exposure. Both enabled
    # modes also temper large recording groups so adjacent/augmented clips from
    # one source cannot dominate an epoch. Losses remain unweighted: sampling
    # is the single place where class balance is applied.
    class_balance: str = "off"  # off | sqrt | balanced
    # How much of the backbone to train ("train depth"), counted in trailing
    # stages so it generalizes across architectures: "head" (freeze backbone,
    # train head only), "last_block", "last_n:<k>", "half", or "full".
    train_depth: str = "full"
    # Named, stable policies exposed in the Train lab. ``manual`` preserves the
    # legacy train_depth semantics for hand-authored configs.
    fine_tune_policy: str = "full"  # manual|head|last_stage|last_two|full
    # Head-only epochs before the policy unfreezes encoder stages. A fresh
    # classifier emits large, meaningless gradients; keeping the encoder frozen
    # until the head is sane protects the pretrained representation.
    # 0 = auto, resolved in `sed.train.fine_tuning.resolve_fine_tune_plan` as
    #   warmup = min(5, max(1, round(epochs * 0.1)))   then clamped to epochs-1
    # so epochs=40 -> 4 head-only epochs and the encoder unfreezes at the start
    # of epoch 5. The timing depends only on `epochs`, never on the backbone;
    # the model decides the *depth* (how many stages open), not the *when*.
    # A positive value overrides the auto rule. Ignored by head-only policies
    # and by random/scratch init, which train fully from epoch 1.
    # Distinct from `warmup_epochs`, which is the LR ramp and keeps running
    # during the frozen phase.
    fine_tune_warmup_epochs: int = 0
    # Pretraining projects may either use their labelled corpus directly or
    # train a representation first. BYOL is deliberately architecture-agnostic
    # and can run with any clip-capable encoder.
    pretraining_objective: str = "supervised"  # supervised | byol
    ssl_ema_decay: float = 0.996
    # Keep frozen-stage BatchNorm running stats fixed, and (when set) freeze
    # BatchNorm affine params in the trainable stages too — standard for
    # fine-tuning on small datasets.
    freeze_batchnorm: bool = False
    # What to do with the classifier head when initializing from a parent run
    # (`model.initialization = "parent"`). The encoder always transfers; this is
    # only about the final Linear.
    #
    # "matching_classes" (default) copies the parent's weight row and bias for
    # every class name that exists in BOTH runs, and leaves new classes random.
    # The parent's rows were fitted against the exact embedding the child starts
    # from, so a shared class begins near its solution instead of re-deriving it
    # — this is what makes a chained pretrain → fine-tune converge in the epoch
    # budget the presets assume. "random" restores the older behaviour of always
    # starting the head from scratch.
    transfer_head_init: str = "matching_classes"  # matching_classes | random
    # Explicit "this run's class ← parent class" rows for the classifier warm
    # start, applied on top of exact-name matching (which never needs an entry).
    #
    # The two stages of this pipeline rarely share positive class names: stage
    # one labels the umbrella event (`cheers`) and stage two labels the vessels
    # (`riedel_wine`, …), so exact matching warm-starts `_negative` and leaves
    # every target row random — at a fine-tune learning rate that is the usual
    # cause of a collapsed transfer. Mapping the vessels onto the parent's
    # `cheers` row starts them from a solution that already lives in this exact
    # embedding. Several children may share one parent row; each copy gets a
    # small deterministic jitter so they do not start bit-identical.
    #
    # Ignored unless `model.initialization = "parent"` and
    # `transfer_head_init = "matching_classes"`.
    transfer_head_map: dict[str, str] = Field(default_factory=dict)
    # Per-stage LR decay applied stem→head across trainable backbone stages
    # (1.0 = uniform lr_backbone, as before; <1.0 gives earlier stages a lower LR).
    lr_layer_decay: float = 1.0
    loss: str = "bce_logits"
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 2.0
    # SpecAugment-style masking on training spectrograms only. Off by default;
    # enable after a clean baseline when extra invariance is useful. Frequency
    # masks apply to feature rows (mel bands or linear-frequency bins).
    specaugment_enabled: bool = False
    specaugment_freq_mask: int = 24   # max masked feature rows per mask
    specaugment_time_mask: int = 16   # max masked frames per mask (of ~100)
    # Batch mixup on training spectrograms + soft targets (multi-label BCE clip
    # task only). 0 disables; 0.2-0.4 is the usual range for audio tagging.
    mixup_alpha: float = 0.0
    num_workers: int = 4
    precision: str = "16-mixed"
    auto_export: bool = True
    # Which frozen-test decision rules to score when training finishes, as rule
    # ids ("fixed@0.90", "argmax", "deployed", …) — the same ids Batch evaluate
    # writes, so a fresh run lands in the same runs-table columns as a re-scored
    # one and needs no separate evaluation to be comparable. Empty = no scoring.
    # `auto_test` / `auto_test_threshold` below are the pre-rule spelling of the
    # same intent, kept so older config.toml files keep working; when
    # `auto_test_rules` is unset they are what `resolved_auto_test_rules()`
    # derives from.
    auto_test_rules: list[str] | None = None
    auto_test: bool = True
    auto_test_threshold: float = 0.90
    # Optional: fit per-class trigger thresholds on the run's val split after
    # training and SAVE them into the bundle. That saved layer is what the
    # "deployed" rule then scores; it changes decisions, never learned weights.
    calibrate_thresholds: bool = False
    # Optional after-training false-alarm calibration: sweep thresholds against
    # recorded negative CLIPS until the system fires no more than
    # `calibrate_per_1000` times per 1000 of them, then save into the bundle.
    #
    # Per 1000 clips, not per hour: negatives are captured as onset-triggered
    # batches while the room is busy, so "clips" is the denominator the material
    # actually supports. Hours are unknown — nothing records how long the batch
    # was drawn from. `calibrate_fph` is accepted as a legacy alias.
    calibrate_after: bool = False
    calibrate_per_1000: float = 1.0
    calibrate_negsrc: str = "venue"
    # Optional after-training step: re-estimate BatchNorm running statistics by
    # streaming training batches through the frozen weights (only BN layers in
    # train mode). Useful after deep freezing, mixup, or small-batch fine-tunes
    # where running stats drifted from the deployment input distribution; it
    # deliberately overrides freeze_batchnorm's preserved upstream stats. The
    # recalibrated checkpoint feeds calibration/auto-test/export downstream.
    bn_recalibrate: bool = False
    bn_recalibrate_batches: int = 50
    # Which artifacts auto-export builds, as a subset of ("onnx", "fp16",
    # "int8"). Default stops at FP16: INT8 is 3-4x faster but is lossy, and it
    # should only reach production after the regression below has been measured.
    export_targets: list[str] = Field(default_factory=lambda: ["onnx", "fp16"])
    # After building INT8, score BOTH IRs on the frozen test and deploy INT8
    # only if it stays within `int8_max_regression`. Without this the export
    # deploys INT8 whenever quantization merely *succeeds*, which says nothing
    # about accuracy.
    export_verify_int8: bool = True
    # How much recall-at-budget INT8 may give up against FP16 before the export
    # points `active_openvino.xml` back at FP16 (some backbones — e.g.
    # MobileNetV3's SE blocks — lose several points to post-training
    # quantization). Negative = always keep INT8.
    #
    # Measured on recall at the false-alarm budget rather than F1: F1 moves with
    # the test set's negative:positive balance, so an F1 tolerance means
    # different things on different benchmarks. `int8_max_f1_regression` is
    # accepted as the legacy spelling.
    int8_max_regression: float = 0.02

    @model_validator(mode="before")
    @classmethod
    def _migrate_int8_tolerance(cls, data: Any) -> Any:
        """Accept the old F1-based spelling of the INT8 deploy tolerance."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "int8_max_f1_regression" in values and "int8_max_regression" not in values:
            values["int8_max_regression"] = values["int8_max_f1_regression"]
        values.pop("int8_max_f1_regression", None)
        return values

    @model_validator(mode="before")
    @classmethod
    def _migrate_calibrate_budget(cls, data: Any) -> Any:
        """Accept the old per-hour budget key.

        The unit changed (per hour -> per 1000 clips) when calibration moved to
        recorded clip batches, whose denominator is clips. The numeric value
        carries over unchanged: both express "how many false fires I tolerate",
        and every project on disk set it to a small integer.
        """
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "calibrate_fph" in values and "calibrate_per_1000" not in values:
            values["calibrate_per_1000"] = values["calibrate_fph"]
        values.pop("calibrate_fph", None)
        return values

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_class_balance(cls, data: Any) -> Any:
        """Collapse old sampling/loss knobs into the canonical policy.

        Historical snapshots may contain both ``sampling_strategy`` and
        ``negative_weight``. A group-balanced sampler takes precedence; a
        standalone negative weight above one becomes the safer tempered mode.
        Explicit ``class_balance`` always wins. The legacy keys are discarded
        so new snapshots and UI payloads expose only one source of truth.
        """
        if not isinstance(data, dict):
            return data
        values = dict(data)
        # Old explicit values were deliberate and remain manual. Projects that
        # omitted the old default adopt the new model-aware automatic behavior.
        if "head_dropout_auto" not in values:
            values["head_dropout_auto"] = "head_dropout" not in values
        if "class_balance" not in values:
            strategy_raw = values.get("sampling_strategy")
            strategy = str(strategy_raw or "natural").strip().lower()
            if strategy not in {"natural", "class_group_sqrt"}:
                raise ValueError(
                    "legacy training.sampling_strategy must be 'natural' or "
                    "'class_group_sqrt'"
                )
            if strategy == "class_group_sqrt":
                values["class_balance"] = "balanced"
            elif "negative_weight" in values:
                try:
                    legacy_weight = float(values["negative_weight"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "legacy training.negative_weight must be numeric"
                    ) from exc
                if legacy_weight <= 0:
                    raise ValueError(
                        "legacy training.negative_weight must be positive"
                    )
                values["class_balance"] = "sqrt" if legacy_weight > 1.0 else "off"
        values.pop("sampling_strategy", None)
        values.pop("negative_weight", None)
        return values

    @model_validator(mode="after")
    def _check_training(self) -> "TrainingCfg":
        allowed_losses = {
            "cross_entropy", "ce", "softmax_ce", "categorical_crossentropy",
            "asymmetric_focal",
            "bce", "bce_logits", "binary_cross_entropy",
        }
        self.loss = self.loss.strip().lower()
        if self.loss not in allowed_losses:
            raise ValueError(
                "training.loss must be one of: cross_entropy, categorical_crossentropy, "
                "asymmetric_focal, bce_logits"
            )
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("training batch_size and epochs must be positive")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("training.accumulate_grad_batches must be positive")
        if self.lr_backbone <= 0 or self.lr_head <= 0:
            raise ValueError("training learning rates must be positive")
        if self.weight_decay < 0 or self.warmup_epochs < 0 or self.patience < 0:
            raise ValueError("training weight_decay, warmup_epochs and patience must be >= 0")
        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError("training.head_dropout must be in [0, 1) (0 disables it)")
        if self.specaugment_freq_mask < 0 or self.specaugment_time_mask < 0:
            raise ValueError("training specaugment mask widths must be >= 0")
        if self.mixup_alpha < 0:
            raise ValueError("training.mixup_alpha must be >= 0 (0 disables mixup)")
        if self.num_workers < 0:
            raise ValueError("training num_workers must be >= 0")
        # Lightning historically accepted the short ``32`` spelling.  Keep
        # old project files and run snapshots loadable, but persist one
        # unambiguous spelling for each precision mode exposed by the Train
        # tab.  In particular, ``32`` and ``32-true`` are not two different
        # operator choices.
        precision_aliases = {
            "16": "16-mixed",
            "bf16": "bf16-mixed",
            "32": "32-true",
        }
        precision = str(self.precision or "16-mixed").strip().lower()
        self.precision = precision_aliases.get(precision, precision)
        if self.precision not in {"16-mixed", "bf16-mixed", "32-true"}:
            raise ValueError(
                "training.precision must be '16-mixed', 'bf16-mixed', or '32-true'"
            )
        if not (0.0 < self.auto_test_threshold < 1.0):
            raise ValueError("training.auto_test_threshold must be in (0, 1)")
        if self.bn_recalibrate_batches < 1:
            raise ValueError("training.bn_recalibrate_batches must be >= 1")
        self.transfer_head_init = str(self.transfer_head_init or "matching_classes").strip().lower()
        if self.transfer_head_init not in {"matching_classes", "random"}:
            raise ValueError(
                "training.transfer_head_init must be 'matching_classes' or 'random'"
            )
        head_map: dict[str, str] = {}
        for child, parent in (self.transfer_head_map or {}).items():
            child_name = str(child).strip()
            parent_name = str(parent).strip()
            if not child_name or not parent_name:
                raise ValueError(
                    "training.transfer_head_map entries need a non-empty class name "
                    "on both sides"
                )
            # A self-mapping is what exact matching already does; keeping it would
            # only make the resolved provenance lie about where the row came from.
            if child_name != parent_name:
                head_map[child_name] = parent_name
        self.transfer_head_map = head_map
        if self.calibrate_per_1000 < 0:
            raise ValueError("training.calibrate_per_1000 must be >= 0")
        if self.calibrate_negsrc not in {"venue", "library"}:
            raise ValueError("training.calibrate_negsrc must be 'venue' or 'library'")
        targets = [str(t).strip().lower() for t in (self.export_targets or []) if str(t).strip()]
        unknown = sorted(set(targets) - {"onnx", "fp16", "int8"})
        if unknown:
            raise ValueError(
                f"training.export_targets: unknown target(s) {unknown} "
                "(expected any of 'onnx', 'fp16', 'int8')"
            )
        # INT8 is a quantization of the FP16 IR, and any IR needs the ONNX it is
        # converted from — so asking for the deeper target implies the shallower
        # ones rather than failing.
        if "int8" in targets and "fp16" not in targets:
            targets.append("fp16")
        if targets and "onnx" not in targets:
            targets.append("onnx")
        self.export_targets = [t for t in ("onnx", "fp16", "int8") if t in targets]
        if self.auto_test_rules is not None:
            from sed.train.eval import parse_rule_id
            # Normalize through TestRule so a config can never carry a rule id
            # the evaluator would reject, and "fixed@0.9" becomes "fixed@0.90".
            seen: list[str] = []
            for raw in self.auto_test_rules:
                rule_id = parse_rule_id(str(raw)).id
                if rule_id not in seen:
                    seen.append(rule_id)
            self.auto_test_rules = seen
        balance_aliases = {
            "none": "off",
            "natural": "off",
            "moderate": "sqrt",
            "auto": "sqrt",
            "full": "balanced",
            "equal": "balanced",
            "class_group_sqrt": "balanced",
        }
        balance = str(self.class_balance or "off").strip().lower()
        self.class_balance = balance_aliases.get(balance, balance)
        if self.class_balance not in {"off", "sqrt", "balanced"}:
            raise ValueError(
                "training.class_balance must be 'off', 'sqrt', or 'balanced'"
            )
        depth = str(self.train_depth or "full").strip().lower()
        valid_depth = (
            depth in {"full", "all", "head", "head_only", "none", "last_block",
                      "last", "half"}
            or depth.startswith("last_n:") or depth.startswith("last_n=")
            or (depth.startswith("last") and depth[4:].isdigit())
        )
        if not valid_depth:
            raise ValueError(
                "training.train_depth must be one of: head, last_block, "
                "last_n:<k>, half, full"
            )
        self.train_depth = depth
        if not (0.0 < self.lr_layer_decay <= 1.0):
            raise ValueError("training.lr_layer_decay must be in (0, 1]")
        policy = str(self.fine_tune_policy or "manual").strip().lower()
        aliases = {
            "head_only": "head",
            "last_block": "last_stage",
            "last": "last_stage",
            "last_n:0": "head",
            "last_n:1": "last_stage",
            "last_n:2": "last_two",
            # Legacy UI option: keep loading old configs, but collapse it to a
            # named top-stage policy instead of preserving a vague "half" mode.
            "half": "last_two",
        }
        policy = policy.replace("last_n=", "last_n:")
        policy = aliases.get(policy, policy)
        # Parametric "train the last <k> encoder stages" (k>=3; the model-aware
        # depth control emits this — k<=2 fold into the named policies above).
        parametric = policy.startswith("last_n:") and policy[len("last_n:"):].isdigit()
        if policy not in {"manual", "head", "last_stage", "last_two", "full"} and not parametric:
            raise ValueError(
                "training.fine_tune_policy must be manual, head, last_stage, "
                "last_two, full, or last_n:<k>"
            )
        self.fine_tune_policy = policy
        if self.fine_tune_warmup_epochs < 0:
            raise ValueError("training.fine_tune_warmup_epochs must be >= 0")
        self.pretraining_objective = str(self.pretraining_objective or "supervised").strip().lower()
        if self.pretraining_objective not in {"supervised", "byol"}:
            raise ValueError("training.pretraining_objective must be supervised or byol")
        if not (0.0 < float(self.ssl_ema_decay) < 1.0):
            raise ValueError("training.ssl_ema_decay must be in (0, 1)")
        return self

    def resolved_auto_test_rules(self) -> list[str]:
        """Frozen-test rule ids to score when training finishes.

        `auto_test_rules` when it is set; otherwise the pre-rule keys say the
        same thing in the old spelling: no auto-test at all, the run's own
        (calibrated) decision layer, or one shared fixed confidence.
        """
        if self.auto_test_rules is not None:
            return list(self.auto_test_rules)
        if not self.auto_test:
            return []
        if self.calibrate_thresholds:
            return ["deployed"]
        return [f"fixed@{self.auto_test_threshold:.2f}"]


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
    data_library/mined/<class>/ as trainable mined library clips. Projects with
    `_negative` configured automatically get per-session mined source rules for
    hard-negative training; humans can still remove or reassign those rules in
    Inspect.
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


# Class names that read as a hard-negative bucket rather than a lighting cue, so
# a project that adds one without an explicit `[targets]` entry gets suppressor
# behaviour by default. Only a default — `[targets] <class> = { enabled = true }`
# always wins, and any other name can be made a suppressor the same way.
_SUPPRESSOR_CLASS_NAMES: frozenset[str] = frozenset({
    "hardneg", "hardnegs", "hard_neg", "hard_negs",
    "hard_negative", "hard_negatives", "hardnegative", "hardnegatives",
})


class TargetCfg(BaseModel):
    """One inference class → trigger / threshold / mining / ArtNet spec.

    `[targets]` is the single place that lists the classes inference cares about
    and how each behaves:
      - `enabled`   eligible to fire a trigger (and ArtNet, if configured).
                    A class with `enabled = false` is a *suppressor*: it stays a
                    model output and still competes for the window, so when it
                    wins the system abstains instead of firing the runner-up.
                    That is how a `hardnegs` class earns its keep — see
                    `Config.class_roles`.
      - `veto_max`  suppressor-only. When set, the window is discarded outright
                    if this class scores above it, exactly like
                    `inference.negative_max` does for `_negative`. Leave it unset
                    for a suppressor that should only compete.
      - `threshold` per-class decision threshold; falls back to legacy
                    `[thresholds].<class>` then `inference.default_threshold`
                    (for `_negative`, `inference.negative_max`).
      - `mine`      save detections of this class when mining is on (`--mine`
                    or `mining.enabled`). `_negative` may set `mine = true`.
      - `mine_threshold`  confidence a class must clear to be mined, independent
                    of `threshold` (the deployment/trigger decision). Falls back
                    to the legacy `mining.per_class_min_confidence` map, then to
                    `threshold`/`class_threshold` when unset — see
                    `Config.mine_threshold`. Lets a class be mined at a looser
                    (or tighter) bar than the one that fires it.
      - `role`      explicit override of `Config.class_roles`'s `target` /
                    `negative` / `suppressor` classification. Leave unset to use
                    the `enabled`-based default described there — `role` only
                    needs setting to make a class *other than* `_negative`
                    behave as a veto/background class (role = "negative"), e.g.
                    to split one large `_negative` pool into named buckets such
                    as `hardneg` / `cutlery` that each stay individually
                    trainable but still pool into the shared veto gate and
                    false-alarm metric.
      - `channel` / `value`  DMX channel + value sent on trigger.

    Example: 3 glass classes share channel 1 by using distinct values:
      clink_wine_thin   → channel=1, value=85
      clink_wine_thick  → channel=1, value=170
      clink_champagne   → channel=1, value=255
    The downstream lighting controller maps value → cue/scene.
    """
    enabled: bool = True
    threshold: float | None = None         # per-class decision threshold (see Config.class_threshold)
    veto_max: float | None = None          # suppressor-only hard veto (see class docstring)
    mine: bool = False                     # save mined clips of this class when mining is on
    mine_threshold: float | None = None    # per-class mining confidence (see Config.mine_threshold)
    role: Literal["target", "negative", "suppressor"] | None = None  # explicit role override
    channel: int = 1                       # 1-indexed DMX channel
    value: int = 255                       # DMX value 0-255 sent on trigger

    @model_validator(mode="after")
    def _check_target(self) -> "TargetCfg":
        if self.veto_max is not None and not (0.0 < float(self.veto_max) <= 1.0):
            raise ValueError("[targets] veto_max must be in (0, 1]")
        if self.veto_max is not None and self.enabled and self.role is None:
            raise ValueError(
                "[targets] veto_max applies to suppressor classes only "
                "(enabled = false); an enabled class cannot veto itself"
            )
        if self.veto_max is not None and self.role == "target":
            raise ValueError(
                "[targets] veto_max applies to negative/suppressor classes only "
                "(role != \"target\"); a target class cannot veto itself"
            )
        if self.mine_threshold is not None and not (0.0 < float(self.mine_threshold) <= 1.0):
            raise ValueError("[targets] mine_threshold must be in (0, 1]")
        return self


class ProjectMeta(BaseModel):
    id: str = "default"
    name: str = "Unnamed Project"
    created_at: str = ""
    description: str = ""
    kind: str = "venue"


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

    def _role_for(self, cls: str) -> str:
        """Resolve one class's role — the single source of truth `class_roles` fans out.

        Precedence, most specific first:
          1. Explicit `[targets].<cls>.role` — always wins.
          2. The literal name `_negative` → `"negative"`, unconditionally. This
             sits *above* the legacy `enabled` tier on purpose: a `[targets]`
             entry that exists only to set `mine = true` (leaving `enabled` at
             its default `True`) must not be reinterpreted as `role="target"`.
          3. A `[targets]` entry with no `role` — legacy `enabled`-bool tier:
             `"target"` if enabled else `"suppressor"`.
          4. No entry at all — a name that reads as a hard-negative bucket
             (`_SUPPRESSOR_CLASS_NAMES`) defaults to `"suppressor"`, anything
             else to `"target"`.

        Any class other than `_negative` can be given `role = "negative"`
        explicitly, which is what lets a project split one large `_negative`
        pool into named, individually-trainable buckets (`hardneg`, `cutlery`,
        `vosec`, …) that still all act as background for the veto gate, the
        loss's negative focusing, the sampler's background mass, and the
        deployment false-alarm rate.
        """
        name = str(cls)
        target = self.targets.get(name)
        if target is not None and target.role is not None:
            return target.role
        if name == "_negative":
            return "negative"
        if target is not None:
            return "target" if target.enabled else "suppressor"
        return "suppressor" if name.lower() in _SUPPRESSOR_CLASS_NAMES else "target"

    def class_threshold(self, cls: str) -> float:
        """Effective decision threshold for one class.

        Resolution order: per-class `[targets].<cls>.threshold` → (for any
        `role == "negative"` class, which includes `_negative` by default)
        `inference.negative_max` → legacy `[thresholds].<cls>` →
        `inference.default_threshold`. The legacy `[thresholds]` block is kept
        as a fallback so older configs and frozen model snapshots keep working.
        """
        target = self.targets.get(cls)
        if target is not None and target.threshold is not None:
            return float(target.threshold)
        if self._role_for(cls) == "negative":
            return float(self.inference.negative_max)
        if cls in self.thresholds:
            return float(self.thresholds[cls])
        return float(self.inference.default_threshold)

    def mine_threshold(self, cls: str) -> float:
        """Effective mining confidence for one class — independent of `class_threshold`.

        Resolution order: per-class `[targets].<cls>.mine_threshold` → legacy
        `mining.per_class_min_confidence[cls]` → `class_threshold(cls)` (today's
        implicit behaviour, unchanged when neither is set). Lets a class be
        mined at a different bar than the one that fires/vetoes it — e.g.
        capturing near-miss `_negative`/`hardneg` clips well below
        `negative_max` for human review, without lowering the live veto.
        """
        target = self.targets.get(cls)
        if target is not None and target.mine_threshold is not None:
            return float(target.mine_threshold)
        legacy = self.mining.per_class_min_confidence.get(cls)
        if legacy is not None:
            return float(legacy)
        return self.class_threshold(cls)

    def effective_thresholds(self, classes: list[str]) -> dict[str, float]:
        """`class_threshold` for every class, as a dict."""
        return {cls: self.class_threshold(cls) for cls in classes}

    def window_lead_in_s(self) -> float:
        """How far into the model window the triggering onset sits, in seconds.

        The recording preroll and the model window are different lengths:
        `onset.record_seconds` says how much audio to keep around a detected
        transient (deliberately generous, so one library serves projects with
        different windows), while `features.window_seconds` is what the model
        actually consumes. Reusing the recording preroll as the window lead-in
        only works while the two happen to be equal — at a 0.4 s window a 300 ms
        preroll would spend three quarters of the input on pre-onset background
        and truncate the decay of the very transient being classified.

        So the lead-in is the configured preroll, capped at 30% of the window.
        At the historical 1.0 s window with a 300 ms preroll this is exactly the
        old behaviour; at 0.4 s it yields a 120 ms lead-in and 280 ms of event.

        Both the live onset-gated runtime and the training crop read this, so
        the model is trained on the same window geometry it is scored on.
        """
        window = float(self.features.window_seconds)
        return min(float(self.onset.preroll_ms) / 1000.0, window * 0.30)

    def mining_classes(self) -> set[str]:
        """Classes flagged for mining in `[targets]` (may include `_negative`)."""
        return {cls for cls, target in self.targets.items() if target.mine}

    def class_roles(self, classes: list[str]) -> dict[str, str]:
        """What each model output does at decision time.

        ``"negative"``   — never competes for the window, only vetoes it
                           (`_negative` by default; any class can be given
                           `role = "negative"` explicitly — see `_role_for`).
        ``"target"``     — may fire a trigger.
        ``"suppressor"`` — a real output that never fires, but competes for the
                           window so that when it wins, nothing fires. This is
                           what makes a hard-negative class (cutlery, metal,
                           glass handling) reduce false positives instead of
                           being ignored.

        See `_role_for` for the full resolution order and precedence.
        """
        return {str(cls): self._role_for(str(cls)) for cls in classes}

    def class_vetoes(self, classes: list[str]) -> dict[str, float]:
        """Every negative/suppressor class that hard-vetoes a window, and at what score.

        A `role == "negative"` class (which includes `_negative` by default)
        vetoes at `inference.negative_max` unless it sets its own explicit
        `[targets] <class>.veto_max`. A `role == "suppressor"` class only
        vetoes if it sets `veto_max` explicitly — competing for the window is
        its default behaviour, vetoing is opt-in.
        """
        out: dict[str, float] = {}
        roles = self.class_roles(classes)
        for cls, role in roles.items():
            if role not in ("negative", "suppressor"):
                continue
            target = self.targets.get(cls)
            if target is not None and target.veto_max is not None:
                out[cls] = float(target.veto_max)
            elif role == "negative":
                out[cls] = float(self.inference.negative_max)
        return out

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
    def _sync_model_channels_to_feature(self) -> "Config":
        colormap = str(getattr(self.features, "colormap", "none") or "none").lower()
        self.model.channels = 3 if colormap not in ("", "none") else 1
        if str(getattr(self.features, "frontend", "efficientat")).lower() == "ced":
            if int(self.model.sample_rate) != 16000:
                raise ValueError("features.frontend='ced' requires model.sample_rate=16000")
        return self

    @model_validator(mode="after")
    def _check_fixed_model_feature_contract(self) -> "Config":
        validate_clink_cnn_v2_feature_contract(self)
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


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        # Model-recipe parameters are deliberately JSON-like, so keep them in
        # a normal TOML inline table instead of serialising opaque JSON strings.
        def _key(k: Any) -> str:
            text = str(k)
            return text if re.match(r"^[A-Za-z0-9_-]+$", text) else _format_toml_value(text)
        return "{ " + ", ".join(f"{_key(k)} = {_format_toml_value(v)}" for k, v in value.items()) + " }"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _set_toml_key(text: str, section: str, key: str, value: Any) -> str:
    """Set ``[section] key = value`` in TOML text, preserving comments/layout.

    Replaces an existing key inside its section, inserts it just under the
    section header if absent, or appends a new section at EOF.
    """
    new_line = f"{key} = {_format_toml_value(value)}"
    header_re = re.compile(rf"(?m)^\[{re.escape(section)}\][^\n]*$")
    hm = header_re.search(text)
    if not hm:
        prefix = text if (not text or text.endswith("\n")) else text + "\n"
        return f"{prefix}\n[{section}]\n{new_line}\n"
    body_start = text.find("\n", hm.end())
    if body_start == -1:
        return text + f"\n{new_line}\n"
    body_start += 1
    nm = re.compile(r"(?m)^\[[^\]]+\][^\n]*$").search(text, body_start)
    body_end = nm.start() if nm else len(text)
    body = text[body_start:body_end]
    key_re = re.compile(rf"(?m)^(\s*){re.escape(key)}\s*=.*$")
    if key_re.search(body):
        new_body = key_re.sub(lambda mm: f"{mm.group(1)}{new_line}", body, count=1)
    else:
        new_body = new_line + "\n" + body
    return text[:body_start] + new_body + text[body_end:]


def _remove_toml_key(text: str, section: str, key: str) -> str:
    """Remove ``key`` from ``[section]`` in TOML text, if present.

    Mirrors `_set_toml_key`'s section/body location logic exactly, so a key
    written by one is reliably found and dropped by the other. A missing
    section or key is a no-op, not an error — deleting something already gone
    is the common case on a rename/remove round-trip.
    """
    header_re = re.compile(rf"(?m)^\[{re.escape(section)}\][^\n]*$")
    hm = header_re.search(text)
    if not hm:
        return text
    body_start = text.find("\n", hm.end())
    if body_start == -1:
        return text
    body_start += 1
    nm = re.compile(r"(?m)^\[[^\]]+\][^\n]*$").search(text, body_start)
    body_end = nm.start() if nm else len(text)
    body = text[body_start:body_end]
    key_re = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*\n?")
    new_body = key_re.sub("", body, count=1)
    return text[:body_start] + new_body + text[body_end:]


def update_config_toml_values(
    path: Path,
    updates: dict[str, dict[str, Any]],
    *,
    removals: dict[str, list[str]] | None = None,
) -> None:
    """Write nested ``{section: {key: value}}`` into a project config.toml.

    Hand-edited comments and unrelated keys are preserved (regex surgery rather
    than a full re-dump), mirroring ``registry.set_active_model``. ``removals``
    (``{section: [key, ...]}``) deletes keys instead — applied first, so a
    rename that both drops an old key and sets a new one in the same section
    cannot have the deletion undo the write.
    """
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for section, keys in (removals or {}).items():
        for key in keys:
            text = _remove_toml_key(text, section, key)
    for section, kv in updates.items():
        for key, value in kv.items():
            text = _set_toml_key(text, section, key, value)
    if not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def resolved_features(features: "FeaturesCfg") -> "FeaturesCfg":
    """Re-run feature validation so the two axes reflect any in-place edits.

    ``compression`` and the ``scaling_*`` fields are settled by the validator
    from whatever spelling a config arrived in. A caller that mutates a live
    Config (``cfg.features.pcen = True``, ``cfg.features.feature_type =
    "linear"``) bypasses that, leaving the canonical fields describing the
    previous feature. Anything that reads them to make a decision — the training
    launch gate, the SpecAugment fill, the manifest record — resolves through
    here first. Falls back to the input if the mutated combination is invalid,
    so a diagnostic path never turns into a crash.
    """
    try:
        return FeaturesCfg.model_validate(features.model_dump())
    except Exception:  # noqa: BLE001
        return features


def effective_split_test_enabled(cfg: "Config") -> bool:
    """Effective unified split-test toggle: off unless explicitly configured."""
    configured = getattr(cfg.dataset, "split_test_enabled", None)
    if configured is not None:
        return bool(configured)
    return False


def effective_split_test_fraction(cfg: "Config") -> float:
    """Effective unified split-test fraction, including legacy aliases."""
    configured = getattr(cfg.dataset, "split_test_fraction", None)
    if configured is not None:
        return float(configured)
    return float(getattr(cfg.dataset, "pretraining_test_fraction", 0.1) or 0.0)


def project_config_template(
    project_id: str,
    name: str = "",
    created_at: str = "",
    kind: str = "",
) -> str:
    """Return the clean, prefilled config.toml text for a new project."""
    safe_name = name or project_id
    kind_line = f'kind = "{kind}"\n' if kind else ""
    return f'''# SoundEventDetection project config.
# Edit project-specific values here; omitted keys use defaults from sed.config.
# Dataset UI: uv run sed dataset inspect --project {project_id} --serve
# Inspect state: projects/{project_id}/dataset/dataset_selection.json

[project]
id = "{project_id}"
name = "{safe_name}"
created_at = "{created_at}"
description = ""
{kind_line}

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
record_seconds = 0.4 # saved clip length; keep >= features.window_seconds
preroll_ms = 120 # lead-in when recording; capped at 30% of the model window
recording_cooldown_ms = 600 # global delay between accepted recordings
randomize_preroll = true
preroll_ms_range = [60, 240]
multi_mic_trigger = "any" # any (save fired) | any_all (save all) | mix (downmix) | ch0 (mic 0)
cross_channel_merge_ms = 500 # for "any"/"any_all": merge mic onsets into one clip

[model]
backbone = "mn10_as" # mn01 | mn10_as | mn20_as | mn40_as
pretrained = true
sample_rate = 32000

[features]
frontend = "efficientat" # efficientat | ced
# Model window: 0.4 s keeps the ~0.2 s glass transient dominant rather than
# diluting it in background. Clips are cropped around the onset, not centred.
window_seconds = 0.4

[training]
epochs = 60
loss = "bce_logits" # bce_logits | cross_entropy/categorical_crossentropy | asymmetric_focal
class_balance = "off" # off | sqrt (moderate) | balanced (full)
specaugment_enabled = false
precision = "16-mixed"
auto_export = true
# Frozen-test rules scored after training — same ids Batch evaluate writes.
auto_test_rules = ["fixed@0.90"]
[inference]
# Every key in this block is live on the edge device: decision/calibration
# parameters set here override the deployed model bundle (production tuning
# happens on site). Keys you leave out keep the bundle's trained/calibrated
# values. Feature/model geometry (hop_ms, window, mels) stays frozen with the
# bundle and cannot be changed here.
inference_channels = [0]
multi_mic_mode = "max" # max | best | mean | single
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
    kind: str = "",
) -> Path:
    """Create a prefilled projects/<id>/config.toml."""
    if layout.config_toml.exists() and not overwrite:
        return layout.config_toml
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.config_toml.write_text(
        project_config_template(layout.id, name=name, created_at=created_at, kind=kind),
        encoding="utf-8",
    )
    return layout.config_toml


def is_pretraining_project(cfg: Config) -> bool:
    """Whether a config is marked as a representation-learning project."""
    return str(getattr(getattr(cfg, "project", None), "kind", "venue") or "venue").lower() == "pretraining"


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
# Honored when present in older runtime_settings.json files; new files carry
# only targets/thresholds — decision knobs are tuned live on the edge instead
# (see _EDGE_TUNABLE_INFERENCE_KEYS).
_RUNTIME_INFERENCE_KEYS = {
    "hop_ms",
    "default_threshold",
    "consecutive_frames",
    "min_dbfs",
}

# Inference decision knobs the operator may tune in the edge config.toml after
# deployment (production calibration happens on site). Only keys explicitly
# present in the edge file override the bundle. hop_ms is NOT here: window
# framing is part of the frozen model contract.
_EDGE_TUNABLE_INFERENCE_KEYS = (
    "min_dbfs",
    "default_threshold",
    "consecutive_frames",
    "negative_max",
    "margin_to_second_best",
    "smoothing_enabled",
    "smoothing_median_window",
    "smoothing_ema_alpha",
    "hysteresis_low",
    "trigger_refractory_ms",
)


def runtime_settings_from_config(cfg: Config) -> dict[str, Any]:
    """Extract per-model deployment knobs (calibrated thresholds + targets).

    Site-local operator settings are NOT here: [deploy], [audio], [artnet],
    [logging], [mining], and the tunable [inference] decision keys are read
    live from the edge config.toml, so they never get frozen into a bundle.
    """
    data = cfg.model_dump(mode="json")
    return {block: data[block] for block in _RUNTIME_BLOCKS if block in data}


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
    """Whether this config actually loads an EfficientAT AudioSet checkpoint.

    Random (scratch) and parent initializations never load the upstream
    checkpoint, so no AudioSet feature contract applies to them. Covers both
    the ``mn*`` and ``dymn*`` families.
    """
    return (
        bool(cfg.model.pretrained)
        and str(cfg.model.initialization or "upstream") == "upstream"
        and str(cfg.model.backbone).startswith(("mn", "dymn"))
    )


def pretrained_feature_mismatches(cfg: Config) -> list[str]:
    """Return config fields that violate the selected pretrained checkpoint.

    Covers the numeric contract plus the normalization: a linear feature or PCEN
    produces a value distribution the AudioSet weights never saw, so both are
    reported as mismatches alongside any off-shape parameter.
    """
    if not (
        bool(cfg.model.pretrained)
        and str(cfg.model.initialization or "upstream") == "upstream"
    ):
        return []
    from sed.audio.features import feature_channels

    actual = {
        "features.frontend": cfg.features.frontend,
        "model.sample_rate": cfg.model.sample_rate,
        "model.channels": feature_channels(cfg.features),
        "features.n_fft": cfg.features.n_fft,
        "features.win_length": cfg.features.win_length,
        "features.hop_length": cfg.features.hop_length,
        "features.n_mels": cfg.features.n_mels,
        "features.fmin": cfg.features.fmin,
        "features.fmax": cfg.features.fmax,
    }
    # Prefer the contract baked into the selected checkpoint: hop/mels variants
    # (mn10_as_hop_5, mn10_as_mels_40, …) were trained at those parameters, so
    # matching them must not warn. Lazy import — model_lab imports this module.
    contract: dict[str, Any] | None = None
    try:
        from sed.train.model_lab import expected_contract_for_backbone

        contract = expected_contract_for_backbone(cfg.model.backbone)
    except Exception:  # noqa: BLE001
        contract = None
    if contract is None:
        if is_efficientat_pretrained_config(cfg):
            contract = dict(EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT)
        else:
            return []
    contract = dict(contract)
    # window_seconds is a deployment choice, not a weight contract: the encoder
    # is fully convolutional and AudioSet training itself used 10 s crops, so a
    # short detection window transfers fine and must not push users to scratch.
    contract.pop("features.window_seconds", None)
    mismatches = [
        f"{key}={actual[key]!r} (expected {expected!r})"
        for key, expected in contract.items()
        if key in actual and actual[key] != expected
    ]
    if contract.get("features.feature_type") == "mel" and cfg.features.feature_type != "mel":
        mismatches.append(
            f"features.feature_type={cfg.features.feature_type!r} (expected 'mel')"
        )
    if cfg.features.pcen:
        mismatches.append("features.pcen=True (pretrained weights expect their published normalization)")
    return mismatches


def pretrained_contract_warning(cfg: Config) -> str:
    """Human-readable warning when features are incompatible with the pretrained weights.

    Returns an empty string when on-contract (or not pretrained). The Model tab
    surfaces this text; it suggests scratch initialization but never forces it.
    """
    mismatches = pretrained_feature_mismatches(cfg)
    if not mismatches:
        return ""
    family = (
        "AudioSet-pretrained EfficientAT weights"
        if str(cfg.model.backbone).startswith(("mn", "dymn"))
        else f"pretrained {cfg.model.backbone} weights"
    )
    return (
        f"This feature is incompatible with the {family}: " + "; ".join(mismatches)
        + ". The weights were trained on their published feature contract, so "
        "they may transfer poorly. Set the model initialization to scratch "
        "(random) to train this feature from scratch."
    )


def validate_pretrained_feature_contract(
    cfg: Config,
    *,
    phase: str = "configuration",
) -> None:
    """Warn (never block) when a pretrained model is paired with an incompatible feature.

    Training never blocks on the feature design — the feature the operator builds
    in the lab is the model's ground-truth input. With an AudioSet-pretrained
    backbone the original mel contract is strongly recommended, but the operator
    may intentionally diverge (linear / PCEN / off-shape) and train from scratch
    via scratch (random) initialization, so this only logs guidance.
    """
    warning = pretrained_contract_warning(cfg)
    if warning:
        log.warning("[%s] %s", phase, warning)


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


def load_config_for_inference_model(project_id: str, model_id: str = "") -> Config:
    """Load one model bundle's config plus generated runtime settings.

    When ``model_id`` is empty, the active model subfolder comes from
    [deploy] active_model in the edge config.toml. When it is set, that model
    bundle is used without writing config.toml, so callers can test an exported
    model before selecting it for deployment.

    Model-correctness settings (feature geometry, hop_ms, classes) come from
    that frozen bundle. Machine-local settings ([audio] sound card, [deploy],
    [inference] topology, [artnet], [logging], [mining]) come from the edge
    config.toml, and any [inference] decision key or [targets] entry the
    operator sets there overrides the bundle — production calibration happens
    on the deployed device.
    """
    layout = ProjectLayout(project_id)
    try:
        edge = load_config(project_id)
    except FileNotFoundError:
        edge = Config()
    selected_model = str(model_id or edge.deploy.active_model or "")
    model_dir = resolve_active_model_dir(layout, selected_model)
    cfg = load_config_snapshot(model_dir / "training_config.json")
    cfg = apply_runtime_settings(
        cfg, load_runtime_settings(model_dir / "runtime_settings.json")
    )
    validate_pretrained_feature_contract(cfg, phase="inference")
    # Machine-local/operator settings come from the edge config.toml, never the
    # bundle: [audio] (sound card), [deploy], [artnet], [logging], [mining], and
    # [inference] topology (inference_channels / multi_mic_mode /
    # onset_gating_enabled). Model shape ([model], [features], hop_ms) stays
    # frozen with the bundle.
    cfg.artnet = edge.artnet.model_copy(deep=True)
    cfg.deploy = edge.deploy.model_copy(deep=True)
    cfg.deploy.active_model = model_dir.name
    cfg.audio = edge.audio.model_copy(deep=True)
    cfg.inference.inference_channels = list(edge.inference.inference_channels)
    cfg.inference.multi_mic_mode = edge.inference.multi_mic_mode
    cfg.inference.onset_gating_enabled = edge.inference.onset_gating_enabled
    cfg.logging = edge.logging.model_copy(deep=True)
    cfg.mining = edge.mining.model_copy(deep=True)
    # Decision/calibration parameters are edge-tunable: production calibration
    # happens on the deployed device. Only keys the operator explicitly set in
    # the edge config.toml override the bundle — absent keys keep the
    # trained/calibrated values (the template lists the common ones).
    edge_inference_raw = _read_toml(layout.config_toml).get("inference", {})
    for key in _EDGE_TUNABLE_INFERENCE_KEYS:
        if key in edge_inference_raw:
            setattr(cfg.inference, key, getattr(edge.inference, key))
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


def load_config_for_inference(project_id: str) -> Config:
    """Load the active model's config plus generated runtime settings."""
    return load_config_for_inference_model(project_id)
