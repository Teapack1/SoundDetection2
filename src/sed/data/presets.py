"""Global audio-feature preset store.

Feature presets are small JSON files under ``presets/audio_features/<name>.json``
at the repo root, shared across projects. A preset captures everything the
Audio Features lab needs to reproduce a spectrogram view and (for mel) a
trainable feature config: the feature type, the STFT/mel parameters, the model
feature rate, and an optional backbone/pretrained suggestion.

Applying a preset writes its parameters into a project's ``config.toml``
(``[features]`` plus ``[model]`` sample_rate/backbone/pretrained when set) so the
choice flows into training, the frozen model bundle, deployment, and inference.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging

from sed import config as config_mod

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Fields that live on FeaturesCfg and are applied to a project's [features].
_FEATURES_CFG_FIELDS: tuple[str, ...] = (
    "feature_type",
    "frontend",
    "n_fft",
    "win_length",
    "hop_length",
    "n_mels",
    "fmin",
    "fmax",
    "window_seconds",
    "log_power",
    "stft_engine",
    "db_norm",
    "db_min",
    "db_max",
    "colormap",
    "image_size",
    "pcen",
    "pcen_time_constant",
    "pcen_gain",
    "pcen_bias",
    "pcen_power",
    "pcen_eps",
    "standardize",
    "standardize_mean",
    "standardize_std",
    "scaling_method",
    "scaling_scope",
    "scaling_stats",
    "standardize_stats_resolved",
    "standardize_stats_dataset_sha256",
    "standardize_stats_split_seed",
    "standardize_stats_val_fraction",
    "standardize_stats_clips",
)
_PCEN_FLOAT_FIELDS: tuple[str, ...] = (
    "pcen_time_constant", "pcen_gain", "pcen_bias", "pcen_power", "pcen_eps",
)
# sample_rate is the model feature rate ([model].sample_rate), carried alongside
# the feature fields so a preset fully describes the spectrogram view.
_FEATURE_FIELDS: tuple[str, ...] = _FEATURES_CFG_FIELDS + ("sample_rate",)


def _presets_dir() -> Path:
    return config_mod.FEATURE_PRESETS_DIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_preset_name(name: str) -> str:
    """Filesystem-safe slug for a preset name (lowercase kebab-case)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return slug or "preset"


def feature_preset_path(name: str) -> Path:
    return _presets_dir() / f"{slugify_preset_name(name)}.json"


# ---------------------------------------------------------------------------
# Built-in seeds
# ---------------------------------------------------------------------------

_BUILTIN_PRESETS: list[dict[str, Any]] = [
    {
        "name": "geometry-a-efficient-short",
        "label": "Geometry A — efficient short log-mel",
        "description": "48 kHz, 0.4 s, 128 mel, 5 ms hop. A compact starting point for BC-ResNet, "
                       "TC-ResNet, and DS-CNN. fmin 150 Hz keeps all 128 mel filters non-empty at "
                       "n_fft 1024 (fmin 0 left the bottom band black).",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 48000,
        "n_fft": 1024, "win_length": 960, "hop_length": 240, "n_mels": 128,
        "fmin": 150, "fmax": 22000, "window_seconds": 0.4, "backbone": "kws-bcresnet",
        "pretrained": False,
    },
    {
        "name": "clink-v2-native",
        "label": "ClinkCNN-v2 native (44.1 kHz glass)",
        "description": "44.1 kHz, exactly 0.4 s, 128 Kaldi/HTK mel bands and 81 frames. "
                       "The 20 ms window resolves glass resonances, the 5 ms hop preserves "
                       "attack/decay, and the 80-20,000 Hz range rejects rumble and the least "
                       "useful ultrasonic edge while retaining bright glass harmonics.",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 1024, "win_length": 882, "hop_length": 220, "n_mels": 128,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4, "backbone": "clink-cnn-v2",
        "pcen": False, "standardize": False,
        "pretrained": False,
    },
    {
        "name": "clink-v2-dense-mel256",
        "label": "Experimental — ClinkCNN-v2 dense mel256",
        "description": "44.1 kHz, 0.4 s, 256 Kaldi/HTK mel bands and 81 frames. "
                       "FFT 2048 keeps every filter non-empty while the native 20 ms window "
                       "and 5 ms hop isolate mel-band density from physical window resolution.",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 2048, "win_length": 882, "hop_length": 220, "n_mels": 256,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4, "backbone": "clink-cnn-v2",
        "pcen": False, "standardize": False, "pretrained": False,
    },
    {
        "name": "clink-v2-resolved-mel256",
        "label": "Experimental — ClinkCNN-v2 resolved mel256",
        "description": "44.1 kHz, 0.4 s, 256 Kaldi/HTK mel bands and 81 frames. "
                       "FFT 2048 plus a 40 ms window tests genuine resonance resolution "
                       "while retaining the native 5 ms frame spacing.",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 2048, "win_length": 1764, "hop_length": 220, "n_mels": 256,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4, "backbone": "clink-cnn-v2",
        "pcen": False, "standardize": False, "pretrained": False,
    },
    {
        "name": "clink-v2-temporal-mel128",
        "label": "Experimental — ClinkCNN-v2 temporal mel128",
        "description": "44.1 kHz, 0.4 s, native 128-mel/20 ms spectral analysis with "
                       "a 2.5 ms hop and 161 frames to isolate attack/decay time resolution.",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 1024, "win_length": 882, "hop_length": 110, "n_mels": 128,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4, "backbone": "clink-cnn-v2",
        "pcen": False, "standardize": False, "pretrained": False,
    },
    {
        "name": "clink-v2-linear-fft1024",
        "label": "Experimental — ClinkCNN-v2 linear FFT1024",
        "description": "44.1 kHz, 0.4 s, 463 standardized linear log-power bins and "
                       "81 frames. Uses the native 20 ms window/5 ms hop to isolate the "
                       "linear basis from mel compression; estimate train-only statistics first.",
        "feature_type": "linear", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 1024, "win_length": 882, "hop_length": 220, "n_mels": 128,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4, "backbone": "clink-cnn-v2",
        "log_power": True, "stft_engine": "librosa", "pcen": False, "db_norm": False,
        "colormap": "none", "standardize": True,
        "standardize_stats_resolved": False, "pretrained": False,
    },
    {
        "name": "clink-v2-linear-fft2048",
        "label": "Experimental — ClinkCNN-v2 linear FFT2048",
        "description": "44.1 kHz, 0.4 s, 925 standardized linear log-power bins and "
                       "81 frames. FFT 2048 plus a 40 ms window is the accuracy-first "
                       "high-resolution research contract; estimate train-only statistics first.",
        "feature_type": "linear", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 2048, "win_length": 1764, "hop_length": 220, "n_mels": 128,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4, "backbone": "clink-cnn-v2",
        "log_power": True, "stft_engine": "librosa", "pcen": False, "db_norm": False,
        "colormap": "none", "standardize": True,
        "standardize_stats_resolved": False, "pretrained": False,
    },
    {
        "name": "geometry-b-glass-highres",
        "label": "Geometry B — high-resolution glass mode",
        "description": "48 kHz, 0.4 s, 192 mel, 2.5 ms hop for high-frequency transient detail. "
                       "192 mels + fmin 50 Hz keep every mel filter non-empty at n_fft 2048 "
                       "(256 mels left low bands black).",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 48000,
        "n_fft": 2048, "win_length": 1024, "hop_length": 120, "n_mels": 192,
        "fmin": 50, "fmax": 23000, "window_seconds": 0.4, "backbone": "kws-bcresnet",
        "pretrained": False,
    },
    {
        "name": "geometry-c-square-image",
        "label": "Geometry C — square image baseline",
        "description": "48 kHz, 0.4 s, 224 mel and ~224 frames for timm ConvNeXt/MobileNet image-model "
                       "baselines. n_fft 2048 + fmin 50 Hz keep all 224 mel filters non-empty "
                       "(the old n_fft 1024 left 15 low bands black).",
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 48000,
        "n_fft": 2048, "win_length": 1024, "hop_length": 86, "n_mels": 224,
        "fmin": 50, "fmax": 23000, "window_seconds": 0.4, "backbone": "convnext_tiny",
        "pretrained": False,
    },
    {
        "name": "linear-demo",
        "label": "Linear spectrogram (preview only)",
        "description": (
            "Full-resolution linear STFT log-power view at 32 kHz. Experimental — "
            "for inspecting high-frequency detail; needs a from-scratch backbone "
            "before it can train."
        ),
        "feature_type": "linear",
        "frontend": "efficientat",
        "sample_rate": 32000,
        "n_fft": 1024,
        "win_length": 800,
        "hop_length": 320,
        "n_mels": 128,
        "fmin": 0,
        "fmax": 16000,
        "window_seconds": 1.0,
        "log_power": True,
        "backbone": "mn10_as",
        "pretrained": False,
    },
    {
        "name": "legacy-sounddetection",
        "label": "Legacy soundDetection (linear dB image)",
        "description": (
            "Exact legacy-system feature: scipy PSD spectrogram at 44.1 kHz "
            "(Tukey 0.25, nperseg 190 → 96 linear bins), 10·log10 dB normalized "
            "from -20..60 dB to 0–1, 0.4 s window. Pair with the SmallerVGGNet "
            "(legacy port) model and cross-entropy loss to reproduce the legacy "
            "workflow."
        ),
        "feature_type": "linear",
        "frontend": "efficientat",
        "sample_rate": 44100,
        "n_fft": 190,
        "win_length": 190,
        "hop_length": 167,
        "n_mels": 128,
        "fmin": 0,
        "fmax": 22050,
        "window_seconds": 0.4,
        "log_power": True,
        "stft_engine": "scipy",
        "db_norm": True,
        "db_min": -20.0,
        "db_max": 60.0,
        "colormap": "jet",
        "image_size": 96,
        "backbone": "legacy-smallervggnet",
        "pretrained": False,
    },
    {
        "name": "pcen-mel",
        "label": "PCEN mel (from-scratch)",
        "description": (
            "128-band mel at 32 kHz normalized with PCEN (automatic gain control) "
            "instead of log+fast-norm. Robust to loudness; incompatible with the "
            "AudioSet weights, so it trains from scratch (pretrained off)."
        ),
        "feature_type": "mel",
        "frontend": "efficientat",
        "sample_rate": 32000,
        "n_fft": 1024,
        "win_length": 800,
        "hop_length": 320,
        "n_mels": 128,
        "fmin": 0,
        "fmax": 15000,
        "window_seconds": 1.0,
        "log_power": True,
        "pcen": True,
        "backbone": "mn10_as",
        "pretrained": False,
    },
    {
        "name": "ced-context-assisted-1s",
        "label": "CED context-assisted (1.0 s)",
        "description": (
            "CED variable-length evaluation contract: the prediction is assigned to the "
            "center 0.4 s decision region, with 0.3 s acoustic context on each side. "
            "Training clips must be positive only when the annotated event intersects "
            "that center region."
        ),
        "feature_type": "mel",
        "frontend": "ced",
        "sample_rate": 16000,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "fmin": 0,
        "fmax": 8000,
        "window_seconds": 1.0,
        "log_power": True,
        "backbone": "ced-mini",
        "pretrained": True,
    },
]


# Built-ins retired in favor of the virtual model-contract entry the UI now
# always offers ("Model default (unsaved)"). Each maps to its FINAL shipped
# definition: an on-disk copy is deleted only while it still matches that
# content byte-for-byte (modulo timestamps) — a user-modified copy survives as
# an ordinary preset.
_RETIRED_BUILTINS: dict[str, dict[str, Any]] = {
    "efficientat-standard": {
        "name": "efficientat-standard",
        "label": "EfficientAT Standard (mel, pretrained)",
        "description": (
            "128-band log-mel at 32 kHz — the AudioSet-pretrained EfficientAT "
            "contract. Trainable with pretrained backbones (mn10_as/mn20_as/mn40_as)."
        ),
        "feature_type": "mel",
        "frontend": "efficientat",
        "sample_rate": 32000,
        "n_fft": 1024,
        "win_length": 800,
        "hop_length": 320,
        "n_mels": 128,
        "fmin": 0,
        "fmax": 15000,
        "window_seconds": 1.0,
        "log_power": True,
        "backbone": "mn10_as",
        "pretrained": True,
    },
    "ced-tiny-contract": {
        "name": "ced-tiny-contract",
        "label": "CED Tiny Contract (mel, pretrained)",
        "description": (
            "CED-Tiny frontend contract: 16 kHz, 64-band dB mel, n_fft=512, "
            "win=512, hop=160, 0.4 s window. Pair with the CED Tiny model."
        ),
        "feature_type": "mel",
        "frontend": "ced",
        "sample_rate": 16000,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "fmin": 0,
        "fmax": 8000,
        "window_seconds": 0.4,
        "log_power": True,
        "backbone": "ced-tiny",
        "pretrained": True,
    },
}


def _normalize_preset(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw payload into the canonical preset dict (with defaults)."""
    from sed.config import FeaturesCfg, ModelCfg

    feat_defaults = FeaturesCfg()
    sample_rate_default = ModelCfg().sample_rate
    raw = dict(payload or {})
    name = slugify_preset_name(raw.get("name") or raw.get("label") or "preset")
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "label": str(raw.get("label") or name),
        "description": str(raw.get("description") or ""),
    }
    # FeaturesCfg fields, falling back to FeaturesCfg defaults.
    for field in _FEATURES_CFG_FIELDS:
        out[field] = raw.get(field, getattr(feat_defaults, field))
    out["sample_rate"] = raw.get("sample_rate", sample_rate_default)
    # Optional model suggestion. Off-contract features (linear / PCEN) simply ship
    # with pretrained=false so a from-scratch backbone trains on them; the Model
    # tab is the single source of truth for initialization.
    out["backbone"] = str(raw.get("backbone") or "mn10_as")
    out["pretrained"] = bool(raw.get("pretrained", True))
    # Type coercion to match FeaturesCfg so a round-trip validates cleanly.
    out["feature_type"] = str(out["feature_type"] or "mel").lower()
    out["frontend"] = str(out["frontend"] or "efficientat").lower()
    for int_field in ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels", "fmin", "fmax"):
        out[int_field] = int(out[int_field])
    out["window_seconds"] = float(out["window_seconds"])
    out["log_power"] = bool(out["log_power"])
    out["stft_engine"] = str(out.get("stft_engine") or "librosa").lower()
    out["db_norm"] = bool(out.get("db_norm", False))
    out["db_min"] = float(out.get("db_min", -20.0))
    out["db_max"] = float(out.get("db_max", 60.0))
    out["colormap"] = str(out.get("colormap") or "none").lower()
    out["image_size"] = int(out.get("image_size", 96))
    out["pcen"] = bool(out.get("pcen", False))
    for float_field in _PCEN_FLOAT_FIELDS:
        out[float_field] = float(out[float_field])
    out["standardize"] = bool(out.get("standardize", False))
    out["standardize_mean"] = float(out.get("standardize_mean", 0.0))
    out["standardize_std"] = float(out.get("standardize_std", 1.0))
    out["scaling_method"] = str(out.get("scaling_method") or "none").lower()
    out["scaling_scope"] = str(out.get("scaling_scope") or "global").lower()
    raw_scaling_stats = out.get("scaling_stats")
    out["scaling_stats"] = {
        str(k): [float(v) for v in (vals or [])]
        for k, vals in raw_scaling_stats.items()
    } if isinstance(raw_scaling_stats, dict) else {}
    # Legacy presets spell the fitted global z-score as standardize=True; keep
    # the canonical fields coherent so preset comparisons don't drift from the
    # validated FeaturesCfg they will produce.
    if out["standardize"] and out["scaling_method"] == "none":
        out["scaling_method"] = "zscore"
        out["scaling_scope"] = "global"
        if not out["scaling_stats"]:
            out["scaling_stats"] = {
                "mean": [out["standardize_mean"]], "std": [out["standardize_std"]],
            }
    out["standardize_stats_resolved"] = bool(
        out.get("standardize_stats_resolved", True)
    )
    out["standardize_stats_dataset_sha256"] = str(
        out.get("standardize_stats_dataset_sha256", "") or ""
    )
    out["standardize_stats_split_seed"] = int(
        out.get("standardize_stats_split_seed", 0) or 0
    )
    out["standardize_stats_val_fraction"] = float(
        out.get("standardize_stats_val_fraction", 0.0) or 0.0
    )
    out["standardize_stats_clips"] = int(
        out.get("standardize_stats_clips", 0) or 0
    )
    out["created_at"] = str(raw.get("created_at") or _now_iso())
    out["updated_at"] = _now_iso()
    return out


# Built-ins that existed when the marker file was still the bare "1" sentinel.
# A v1 marker means exactly these were seeded; later additions to
# _BUILTIN_PRESETS still get seeded once on such installs.
_SEED_V1_NAMES: frozenset[str] = frozenset({
    "geometry-a-efficient-short", "geometry-b-glass-highres",
    "geometry-c-square-image", "efficientat-standard", "linear-demo", "pcen-mel",
})


def _builtin_content_fields(preset: dict[str, Any]) -> dict[str, Any]:
    """Preset content minus volatile bookkeeping — for drift comparison."""
    return {k: v for k, v in preset.items() if k not in ("created_at", "updated_at")}


def _seed_builtin_presets() -> None:
    """Keep the built-in presets current on every install.

    The marker file records *which* built-ins have been seeded (one name per
    line), so a built-in the user deletes stays deleted while newly shipped
    built-ins still arrive on existing installs (a legacy "1" marker is read as
    the v1 built-in set). A built-in that is still present but whose shipped
    definition has changed (e.g. gaining new fields) is refreshed in place, so
    upgrades reach existing installs without resurrecting deleted presets.
    """
    directory = _presets_dir()
    marker = directory / ".builtins_seeded"
    seeded: set[str] = set()
    if marker.exists():
        text = marker.read_text(encoding="utf-8").strip()
        seeded = set(_SEED_V1_NAMES) if text == "1" else {
            line.strip() for line in text.splitlines() if line.strip()
        }
    directory.mkdir(parents=True, exist_ok=True)
    changed = False
    for preset in _BUILTIN_PRESETS:
        path = feature_preset_path(preset["name"])
        shipped = _normalize_preset(preset)
        if preset["name"] not in seeded:
            # Never seeded here (new built-in / v1 upgrade): create if absent.
            if not path.exists():
                _write_preset_file(path, shipped)
                changed = True
        elif path.exists():
            # Already seeded and not user-deleted: refresh if the shipped
            # definition drifted from what is on disk.
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — unreadable ⇒ rewrite it
                current = None
            if current is None or _builtin_content_fields(current) != _builtin_content_fields(shipped):
                _write_preset_file(path, shipped)
                changed = True
    # Retired built-ins: delete only pristine copies (content still equal to the
    # last shipped definition); user-modified copies live on as ordinary presets.
    for name, final_shipped in _RETIRED_BUILTINS.items():
        path = feature_preset_path(name)
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — unreadable retired file: leave it alone
                current = None
            if isinstance(current, dict):
                current.pop("align_model", None)  # retired key in legacy seeds
                if _builtin_content_fields(current) == _builtin_content_fields(_normalize_preset(final_shipped)):
                    path.unlink()
                    changed = True
        seeded.discard(name)
    all_names = (seeded | {p["name"] for p in _BUILTIN_PRESETS}) - set(_RETIRED_BUILTINS)
    if changed or all_names != seeded or not marker.exists():
        marker.write_text("\n".join(sorted(all_names)) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def _write_preset_file(path: Path, preset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(preset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def list_feature_presets() -> list[dict[str, Any]]:
    """All saved presets (seeding the built-ins on first call), sorted by label."""
    _seed_builtin_presets()
    presets: list[dict[str, Any]] = []
    for path in sorted(_presets_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("align_model", None)  # drop the retired key from legacy seeds
            presets.append(data)
        except Exception:  # noqa: BLE001 — skip a corrupt preset rather than fail the list
            continue
    presets.sort(key=lambda p: str(p.get("label") or p.get("name") or "").lower())
    return presets


def builtin_preset(name: str) -> dict[str, Any] | None:
    """The shipped definition of a built-in preset, independent of what is
    seeded (or user-edited) on disk. Used for architecture default-geometry
    lookups that must not drift with user preset edits."""
    for preset in _BUILTIN_PRESETS:
        if preset.get("name") == name:
            return dict(preset)
    return None


def load_feature_preset(name: str) -> dict[str, Any]:
    # Seed built-ins on a fresh install so a referenced default is loadable even
    # before the list endpoint runs (does not resurrect individually-deleted ones).
    if not _presets_dir().exists():
        _seed_builtin_presets()
    path = feature_preset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"feature preset '{name}' not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("align_model", None)  # drop the retired key from legacy seeds
    return data


def save_feature_preset(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate + persist a preset, returning the normalized record."""
    merged = dict(payload or {})
    merged.setdefault("name", name)
    preset = _normalize_preset(merged)
    # Round-trip the feature fields through FeaturesCfg so a bad combination is
    # rejected before it lands on disk.
    apply_preset_to_features_cfg(config_mod.FeaturesCfg(), preset)
    _write_preset_file(feature_preset_path(preset["name"]), preset)
    return preset


def delete_feature_preset(name: str) -> bool:
    path = feature_preset_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Config <-> preset bridges
# ---------------------------------------------------------------------------


def feature_cfg_to_preset(cfg: Any, *, name: str = "", label: str = "", description: str = "") -> dict[str, Any]:
    """Build a preset dict from a Config (or FeaturesCfg) for the lab/save flow."""
    features = getattr(cfg, "features", cfg)
    model = getattr(cfg, "model", None)
    payload: dict[str, Any] = {field: getattr(features, field) for field in _FEATURES_CFG_FIELDS}
    payload["name"] = name or getattr(features, "preset", "") or "preset"
    payload["label"] = label
    payload["description"] = description
    if model is not None:
        payload["sample_rate"] = int(model.sample_rate)
        payload["backbone"] = model.backbone
        payload["pretrained"] = bool(model.pretrained)
    return _normalize_preset(payload)


def apply_preset_to_features_cfg(features_cfg: Any, preset: dict[str, Any]) -> Any:
    """Return a FeaturesCfg with the preset's feature fields applied (validated)."""
    base = features_cfg.model_dump()
    for field in _FEATURES_CFG_FIELDS:
        if field in preset:
            base[field] = preset[field]
    if "name" in preset:
        # ``name`` is traceability for real saved presets only.  Model-contract
        # and custom payloads deliberately pass name="" so a stale saved-preset
        # marker cannot leak into training/export/inference bundles.
        base["preset"] = str(preset.get("name") or "")
    return config_mod.FeaturesCfg.model_validate(base)


def contract_feature_preset(backbone: str, *, name: str | None = None) -> dict[str, Any] | None:
    """Virtual feature preset for the selected pretrained checkpoint contract.

    This is not written to ``presets/audio_features``.  It is the concrete
    feature design implied by a curated pretrained variant, used by the Model
    tab and by ``/api/feature-presets`` source reporting.
    """
    try:
        from sed.train.model_lab import expected_contract_for_backbone

        contract = expected_contract_for_backbone(backbone)
    except Exception:  # noqa: BLE001
        contract = None
    if contract is None and config_mod.backbone_pretrained_available(backbone):
        contract = dict(config_mod.EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT)
    if not contract:
        return None

    defaults = config_mod.FeaturesCfg()
    payload: dict[str, Any] = {
        field: getattr(defaults, field) for field in _FEATURES_CFG_FIELDS
    }
    payload.update({
        "name": name if name is not None else f"contract:{backbone}",
        "label": f"Contract: {backbone}",
        "description": (
            "Unsaved model-default spectral and value-normalization contract for the "
            "selected pretrained variant, using this project's 0.4 s detection window."
        ),
        "source_kind": "checkpoint_contract",
        "feature_type": contract.get("features.feature_type", defaults.feature_type),
        "frontend": contract.get("features.frontend", defaults.frontend),
        "sample_rate": contract.get("model.sample_rate", config_mod.ModelCfg().sample_rate),
        "n_fft": contract.get("features.n_fft", defaults.n_fft),
        "win_length": contract.get("features.win_length", defaults.win_length),
        "hop_length": contract.get("features.hop_length", defaults.hop_length),
        "n_mels": contract.get("features.n_mels", defaults.n_mels),
        "fmin": contract.get("features.fmin", defaults.fmin),
        "fmax": contract.get("features.fmax", defaults.fmax),
        # EfficientAT's published crop duration is provenance, not a fixed
        # encoder dimension. Preserve its exact spectral/value contract while
        # applying the task-native real-time clip duration.
        "window_seconds": 0.4,
        "upstream_window_seconds": contract.get(
            "features.window_seconds", defaults.window_seconds
        ),
        "log_power": True,
        "pcen": False,
        "stft_engine": "librosa",
        "db_norm": False,
        "colormap": "none",
        "backbone": str(backbone),
        "pretrained": True,
    })
    return payload


def model_default_feature_preset(cfg: Any, *, project_id: str = "") -> dict[str, Any] | None:
    """Return the virtual feature payload for the selected model recipe.

    Upstream initialization has a checkpoint contract.  Random (scratch) and
    parent recipes instead expose the architecture's editable starting
    geometry.  Both are full payloads so the UI never offers a model-default
    choice it cannot load.
    """
    model = getattr(cfg, "model", None)
    if model is None:
        return None
    backbone = str(getattr(model, "backbone", "") or "")
    init = str(getattr(model, "initialization", "upstream") or "upstream").strip().lower()
    if backbone and bool(getattr(model, "pretrained", False)) and init == "upstream":
        contract = contract_feature_preset(backbone)
        if contract is not None:
            return contract

    recipe_id = str(getattr(model, "recipe_id", "") or "")
    if not recipe_id:
        return None
    try:
        from sed.train.model_lab import architecture_by_id, recommended_feature_for

        architecture = architecture_by_id(recipe_id, project_id)
        feature = recommended_feature_for(architecture)
    except Exception:  # noqa: BLE001 - unavailable project architecture has no virtual default
        feature = None
    if not feature:
        return None

    defaults = config_mod.FeaturesCfg()
    payload: dict[str, Any] = {
        field: getattr(defaults, field) for field in _FEATURES_CFG_FIELDS
    }
    payload.update({field: feature[field] for field in _FEATURE_FIELDS if field in feature})
    payload.update({
        "name": f"contract:{backbone or recipe_id}",
        "label": f"Model default: {backbone or architecture.label}",
        "description": "Unsaved editable starting geometry supplied by the selected architecture; it is not a checkpoint contract.",
        "source_kind": "architecture_default",
        "backbone": backbone,
        "pretrained": False,
    })
    return payload


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


def feature_config_matches_preset(cfg: Any, preset: dict[str, Any]) -> bool:
    """Whether ``cfg`` currently contains exactly this preset's feature fields."""
    try:
        normalized = _normalize_preset(preset)
    except Exception:  # noqa: BLE001
        normalized = dict(preset or {})
    features = getattr(cfg, "features", cfg)
    model = getattr(cfg, "model", None)
    for field in _FEATURES_CFG_FIELDS:
        if field in normalized and not _same_value(getattr(features, field), normalized[field]):
            return False
    if model is not None and "sample_rate" in normalized:
        if not _same_value(int(model.sample_rate), int(normalized["sample_rate"])):
            return False
    return True


def feature_source_for_config(cfg: Any, *, project_id: str = "") -> dict[str, str]:
    """Classify the project's current training feature for the inspect UI."""
    preset_name = str(getattr(getattr(cfg, "features", object()), "preset", "") or "")
    if preset_name:
        try:
            preset = load_feature_preset(preset_name)
        except Exception:  # noqa: BLE001
            preset = None
        if preset is not None and feature_config_matches_preset(cfg, preset):
            return {
                "kind": "saved",
                "name": str(preset.get("name") or preset_name),
                "label": str(preset.get("label") or preset.get("name") or preset_name),
                "base_kind": "",
                "base_name": "",
            }
        return {
            "kind": "custom",
            "name": "",
            "label": "Custom audio feature",
            "base_kind": "saved",
            "base_name": preset_name,
        }

    model = getattr(cfg, "model", None)
    backbone = str(getattr(model, "backbone", "") or "")
    contract = model_default_feature_preset(cfg, project_id=project_id) if backbone else None
    if contract is not None and feature_config_matches_preset(cfg, contract):
        return {
            "kind": "contract",
            "name": str(contract.get("name") or f"contract:{backbone}"),
            "label": str(contract.get("label") or f"Model default: {backbone}"),
            "source_kind": str(contract.get("source_kind") or "checkpoint_contract"),
            "base_kind": "",
            "base_name": "",
        }
    return {
        "kind": "custom",
        "name": "",
        "label": "Custom audio feature",
        "source_kind": "",
        "base_kind": "",
        "base_name": "",
    }


# ---------------------------------------------------------------------------
# Apply to a project
# ---------------------------------------------------------------------------

# The preset key recorded inside dataset_selection.json (dataset-scoped record).
DATASET_PRESET_KEY = "feature_preset"


def apply_feature_preset_to_project(
    project_id: str, name: str, *, set_model: bool = True
) -> dict[str, Any]:
    """Apply a saved preset to a project's config.toml + dataset selection.

    Writes the preset's feature fields into ``[features]`` (and, when
    ``set_model``, the model feature rate / backbone / pretrained suggestion into
    ``[model]``), records the preset name in ``dataset_selection.json``, and
    returns the reloaded config, the values written, and any pretrained-contract
    warning text (warn-only; never blocks).
    """
    from sed.data.compose import load_dataset_selection, write_dataset_selection

    preset = load_feature_preset(name)
    layout = config_mod.ProjectLayout(project_id)
    cfg = config_mod.load_config(project_id)

    features = apply_preset_to_features_cfg(cfg.features, preset)
    feature_updates = features.model_dump()
    updates: dict[str, dict[str, Any]] = {"features": feature_updates}
    if set_model:
        from sed.audio.features import feature_channels

        model_updates: dict[str, Any] = {"sample_rate": int(preset["sample_rate"])}
        if preset.get("backbone"):
            model_updates["backbone"] = str(preset["backbone"])
        if "pretrained" in preset:
            model_updates["pretrained"] = bool(preset["pretrained"])
        # A colormap image feature is multi-channel; keep model.channels (the
        # frozen export/inference input shape) in sync with the feature.
        model_updates["channels"] = int(feature_channels(features))
        updates["model"] = model_updates

    config_mod.update_config_toml_values(layout.config_toml, updates)

    # Record the active preset alongside the dataset so previews + training carry it.
    selection = load_dataset_selection(layout)
    selection[DATASET_PRESET_KEY] = preset["name"]
    write_dataset_selection(layout, selection)

    new_cfg = config_mod.load_config(project_id)
    warning = config_mod.pretrained_contract_warning(new_cfg)
    return {
        "ok": True,
        "preset": preset,
        "written": updates,
        "warning": warning,
        "cfg": new_cfg,
    }


# ---------------------------------------------------------------------------
# Preview-only feature (Library tab — dataset thumbnails)
# ---------------------------------------------------------------------------

# The Library tab's "preview feature" is recorded separately from the model /
# training feature. It only chooses how dataset clip thumbnails are rendered for
# browsing and management; it never touches config.toml ([features]/[model]) and
# is unrelated to the model assigned in the Model tab or to training.
DATASET_PREVIEW_KEY = "preview_feature"


def get_preview_preset_name(layout) -> str:
    """Name of the preview-only feature preset recorded for a project (or '')."""
    from sed.data.compose import load_dataset_selection

    try:
        return str(load_dataset_selection(layout).get(DATASET_PREVIEW_KEY) or "")
    except Exception:  # noqa: BLE001 — missing/corrupt selection ⇒ no override
        return ""


def set_preview_preset(project_id: str, name: str) -> dict[str, Any]:
    """Record (or clear) the preview-only feature preset for dataset thumbnails.

    Validates that the preset exists and writes its name into
    ``dataset_selection.json``. Does **not** modify ``config.toml`` — previews
    are independent of the model/training feature. Pass an empty ``name`` to
    revert previews to the project's current config feature.
    """
    from sed.data.compose import load_dataset_selection, write_dataset_selection

    layout = config_mod.ProjectLayout(project_id)
    selection = load_dataset_selection(layout)
    if not str(name or "").strip():
        selection.pop(DATASET_PREVIEW_KEY, None)
        write_dataset_selection(layout, selection)
        return {}
    preset = load_feature_preset(name)
    selection[DATASET_PREVIEW_KEY] = preset["name"]
    write_dataset_selection(layout, selection)
    return preset


def preview_feature_parts(cfg, layout) -> tuple[Any, int]:
    """``(features_cfg, sample_rate)`` to render dataset previews with.

    Returns the preview-only preset's feature when one is assigned, else the
    project's current config feature. Never raises — falls back to the config
    feature if the recorded preset is missing.
    """
    name = get_preview_preset_name(layout)
    if name:
        try:
            preset = load_feature_preset(name)
            return apply_preset_to_features_cfg(cfg.features, preset), int(preset["sample_rate"])
        except Exception:  # noqa: BLE001
            log.warning("preview preset %r unavailable; using project feature", name)
    return cfg.features, int(cfg.model.sample_rate)


def preview_features_dict(cfg, layout) -> dict[str, Any]:
    """Canonical preview-cache contract honoring the preview-only preset."""
    from sed.data.preview_cache import preview_features_from_parts

    features_cfg, sample_rate = preview_feature_parts(cfg, layout)
    return preview_features_from_parts(features_cfg, sample_rate)
