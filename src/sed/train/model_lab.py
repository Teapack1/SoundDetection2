"""Immutable model-recipes and the curated Model Lab catalog.

The trainer still consumes :class:`~sed.train.backbones.BackboneSpec`; this
module is the user-facing layer above it.  It deliberately separates a model
architecture from a checkpoint so a run can say *which* AudioSet or timm
variant it used, rather than merely saying ``pretrained=true``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# These labels describe a model's provenance and supported deployment path.  They
# intentionally do not make a performance claim: a model becomes a local
# recommendation only after a current benchmark in this project.
CATALOG_TAG_DEFINITIONS: dict[str, dict[str, str]] = {
    "established_baseline": {
        "label": "Established baseline",
        "description": "A stable first comparison, not a local performance ranking.",
        "tone": "baseline",
    },
    "research_candidate": {
        "label": "Research candidate",
        "description": "Promote only after a current local frozen-test and false-positive benchmark.",
        "tone": "research",
    },
    "task_specific_candidate": {
        "label": "Task-specific candidate",
        "description": "Designed for this task shape; validate it against the established baseline locally.",
        "tone": "research",
    },
    "spectrogram_baseline": {
        "label": "Spectrogram baseline",
        "description": "An image-style encoder applied to spectrogram features.",
        "tone": "baseline",
    },
    "audio_set_weights": {
        "label": "AudioSet weights available",
        "description": "An upstream AudioSet checkpoint can initialize the encoder.",
        "tone": "neutral",
    },
    "imagenet_weights": {
        "label": "ImageNet weights available",
        "description": "An upstream ImageNet checkpoint can initialize the image-style encoder.",
        "tone": "neutral",
    },
    "no_upstream_weights": {
        "label": "No upstream weights",
        "description": "Starts from scratch or a project parent transfer; no upstream checkpoint is supplied.",
        "tone": "neutral",
    },
    "short_clip_native": {
        "label": "Short-clip native",
        "description": "Its intended input geometry supports this project's short clips directly.",
        "tone": "neutral",
    },
    "openvino_deployable": {
        "label": "OpenVINO deployable",
        "description": "The catalog supports an OpenVINO export path for this architecture.",
        "tone": "baseline",
    },
    "pytorch_runtime_only": {
        "label": "PyTorch runtime only",
        "description": "No supported OpenVINO export path is available for this architecture.",
        "tone": "warning",
    },
    "legacy_compatibility": {
        "label": "Legacy compatibility",
        "description": "Retained to reproduce or compare older project workflows.",
        "tone": "warning",
    },
    "archived": {
        "label": "Archived",
        "description": "Kept for reproducibility; do not use for a new baseline without a deliberate reason.",
        "tone": "warning",
    },
}

from sed.config import CLINK_CNN_V2_FEATURE_CONTRACT, PROJECTS_DIR, REPO_ROOT
from sed.util import sha256_file


EFFICIENTAT_RELEASE = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1"


@dataclass(frozen=True)
class WeightSpec:
    id: str
    label: str
    source_url: str
    upstream_task: str
    expected_feature_contract: dict[str, Any] | None = None
    revision: str = ""
    sha256: str = ""
    license: str = "MIT"
    verified: bool = True

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureSpec:
    id: str
    label: str
    provider: str
    family: str
    source: str
    description: str
    capabilities: tuple[str, ...] = ("clip",)
    # Runtime/train-time support and Model-tab presentation are deliberately
    # separate.  ``capabilities`` remains the compatibility contract for old
    # configs and direct API users; ``selection_tasks`` controls which task
    # pickers offer the architecture for a new recipe.  None means "offer for
    # every supported task" and preserves the historical catalog behaviour.
    selection_tasks: tuple[str, ...] | None = None
    input_channels: int = 1
    params: tuple[dict[str, Any], ...] = ()
    weights: tuple[WeightSpec, ...] = ()
    export_supported: bool = True
    trusted: bool = True
    # Archived: kept in the catalog so old runs/manifests keep resolving and
    # rendering, but hidden from the Model tab's selection list unless the
    # project's current recipe still references it (July 2026 focus trim —
    # untested local approximations and ImageNet-only timm baselines).
    archived: bool = False
    # Optional packages the architecture needs at build time (e.g.
    # "transformers"); surfaced in the UI as an install hint instead of a
    # construction crash.
    requires: tuple[str, ...] = ()
    # Recommended starting feature geometry for the architecture itself —
    # informational for scratch runs (a contract belongs to a checkpoint, and
    # scratch loads none). None → derived by recommended_feature_for().
    default_feature: dict[str, Any] | None = None
    # Compact operator-facing facts rendered as badges in the Model tab.
    tags: tuple[str, ...] = ()
    # Explicit training-only regularization that is fixed by the architecture.
    # Adjustable mechanisms live on their corresponding ``params`` entries.
    regularization: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        selected = self.capabilities if self.selection_tasks is None else tuple(self.selection_tasks)
        unsupported = set(selected) - set(self.capabilities)
        if unsupported:
            raise ValueError(
                f"selection_tasks must be runtime capabilities; unsupported: {sorted(unsupported)}"
            )
        supported_types = {"float", "int", "bool", "choice", "string", "int_list"}
        seen: set[str] = set()
        for param in self.params:
            if not isinstance(param, dict):
                raise ValueError(f"{self.id} constructor parameters must be objects")
            name = str(param.get("name") or "").strip()
            kind = str(param.get("type") or "").strip()
            if not name or name in seen:
                raise ValueError(f"{self.id} has a missing or duplicate constructor parameter name")
            if kind not in supported_types:
                raise ValueError(f"{self.id}.{name} has unsupported type {kind!r}")
            if not str(param.get("help") or "").strip():
                raise ValueError(f"{self.id}.{name} needs a plain-language help description")
            if kind == "choice" and not param.get("choices"):
                raise ValueError(f"{self.id}.{name} choice parameter needs choices")
            if kind == "int_list" and not param.get("length"):
                raise ValueError(f"{self.id}.{name} integer list needs a fixed length")
            if param.get("mechanism") and not param.get("location"):
                raise ValueError(f"{self.id}.{name} regularization needs a location")
            seen.add(name)
        object.__setattr__(self, "selection_tasks", selected)

    def to_payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        data["selection_tasks"] = list(self.selection_tasks or ())
        data["requires"] = list(self.requires)
        data["tags"] = list(self.tags)
        data["available"] = self.dependencies_available()
        return data

    def dependencies_available(self) -> bool:
        """Whether the optional packages this architecture needs are importable.

        find_spec only inspects metadata — nothing heavy is imported here."""
        import importlib.util

        return all(importlib.util.find_spec(pkg) is not None for pkg in self.requires)


@dataclass(frozen=True)
class ModelRecipe:
    schema_version: int
    architecture_id: str
    backbone: str
    task: str
    weight_id: str
    initialization: str
    params: dict[str, Any] = field(default_factory=dict)
    parent_project: str = ""
    parent_run: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _contract(*, n_mels: int = 128, hop_length: int = 320) -> dict[str, Any]:
    return {
        "features.frontend": "efficientat",
        "features.feature_type": "mel",
        "model.sample_rate": 32000,
        "model.channels": 1,
        "features.n_fft": 1024,
        "features.win_length": 800,
        "features.hop_length": hop_length,
        "features.n_mels": n_mels,
        "features.fmin": 0,
        "features.fmax": 15000,
        "features.window_seconds": 1.0,
    }


def _efficient_weight(name: str, filename: str, *, n_mels: int = 128,
                      hop_length: int = 320) -> WeightSpec:
    return WeightSpec(
        id=f"efficientat:{name}", label=name,
        source_url=f"{EFFICIENTAT_RELEASE}/{filename}", upstream_task="AudioSet audio tagging",
        expected_feature_contract=_contract(n_mels=n_mels, hop_length=hop_length),
        revision="EfficientAT-v0.0.1",
    )


_CED_CONTRACT: dict[str, Any] = {
    "features.frontend": "ced",
    "features.feature_type": "mel",
    "model.sample_rate": 16000,
    "model.channels": 1,
    "features.n_fft": 512,
    "features.win_length": 512,
    "features.hop_length": 160,
    "features.n_mels": 64,
    "features.fmin": 0,
    "features.fmax": 8000,
    "features.window_seconds": 0.4,
}


_EFFICIENTSED_CONTRACT: dict[str, Any] = {
    "features.frontend": "efficientat",
    "features.feature_type": "mel",
    "model.sample_rate": 16000,
    "model.channels": 1,
    "features.n_fft": 512,
    "features.win_length": 400,
    "features.hop_length": 160,
    "features.n_mels": 128,
    "features.fmin": 0,
    "features.fmax": 8000,
    "features.window_seconds": 0.4,
}


def _ced_weight(size: str) -> WeightSpec:
    name = f"mispeech/ced-{size}"
    return WeightSpec(
        id=f"hf:mispeech:ced-{size}",
        label=name,
        source_url=f"https://huggingface.co/{name}",
        upstream_task="AudioSet audio tagging",
        expected_feature_contract=dict(_CED_CONTRACT),
        revision="main",
        license="Apache-2.0",
        # CED checkpoints store classifier-prefixed keys; the shared adapter
        # loads through AutoModelForAudioClassification and unwraps the encoder.
        verified=True,
    )


def _efficient_sed_weight(size: str) -> WeightSpec:
    filename = f"fmn{size}_strong.pt"
    checksums = {
        "10": "92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a",
        "20": "107afa1d1735f20591ed360eb7c1dfbc4b256c738339b039b281618c0955aaf8",
    }
    return WeightSpec(
        id=f"efficientsed:fmn{size}:strong",
        label=f"fmn{size}_strong",
        source_url=(
            "https://github.com/theMoro/EfficientSED/releases/download/v0.0.1/"
            f"{filename}"
        ),
        upstream_task="AudioSet Strong sound event detection",
        expected_feature_contract=dict(_EFFICIENTSED_CONTRACT),
        revision="EfficientSED-v0.0.1",
        sha256=checksums[size],
        license="MIT",
        verified=True,
    )


_EFFICIENTAT_WIDTH_RE = re.compile(r"^(dymn\d{2}|mn\d{2})")


def efficientat_base_backbone(name: str) -> str:
    """Canonical width-only architecture name for an EfficientAT backbone name.

    Checkpoint variants (``mn10_as_mels_256``, ``mn10_as_hop_5``, ``mn10_as_fc``,
    …) all share the same MobileNet architecture — the suffix describes the
    feature the *checkpoint* was trained on, not the network. A random
    (scratch) recipe therefore reduces to the base width name (``mn10_as``), so
    the recorded model never implies a feature contract or checkpoint that was
    never loaded. Non-EfficientAT names pass through unchanged.
    """
    m = _EFFICIENTAT_WIDTH_RE.match(str(name or ""))
    if not m:
        return str(name or "")
    from sed.config import BACKBONES

    base_as = f"{m.group(1)}_as"
    if base_as in BACKBONES:
        return base_as
    # Scratch-only widths (mn01, mn30…) are registered without the _as suffix —
    # never reduce to a name the registry can't build.
    if m.group(1) in BACKBONES:
        return m.group(1)
    return base_as


def expected_contract_for_weight(weight_id: str, project_id: str = "") -> dict[str, Any] | None:
    """The feature contract of one concrete checkpoint, resolved by weight id.

    Preferred over :func:`expected_contract_for_backbone` when the caller knows
    the selected weight: hop/mels checkpoint variants (mn10_as_hop_5…) share a
    backbone name-prefix but carry different contracts, and only the weight id
    disambiguates them reliably.
    """
    wid = str(weight_id or "")
    if not wid:
        return None
    for architecture in all_architectures(project_id):
        for weight in architecture.weights:
            if weight.id == wid and weight.expected_feature_contract:
                return dict(weight.expected_feature_contract)
    return None


def expected_contract_for_backbone(backbone: str) -> dict[str, Any] | None:
    """The feature contract baked into a curated pretrained checkpoint.

    Returns None for backbones without a curated per-weight contract (the
    caller falls back to the global EfficientAT contract). Kept import-light
    so config-side validation can call it lazily without pulling torch.
    """
    name = str(backbone)
    for architecture in curated_architectures():
        if architecture.provider != "efficientat":
            if name not in {architecture.id, architecture.source}:
                continue
        for weight in architecture.weights:
            if (weight.label == name or name in {architecture.id, architecture.source}) and weight.expected_feature_contract:
                return dict(weight.expected_feature_contract)
    return None


_EFFICIENT_WEIGHTS = (
    _efficient_weight("mn04_as", "mn04_as_mAP_432.pt"),
    _efficient_weight("mn05_as", "mn05_as_mAP_443.pt"),
    _efficient_weight("mn10_as", "mn10_as_mAP_471.pt"),
    _efficient_weight("mn20_as", "mn20_as_mAP_478.pt"),
    _efficient_weight("mn30_as", "mn30_as_mAP_482.pt"),
    _efficient_weight("mn40_as", "mn40_as_mAP_484.pt"),
    _efficient_weight("mn10_as_hop_5", "mn10_as_hop_5_mAP_475.pt", hop_length=160),
    _efficient_weight("mn10_as_hop_15", "mn10_as_hop_15_mAP_463.pt", hop_length=480),
    _efficient_weight("mn10_as_hop_20", "mn10_as_hop_20_mAP_456.pt", hop_length=640),
    _efficient_weight("mn10_as_hop_25", "mn10_as_hop_25_mAP_447.pt", hop_length=800),
    _efficient_weight("mn10_as_mels_40", "mn10_as_mels_40_mAP_453.pt", n_mels=40),
    _efficient_weight("mn10_as_mels_64", "mn10_as_mels_64_mAP_461.pt", n_mels=64),
    _efficient_weight("mn10_as_mels_256", "mn10_as_mels_256_mAP_474.pt", n_mels=256),
    _efficient_weight("mn10_as_fc", "mn10_as_fc_mAP_465.pt"),
)


_DYMN_WEIGHTS = (
    _efficient_weight("dymn04_as", "dymn04_as.pt"),
    _efficient_weight("dymn10_as", "dymn10_as.pt"),
    _efficient_weight("dymn20_as", "dymn20_as_mAP_493.pt"),
)


_PARAMS_WIDTH_TIME = (
    {"name": "width_mult", "type": "float", "default": 1.0, "min": 0.25, "max": 4.0,
     "help": "Scales the channel count of every conv layer. 2.0 doubles the width and "
             "roughly quadruples compute; 0.5 halves it. Widen only when training loss "
             "stays high — on this task capacity is rarely the limit.",
     "recommended": [
         {"max_train_rows": 2000, "value": 0.5,
          "why": "a narrow encoder has less to memorize on a small set"},
         {"max_groups": 10, "value": 0.5,
          "why": "few independent recording groups: extra width memorizes those conditions "
                 "rather than learning the sound"},
         {"value": 1.0, "why": "the tested default width for this encoder"},
     ]},
    {"name": "time_stride", "type": "choice", "default": 1, "choices": [1, 2],
     "help": "Downsampling factor applied to the time axis by the first conv. 2 halves the "
             "frame rate for roughly twice the speed, but brief transients (glass clinks, "
             "consonants) can fall between frames. Use 2 only for sustained sounds.",
     "recommended": [
         {"value": 1, "why": "short glass and attention events need full temporal resolution"},
     ]},
    {"name": "depth", "type": "int", "default": 4, "min": 1, "max": 12,
     "help": "Number of stacked residual blocks. More blocks give a larger receptive field "
             "and more abstract features at linear cost in compute and memory. Add blocks "
             "when the model underfits a large, varied corpus.",
     "recommended": [
         {"max_train_rows": 2000, "value": 3, "why": "fewer blocks to fit a small set"},
         {"value": 4, "why": "the tested default depth for this encoder"},
     ]},
)


_PARAM_DROPOUT = {"name": "dropout", "type": "float", "default": 0.1, "min": 0.0, "max": 0.8,
                  "help": "Zeros encoder activations during training, off at evaluation and "
                          "export. Stops the network leaning on a few features. Raise it "
                          "when training loss falls well below validation loss; too high "
                          "starves a small model of signal.",
                  "mechanism": "dropout", "location": "encoder_activation",
                  "recommended": [
                      {"max_train_rows": 2000, "value": 0.2,
                       "why": "small sets start overfitting early, so regularize harder"},
                      {"max_groups": 10, "value": 0.2,
                       "why": "few independent recording groups make the corpus redundant, "
                              "which overfits like a smaller one"},
                      {"value": 0.1,
                       "why": "the tested default; a varied corpus already regularizes"},
                  ]}


_PARAMS_SHORT_CNN = _PARAMS_WIDTH_TIME + (_PARAM_DROPOUT,)


def _optional_param(name: str, kind: str, default: Any, help_text: str,
                    **metadata: Any) -> dict[str, Any]:
    """Declared opt-in constructor argument; omitted JSON uses builder default."""
    return {
        "name": name, "type": kind, "default": default, "optional": True,
        "help": help_text, **metadata,
    }


_EFFICIENTAT_MN_PARAMS = (
    _optional_param(
        "reduced_tail", "bool", False,
        "Halves the final MobileNet tail channels. This reduces capacity and compute; "
        "it changes encoder shape, so use it only for scratch experiments.",
    ),
    _optional_param(
        "dilated", "bool", False,
        "Replaces the final spatial stride with dilation, retaining a finer feature map "
        "at higher memory cost.",
    ),
    _optional_param(
        "strides", "int_list", [2, 2, 2, 2],
        "Four per-stage downsampling factors. Use 1 to preserve resolution or 2 to "
        "downsample; lower strides cost more memory and compute.",
        length=4, item_choices=[1, 2],
    ),
    _optional_param(
        "input_dim_f", "int", 128,
        "Frequency rows used when frequency squeeze-and-excitation is enabled. It must "
        "match the selected feature geometry.", min=1,
    ),
    _optional_param(
        "input_dim_t", "int", 1000,
        "Time frames used when temporal squeeze-and-excitation is enabled. It must match "
        "the selected feature geometry.", min=1,
    ),
    _optional_param(
        "se_dims", "choice", "c",
        "Axes used by squeeze-and-excitation: channel, frequency, time, or a declared "
        "combination. Frequency/time choices tie the encoder to the input geometry.",
        choices=["none", "c", "f", "t", "cf", "ct", "ft", "cft"],
    ),
    _optional_param(
        "se_agg", "choice", "max",
        "How concurrent squeeze-and-excitation branches are combined.",
        choices=["max", "avg", "add", "min"],
    ),
    _optional_param(
        "se_r", "int", 4,
        "Squeeze-and-excitation reduction ratio. Larger values use a narrower gating "
        "bottleneck and slightly less compute.", min=1,
    ),
)


_EFFICIENTAT_DYMN_PARAMS = (
    _optional_param(
        "strides", "int_list", [2, 2, 2, 2],
        "Four per-stage downsampling factors. Use 1 to preserve resolution or 2 to "
        "downsample; lower strides cost more memory and compute.",
        length=4, item_choices=[1, 2],
    ),
    _optional_param("context_ratio", "int", 4,
                    "Divides expanded channels to size the dynamic context vector.", min=1),
    _optional_param("max_context_size", "int", 128,
                    "Upper channel limit for the dynamic context vector.", min=8),
    _optional_param("min_context_size", "int", 32,
                    "Lower channel limit for the dynamic context vector.", min=1),
    _optional_param("dyrelu_k", "int", 2,
                    "Number of learned linear mappings mixed by Dynamic ReLU.", min=1),
    _optional_param("no_dyrelu", "bool", False,
                    "Disable Dynamic ReLU and use the static activation path."),
    _optional_param("dyconv_k", "int", 4,
                    "Number of candidate kernels mixed by each dynamic convolution.", min=1),
    _optional_param("no_dyconv", "bool", False,
                    "Disable dynamic convolution and use the static convolution path."),
    _optional_param("T_max", "float", 30.0,
                    "Starting softmax temperature for dynamic-kernel mixing.", min=1.0),
    _optional_param("T0_slope", "float", 1.0,
                    "Early-epoch temperature decrease per epoch.", min=0.000001),
    _optional_param("T1_slope", "float", 0.02,
                    "Late-epoch temperature decrease per epoch.", min=0.0),
    _optional_param("T_min", "float", 1.0,
                    "Minimum dynamic-kernel mixing temperature.", min=0.0),
    _optional_param("no_ca", "bool", False,
                    "Disable coordinate attention inside dynamic blocks."),
    _optional_param(
        "use_dy_blocks", "choice", "all",
        "Use dynamic blocks everywhere or only where MobileNet would use "
        "squeeze-and-excitation.", choices=["all", "replace_se"],
    ),
)


def _short_event_feature(source: str = "short-event mel default") -> dict[str, Any]:
    return {
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 1024, "win_length": 882, "hop_length": 220, "n_mels": 128,
        "fmin": 0, "fmax": 22050, "window_seconds": 0.4,
        "pcen": False, "log_power": True, "stft_engine": "librosa",
        "db_norm": False, "colormap": "none", "image_size": 96,
        # Explicit off: this dict can be merged onto a project's existing
        # [features] (Model tab "Verify"), so an omitted key would silently
        # inherit a stale value (e.g. standardize=True left over from a prior
        # linear preset) instead of the plain log-mel this default means.
        "standardize": False, "standardize_mean": 0.0, "standardize_std": 1.0,
        "scaling_method": "none", "scaling_scope": "global", "scaling_stats": {},
        "standardize_stats_resolved": True, "standardize_stats_dataset_sha256": "",
        "standardize_stats_split_seed": 0, "standardize_stats_val_fraction": 0.0,
        "standardize_stats_clips": 0,
        "source": source,
    }


def _clink_v2_feature(source: str = "ClinkCNN-v2 0.4 s feature contract") -> dict[str, Any]:
    # Native 44.1 kHz. A 20 ms Hann window resolves glass resonances while the
    # ~5 ms hop retains the attack/decay envelope. EfficientAT's pre-emphasis,
    # Kaldi/HTK mel bank, natural log and fast normalization are kept intact.
    # 17,640 samples -> 81 centered frames after the one-sample pre-emphasis.
    return {
        **_contract_as_feature(CLINK_CNN_V2_FEATURE_CONTRACT),
        "source": source,
    }


_TIMM_PARAMS = (
    {"name": "drop_rate", "type": "float", "default": 0.0, "min": 0.0, "max": 0.9,
     "help": "Zeros values of the pooled embedding just before the classifier, during "
             "training only. Leave it at 0: the Train tab's classifier dropout acts on the "
             "same vector and is the knob to use, because it is a training setting rather "
             "than part of the model recipe.",
     "recommended": [
         {"value": 0.0,
          "why": "the Train tab's classifier dropout owns this location — setting it there "
                 "keeps it out of the recipe signature, so two runs that differ only in "
                 "dropout stay comparable"},
     ], "mechanism": "dropout", "location": "pooled_embedding"},
    {"name": "drop_path_rate", "type": "float", "default": 0.0, "min": 0.0, "max": 0.9,
     "help": "Stochastic depth: each residual block is skipped entirely with this "
             "probability during training, ramped from 0 at the stem to this value at the "
             "last block, so the network trains as an ensemble of shallower nets. This is "
             "the knob for regularizing the encoder itself. Raise it when training loss "
             "falls well below validation loss; lower it when the run underfits.",
     "recommended": [
         {"init": ["random", "parent"], "max_train_rows": 2000, "value": 0.0,
          "why": "a scratch encoder needs signal before it needs noise at this size"},
         {"init": ["random", "parent"], "value": 0.1,
          "why": "scratch runs train long enough for stochastic depth to pay for itself"},
         {"max_train_rows": 2000, "value": 0.0,
          "why": "a short fine-tune on a small set cannot absorb the extra noise — it only "
                 "slows convergence"},
         {"max_groups": 10, "value": 0.05,
          "why": "few independent recording groups: cap it here, since more sessions and "
                 "venues help far more than stronger regularization"},
         {"max_train_rows": 20000, "value": 0.05,
          "why": "mild stochastic depth helps once a full fine-tune has enough data to overfit"},
         {"value": 0.1,
          "why": "a large, diverse corpus supports the standard ImageNet-scale setting"},
     ], "mechanism": "stochastic_depth", "location": "residual_blocks"},
)


def _timm_arch(arch_id: str, label: str, source: str, description: str,
               *, archived: bool = False) -> ArchitectureSpec:
    """A curated timm/ImageNet backbone for single-channel spectrograms.

    ImageNet-pretrained weights transfer to log-mel spectrograms; ``TimmAdapter``
    builds the model with ``in_chans=1`` and global average pooling, so any input
    geometry works and OpenVINO export uses only standard ops. For noisy
    real-world audio the AudioSet-pretrained EfficientAT ``mn`` family is still
    preferable — these are strong, well-understood general-purpose CNN baselines.
    The default feature is a 0.4 s log-mel geometry (scratch runs load no
    checkpoint, so this is informational only).
    """
    return ArchitectureSpec(
        id=arch_id, label=label, provider="timm", family="timm", source=source,
        description=description, capabilities=("clip",), params=_TIMM_PARAMS,
        weights=(WeightSpec(f"timm:{source}:default", "timm ImageNet weights",
                            f"timm://{source}", "ImageNet classification",
                            license="Apache-2.0"),),
        archived=archived,
        default_feature=_short_event_feature(f"{label} 0.4 s mel default"),
        tags=("spectrogram_baseline", "imagenet_weights", "openvino_deployable"),
    )


def curated_architectures() -> list[ArchitectureSpec]:
    """Static catalog available even when optional training packages are absent."""
    return [
        ArchitectureSpec(
            id="efficientat-mn", label="EfficientAT MobileNet", provider="efficientat",
            family="efficientat", source="mn10_as",
            description="AudioSet-pretrained MobileNet variants with published log-mel contracts.",
            capabilities=("clip", "frame"), params=_EFFICIENTAT_MN_PARAMS,
            weights=_EFFICIENT_WEIGHTS,
            tags=("established_baseline", "audio_set_weights", "short_clip_native", "openvino_deployable"),
        ),
        ArchitectureSpec(
            id="efficientat-dymn", label="EfficientAT Dynamic MobileNet", provider="efficientat",
            family="efficientat", source="dymn10_as",
            description="AudioSet-pretrained Dynamic MobileNet (DyMN) with dynamic blocks. "
            "It uses the same 128-mel @ 32 kHz log-mel contract as MN. "
            "Trainable here; OpenVINO export is disabled until DyMN's dynamic blocks have "
            "a supported export path.",
            capabilities=("clip", "frame"), params=_EFFICIENTAT_DYMN_PARAMS,
            weights=_DYMN_WEIGHTS, export_supported=False,
            tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only"),
        ),
        # ImageNet-pretrained timm CNN baselines for spectrograms. They remain
        # useful comparisons, but require local evidence before any ranking.
        _timm_arch("timm-resnet18", "ResNet-18", "resnet18",
                   "18-layer residual CNN — the fast, reliable spectrogram baseline (PANNs/PSLA lineage)."),
        _timm_arch("timm-resnet34", "ResNet-34", "resnet34",
                   "34-layer residual CNN with BatchNorm and residual blocks."),
        _timm_arch("timm-resnet50", "ResNet-50", "resnet50",
                   "50-layer bottleneck ResNet — more capacity for harder classes at higher compute."),
        _timm_arch("timm-efficientnet-b0", "EfficientNet-B0", "efficientnet_b0",
                   "Compact EfficientNet baseline for small spectrogram experiments."),
        _timm_arch("timm-efficientnet-b1", "EfficientNet-B1", "efficientnet_b1",
                   "Slightly larger EfficientNet — more capacity than B0 at modest extra cost."),
        _timm_arch("timm-convnext-nano", "ConvNeXt Nano", "convnext_nano",
                   "Small modern ConvNeXt — strong features at low parameter count."),
        _timm_arch("timm-convnext-tiny", "ConvNeXt Tiny", "convnext_tiny",
                   "Modern high-capacity ConvNeXt (ImageNet, not the AudioSet ConvNeXt-Tiny-AT checkpoint). "
                   "Downsamples 32×, aggressive for ~0.4 s clips — a secondary choice."),
        # Retired baseline kept archived so any prior run referencing it still resolves.
        _timm_arch("timm-mobilenetv3-small", "MobileNetV3 Small", "mobilenetv3_small_100",
                   "Efficient vision transfer baseline for square spectrogram experiments.",
                   archived=True),
        ArchitectureSpec(
            "legacy-smallervggnet", "SmallerVGGNet (legacy port)", "local", "custom",
            "sed.train.legacy_vgg:SmallerVGGNet",
            "Exact port of the legacy soundDetection Keras model. It consumes the JET colormap image "
            "the audio feature produces (set colormap=jet on the 'Legacy soundDetection' feature "
            "preset) and classifies it with the SmallerVGGNet conv stack (1024-d dense embedding). "
            "Pair with that preset and cross-entropy loss to reproduce the legacy workflow.",
            ("clip",),
            input_channels=3,
            params=(
                {"name": "dense_units", "type": "int", "default": 1024, "min": 64, "max": 4096,
                 "help": "Width of the dense embedding between the conv stack and the "
                         "classifier. This layer is flattened from the feature map, so it "
                         "holds most of the model's parameters and is shape-dependent — "
                         "changing the feature geometry changes its size.",
                 "recommended": [{"max_train_rows": None, "value": 1024,
                                  "why": "keep the legacy value or the comparison is not a port"}]},
                {"name": "conv_dropout", "type": "float", "default": 0.25, "min": 0.0, "max": 0.9,
                 "help": "Zeros activations after each conv block during training, off at "
                         "evaluation and export. Keep the legacy value when reproducing the "
                         "original model.",
                 "mechanism": "dropout", "location": "encoder_activation",
                 "recommended": [{"value": 0.25,
                                  "why": "the original soundDetection value — changing it "
                                         "makes this no longer a port"}]},
                {"name": "dense_dropout", "type": "float", "default": 0.5, "min": 0.0, "max": 0.9,
                 "help": "Zeros values of the dense embedding during training, off at "
                         "evaluation and export. The Train tab's classifier dropout acts "
                         "just after this same vector, so with auto on it resolves to 0.",
                 "mechanism": "dropout", "location": "pooled_embedding",
                 "recommended": [{"value": 0.5,
                                  "why": "the original soundDetection value — changing it "
                                         "makes this no longer a port"}]},
            ),
            tags=("no_upstream_weights", "legacy_compatibility", "openvino_deployable"),
        ),
        ArchitectureSpec("transient-cnn-gap", "Transient CNN GAP", "local", "custom",
                         "sed.train.short_audio:TransientCNNGAP",
                         "Small local CNN with global pooling for short glass transients.",
                         ("clip", "frame"), params=_PARAMS_SHORT_CNN,
                         default_feature=_short_event_feature(),
                         tags=("no_upstream_weights", "short_clip_native", "openvino_deployable")),
        ArchitectureSpec("transient-resnet", "Transient ResNet", "local", "custom",
                         "sed.train.short_audio:TransientResNet",
                         "Shallow residual spectrogram CNN with low temporal downsampling.",
                         ("clip", "frame"), params=_PARAMS_SHORT_CNN,
                         default_feature=_short_event_feature(),
                         tags=("no_upstream_weights", "short_clip_native", "openvino_deployable")),
        ArchitectureSpec(
            "clink-cnn-v2", "ClinkCNN-v2 (frequency-forward)", "local", "custom",
            "sed.train.clink_cnn_v2:ClinkCNNv2",
            "Task-specific encoder for 0.4 s glass impacts. It preserves explicit "
            "resonance position and attack/decay structure with raw, positive-delta and fixed "
            "frequency-coordinate planes; anisotropic residual mixing; temporal attentive "
            "statistics; and frequency-pyramid pooling. Designed for <=1.2M parameters, "
            "<=0.25 GMAC/clip and OpenVINO CPU deployment. Native 44.1 kHz / 128-mel input.",
            ("clip",),
            default_feature=_clink_v2_feature(),
            tags=("task_specific_candidate", "no_upstream_weights", "short_clip_native", "openvino_deployable"),
            regularization=(
                {
                    "name": "pool_dropout", "label": "pooling dropout",
                    "mechanism": "dropout", "location": "pooled_embedding",
                    "value": 0.3,
                    "help": "Zeros values in the 512-value pooled embedding before the "
                            "shared classifier, whenever the encoder trains. Built into the "
                            "encoder, so it is not a constructor argument and cannot be "
                            "changed here. The Train tab's classifier dropout acts on the "
                            "same vector — with auto on it resolves to 0 so the two are "
                            "never stacked.",
                },
                {
                    "name": "drop_path", "label": "stochastic depth",
                    "mechanism": "stochastic_depth", "location": "residual_blocks",
                    "value": {"min": 0.0, "max": 0.1},
                    "min_trainable_stages": 2,
                    "help": "Skips residual branches per sample, ramping from 0 at the first "
                            "block to 0.1 at the last block, so the encoder trains as an "
                            "ensemble of shallower nets. Built into the encoder and active "
                            "only once at least two stages train.",
                },
            ),
        ),
        ArchitectureSpec("kws-bcresnet", "BC-ResNet", "local", "custom", "sed.train.kws:BCResNet",
                         "Local broadcast-residual KWS template for short spectrograms.", ("clip", "frame"),
                         params=_PARAMS_WIDTH_TIME,
                         default_feature=_short_event_feature("BC-ResNet 0.4 s short-event geometry"),
                         tags=("no_upstream_weights", "short_clip_native", "openvino_deployable")),
        ArchitectureSpec("kws-tcresnet", "TC-ResNet", "local", "custom", "sed.train.kws:TCResNet",
                         "Local temporal residual KWS template with low time downsampling.", ("clip", "frame"),
                         params=_PARAMS_WIDTH_TIME, archived=True,
                         tags=("no_upstream_weights", "short_clip_native", "openvino_deployable")),
        ArchitectureSpec("kws-dscnn", "DS-CNN", "local", "custom", "sed.train.kws:DSCNN",
                         "Local depthwise-separable CNN KWS template.", ("clip", "frame"),
                         params=_PARAMS_WIDTH_TIME, archived=True,
                         tags=("no_upstream_weights", "short_clip_native", "openvino_deployable")),
        ArchitectureSpec("ced-tiny", "CED Tiny", "local", "custom",
                         "sed.train.external_audio:CEDTinyEncoder",
                         "CED-Tiny AudioSet Transformer from mispeech/ced-tiny (transformers). "
                         "Trains and evaluates with the CED 16 kHz / 64-mel dB frontend; "
                         "live inference runs the PyTorch checkpoint (no OpenVINO export).",
                         ("clip",), weights=(_ced_weight("tiny"),), export_supported=False,
                         requires=("transformers",),
                         default_feature={**_contract_as_feature(_CED_CONTRACT), "source": "CED-Tiny feature contract"},
                         tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only")),
        ArchitectureSpec("ced-mini", "CED Mini", "local", "custom",
                         "sed.train.external_audio:CEDMiniEncoder",
                         "9.6M-parameter CED AudioSet transformer for a compact transformer comparison.",
                         ("clip",), weights=(_ced_weight("mini"),), export_supported=False,
                         requires=("transformers",),
                         default_feature={**_contract_as_feature(_CED_CONTRACT), "source": "CED-Mini feature contract"},
                         tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only")),
        ArchitectureSpec("ced-small", "CED Small", "local", "custom",
                         "sed.train.external_audio:CEDSmallEncoder",
                         "22M-parameter CED AudioSet transformer with a larger encoder capacity.",
                         ("clip",), weights=(_ced_weight("small"),), export_supported=False,
                         requires=("transformers",),
                         default_feature={**_contract_as_feature(_CED_CONTRACT), "source": "CED-Small feature contract"},
                         tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only")),
        ArchitectureSpec("ced-base", "CED Base", "local", "custom",
                         "sed.train.external_audio:CEDBaseEncoder",
                         "86M-parameter CED AudioSet transformer retained as a large-capacity benchmark.",
                         ("clip",), weights=(_ced_weight("base"),), export_supported=False,
                         requires=("transformers",),
                         default_feature={**_contract_as_feature(_CED_CONTRACT), "source": "CED-Base feature contract"},
                         tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only")),
        ArchitectureSpec("efficient-sed-fmn10", "EfficientSED fMN10", "local", "custom",
                         "sed.train.efficient_sed:EfficientSEDfMN10",
                         "Official frame-wise MobileNet with AudioSet Strong weights and a 10 ms-hop frontend.",
                         ("clip", "frame"), weights=(_efficient_sed_weight("10"),), export_supported=False,
                         selection_tasks=("frame",),
                         default_feature={**_contract_as_feature(_EFFICIENTSED_CONTRACT), "source": "EfficientSED fMN10 contract"},
                         tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only")),
        ArchitectureSpec("efficient-sed-fmn20", "EfficientSED fMN20", "local", "custom",
                         "sed.train.efficient_sed:EfficientSEDfMN20",
                         "Wider official frame-wise MobileNet with AudioSet Strong weights.",
                         ("clip", "frame"), weights=(_efficient_sed_weight("20"),), export_supported=False,
                         selection_tasks=("frame",),
                         default_feature={**_contract_as_feature(_EFFICIENTSED_CONTRACT), "source": "EfficientSED fMN20 contract"},
                         tags=("research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only")),
    ]


def project_architectures(project_id: str) -> list[ArchitectureSpec]:
    """Read trusted project-local architecture recipes without importing code."""
    root = PROJECTS_DIR / project_id / "architectures"
    out: list[ArchitectureSpec] = []
    if not root.exists():
        return out
    for recipe_path in sorted(root.glob("*/recipe.json")):
        try:
            raw = json.loads(recipe_path.read_text(encoding="utf-8"))
            slug = recipe_path.parent.name
            source = str(raw.get("source") or f"project:{project_id}:{slug}")
            if source != f"project:{project_id}:{slug}":
                continue
            out.append(ArchitectureSpec(
                id=str(raw.get("id") or f"project:{project_id}:{slug}"),
                label=str(raw.get("label") or slug.replace("_", " ").title()),
                provider="local", family="custom", source=source,
                description="Project-local custom encoder (source snapshotted into each run).",
                capabilities=tuple(raw.get("capabilities") or ["clip"]),
                selection_tasks=(
                    tuple(raw["selection_tasks"])
                    if raw.get("selection_tasks") is not None else None
                ),
                params=tuple(raw.get("params") or ()),
            ))
        except Exception:  # noqa: BLE001
            continue
    return out


def all_architectures(project_id: str = "") -> list[ArchitectureSpec]:
    return curated_architectures() + (project_architectures(project_id) if project_id else [])


def architecture_by_id(architecture_id: str, project_id: str = "") -> ArchitectureSpec:
    for spec in all_architectures(project_id):
        if spec.id == architecture_id:
            return spec
    raise KeyError(f"unknown curated architecture {architecture_id!r}")


def _regularization_active(value: Any) -> bool:
    if isinstance(value, dict):
        return float(value.get("max", 0.0) or 0.0) > 0.0
    try:
        return float(value or 0.0) > 0.0
    except (TypeError, ValueError):
        return bool(value)


def _requested_trainable_stages(train_depth: str) -> float:
    depth = str(train_depth or "full").strip().lower()
    if depth in {"head", "head_only", "none"}:
        return 0.0
    if depth in {"last", "last_block", "last_stage", "last1"}:
        return 1.0
    match = re.match(r"^last_n[:=](\d+)$", depth)
    if match:
        return float(int(match.group(1)))
    return float("inf")


def architecture_dropout_info(
    architecture_id: str,
    params: dict | None = None,
    project_id: str = "",
    *,
    train_depth: str = "full",
) -> dict[str, Any]:
    """Return the explicit, effective dropout stack for one architecture.

    Every architecture-owned mechanism is inactive during a genuinely frozen
    (head-only) backbone phase because the full encoder is kept in eval mode.
    Partial/full target depths activate the trailing encoder path, including
    final pooling dropout.  This distinction drives the classifier-dropout
    automatic default and prevents frozen and unfrozen training from being
    treated as the same regularization stack.
    """
    values = dict(params or {})
    try:
        spec = architecture_by_id(str(architecture_id or ""), project_id)
    except KeyError:
        return {
            "architecture_id": str(architecture_id or ""), "label": "",
            "rows": [], "encoder_trainable": False,
            "recommended_head_dropout": 0.2,
        }
    requested_stages = _requested_trainable_stages(train_depth)
    encoder_trainable = requested_stages > 0
    rows: list[dict[str, Any]] = []
    for param in spec.params or ():
        mechanism = str(param.get("mechanism") or "")
        if not mechanism:
            continue
        name = str(param.get("name") or "")
        default = param.get("default")
        value = values.get(name, default)
        rows.append({
            "name": name,
            "label": str(param.get("label") or name.replace("_", " ")),
            "mechanism": mechanism,
            "location": str(param.get("location") or "encoder"),
            "value": value,
            "default": default,
            "fixed": False,
            "owner": "model",
            "help": str(param.get("help") or ""),
            "active": encoder_trainable and _regularization_active(value),
            "overridden": name in values and values[name] != default,
        })
    for item in spec.regularization or ():
        value = item.get("value")
        min_trainable_stages = int(item.get("min_trainable_stages", 1) or 1)
        rows.append({
            "name": str(item.get("name") or ""),
            "label": str(item.get("label") or item.get("name") or "regularization"),
            "mechanism": str(item.get("mechanism") or "dropout"),
            "location": str(item.get("location") or "encoder"),
            "value": value,
            "default": value,
            "fixed": True,
            "owner": "architecture",
            "help": str(item.get("help") or ""),
            "min_trainable_stages": min_trainable_stages,
            "active": (
                requested_stages >= min_trainable_stages
                and _regularization_active(value)
            ),
            "overridden": False,
        })
    final_embedding_dropout = any(
        row["active"]
        and row["mechanism"] == "dropout"
        and row["location"] == "pooled_embedding"
        for row in rows
    )
    return {
        "architecture_id": spec.id,
        "label": spec.label,
        "rows": rows,
        "encoder_trainable": encoder_trainable,
        "recommended_head_dropout": 0.0 if final_embedding_dropout else 0.2,
    }


#: Initialization modes a recommendation band can be scoped to. These mirror the
#: Model tab's ``initialization`` selector.
INIT_MODES = ("upstream", "random", "parent")


def guidance_context(dataset: dict[str, Any] | None = None, *,
                     train_rows: int = 0, init: str = "upstream") -> dict[str, Any]:
    """Normalize the dataset/initialization facts a recommendation band reads.

    ``dataset`` is the shape dict produced by the inspect server
    (``_dataset_shape``): row count, per-class counts, negative share and the
    smallest per-class count of independent recording groups. Everything is
    optional — a missing fact simply cannot constrain a band.
    """
    data = dict(dataset or {})
    rows = int(data.get("rows") or train_rows or 0)
    groups = int(data.get("min_class_groups") or 0)
    classes = data.get("classes") or []
    return {
        "rows": rows,
        "groups": groups,
        "n_classes": int(data.get("n_classes") or len(classes) or 0),
        "negative_share": float(data.get("negative_share") or 0.0),
        "init": str(init or "upstream"),
    }


def _band_matches(band: dict[str, Any], ctx: dict[str, Any]) -> bool:
    """Whether every constraint a band declares holds for ``ctx``.

    Undeclared keys are unconstrained, so size-only bands keep behaving exactly
    as they did before initialization and group scoping existed. A constraint
    that needs a fact the context does not have (``groups == 0`` means "not
    measured") is skipped rather than failing, so a project without composed
    diagnostics still gets the size-based advice.
    """
    ceiling = band.get("max_train_rows")
    if ceiling is not None and ctx["rows"] > int(ceiling):
        return False
    floor = band.get("min_train_rows")
    if floor is not None and ctx["rows"] < int(floor):
        return False
    inits = band.get("init")
    if inits and ctx["init"] not in {str(i) for i in inits}:
        return False
    if ctx["groups"] > 0:
        group_ceiling = band.get("max_groups")
        if group_ceiling is not None and ctx["groups"] > int(group_ceiling):
            return False
        group_floor = band.get("min_groups")
        if group_floor is not None and ctx["groups"] < int(group_floor):
            return False
    if ctx["n_classes"] > 0:
        class_ceiling = band.get("max_classes")
        if class_ceiling is not None and ctx["n_classes"] > int(class_ceiling):
            return False
    return True


_INIT_WORDS = {"upstream": "pretrained", "random": "scratch", "parent": "parent transfer"}


def _scope_phrase(ctx: dict[str, Any]) -> str:
    """Short human phrase naming the facts a recommendation was resolved from."""
    parts = [_INIT_WORDS.get(ctx["init"], ctx["init"])]
    if ctx["rows"]:
        parts.append(f"{ctx['rows']:,} clips")
    if ctx["groups"]:
        parts.append(f"{ctx['groups']} recording groups")
    return " · ".join(parts)


def _recommended_param_value(param: dict[str, Any],
                             ctx: dict[str, Any]) -> tuple[Any, str]:
    """Pick the first ``recommended`` band whose declared constraints all hold.

    Bands are ``{"max_train_rows": int | None, "init": [...], "max_groups": int,
    "value": ..., "why": str}``, most specific first; a band declaring nothing
    but a value is the open-ended fallback. Falls back to the declared default
    when a knob has no bands or none match.
    """
    for band in param.get("recommended") or ():
        if _band_matches(band, ctx):
            return band.get("value"), str(band.get("why") or "")
    return param.get("default"), ""


def param_guidance(spec: ArchitectureSpec, params: dict[str, Any] | None = None,
                   train_rows: int = 0, *, dataset: dict[str, Any] | None = None,
                   init: str = "upstream") -> list[dict[str, Any]]:
    """Annotate each typed constructor knob for the Model tab readout.

    The Model tab used to render the merged parameter dict as bare JSON, which
    says nothing about what a key does or what to set it to. This resolves, per
    knob: the current value, the architecture default, the value recommended for
    this project's dataset and initialization, and one line each on effect and
    rationale.

    Recommendations depend on more than the row count: a corpus with many clips
    but few independent recording groups is redundant, and a scratch encoder
    needs the opposite regularization schedule from a short pretrained
    fine-tune. Resolving here rather than in the page keeps the bands in one
    place, the way ``sed.train.recommend`` owns the training hyperparameters.
    """
    values = dict(params or {})
    ctx = guidance_context(dataset, train_rows=train_rows, init=init)
    scope = _scope_phrase(ctx)
    rows: list[dict[str, Any]] = []
    for param in spec.params or ():
        name = str(param.get("name", ""))
        if not name:
            continue
        default = param.get("default")
        value = values.get(name, default)
        recommended, why = _recommended_param_value(param, ctx)
        rows.append({
            "name": name,
            "value": value,
            "default": default,
            "optional": bool(param.get("optional", False)),
            "enabled": name in values or not bool(param.get("optional", False)),
            "recommended": recommended,
            "why": why,
            "scope": scope,
            "effect": str(param.get("help") or ""),
            "mechanism": str(param.get("mechanism") or ""),
            "location": str(param.get("location") or ""),
            "overridden": name in values and values[name] != default,
            "off_recommendation": recommended is not None and value != recommended,
        })
    return rows


def weight_by_id(architecture_id: str, weight_id: str, project_id: str = "") -> WeightSpec | None:
    if not weight_id:
        return None
    for weight in architecture_by_id(architecture_id, project_id).weights:
        if weight.id == weight_id:
            return weight
    raise KeyError(f"weight {weight_id!r} is not available for {architecture_id!r}")


_LOCKED_CONSTRUCTOR_KEYS = frozenset({
    "pretrained",
    "num_classes",
    "in_chans",
    "global_pool",
    "pretrained_cfg",
})


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _coerce_param_value(field: dict[str, Any], value: Any) -> Any:
    kind = field.get("type", "float")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{field.get('name', 'value')} must be an integer")
        parsed = int(value)
        if float(value) != float(parsed):
            raise ValueError(f"{field.get('name', 'value')} must be an integer")
        return parsed
    if kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"{field.get('name', 'value')} must be a number")
        return float(value)
    if kind == "bool":
        return _coerce_bool(value)
    if kind == "choice":
        for choice in field.get("choices") or ():
            if value == choice or str(value) == str(choice):
                return choice
    if kind == "string":
        return str(value)
    if kind == "int_list":
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                value = json.loads(text)
            else:
                value = [part.strip() for part in text.split(",") if part.strip()]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field.get('name', 'value')} must be a list of integers")
        parsed: list[int] = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError(f"{field.get('name', 'value')} must contain integers only")
            integer = int(item)
            if float(item) != float(integer):
                raise ValueError(f"{field.get('name', 'value')} must contain integers only")
            parsed.append(integer)
        return parsed
    return value


def validate_recipe_params(architecture: ArchitectureSpec, params: dict[str, Any]) -> dict[str, Any]:
    """Validate the declared constructor contract and reject every extra key."""
    if not isinstance(params, dict):
        raise ValueError("recipe parameters must be an object")
    if architecture.id == "clink-cnn-v2" and params:
        raise ValueError("ClinkCNN-v2 is a fixed architecture and accepts no constructor parameters")
    locked = sorted(k for k in params if k in _LOCKED_CONSTRUCTOR_KEYS)
    if locked:
        raise ValueError(
            "constructor parameter "
            + ", ".join(locked)
            + " is locked by the Model Lab"
        )
    fields = {str(field.get("name") or ""): field for field in architecture.params}
    allowed = {name for name in fields if name}
    unknown = sorted(str(k) for k in params if str(k) not in allowed)
    if unknown:
        raise ValueError(
            "unsupported constructor parameter(s) for "
            f"{architecture.label}: {', '.join(unknown)}. "
            "Use only the declared Model-tab controls."
        )
    out = {
        str(field["name"]): field.get("default")
        for field in architecture.params
        if field.get("default") is not None and not bool(field.get("optional", False))
    }
    out.update(dict(params))
    for param_spec in architecture.params:
        name = str(param_spec["name"])
        if name not in out:
            continue
        value = _coerce_param_value(param_spec, out[name])
        if param_spec.get("choices") is not None and value not in param_spec["choices"]:
            raise ValueError(f"{name} must be one of {param_spec['choices']}")
        if param_spec.get("min") is not None and value < param_spec["min"]:
            raise ValueError(f"{name} must be >= {param_spec['min']}")
        if param_spec.get("max") is not None and value > param_spec["max"]:
            raise ValueError(f"{name} must be <= {param_spec['max']}")
        if param_spec.get("type") == "int_list":
            if param_spec.get("length") is not None and len(value) != int(param_spec["length"]):
                raise ValueError(
                    f"{name} must contain exactly {int(param_spec['length'])} integers"
                )
            item_choices = param_spec.get("item_choices")
            if item_choices is not None and any(item not in item_choices for item in value):
                raise ValueError(f"{name} values must be one of {item_choices}")
            if param_spec.get("item_min") is not None and any(
                item < param_spec["item_min"] for item in value
            ):
                raise ValueError(f"{name} values must be >= {param_spec['item_min']}")
            if param_spec.get("item_max") is not None and any(
                item > param_spec["item_max"] for item in value
            ):
                raise ValueError(f"{name} values must be <= {param_spec['item_max']}")
        out[name] = value
    ordered = {
        str(field["name"]): out[str(field["name"])]
        for field in architecture.params
        if str(field["name"]) in out
    }
    for key, value in out.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _contract_as_feature(contract: dict[str, Any]) -> dict[str, Any]:
    """Flatten a per-weight contract into the feature-dict shape the UI uses."""
    return {
        "feature_type": contract.get("features.feature_type", "mel"),
        "frontend": contract.get("features.frontend", "efficientat"),
        "sample_rate": contract.get("model.sample_rate", 32000),
        "n_fft": contract.get("features.n_fft"),
        "win_length": contract.get("features.win_length"),
        "hop_length": contract.get("features.hop_length"),
        "n_mels": contract.get("features.n_mels"),
        "fmin": contract.get("features.fmin"),
        "fmax": contract.get("features.fmax"),
        "window_seconds": contract.get("features.window_seconds", 1.0),
        "pcen": False, "log_power": True, "stft_engine": "librosa",
        "db_norm": False, "colormap": "none", "image_size": 96,
        "standardize": bool(contract.get("features.standardize", False)),
        "standardize_mean": float(contract.get("features.standardize_mean", 0.0)),
        "standardize_std": float(contract.get("features.standardize_std", 1.0)),
        # A checkpoint contract may pin its own dataset scaling (an external
        # encoder's fitted constants); absent keys mean plainly none.
        "scaling_method": str(contract.get("features.scaling_method", "none")),
        "scaling_scope": str(contract.get("features.scaling_scope", "global")),
        "scaling_stats": dict(contract.get("features.scaling_stats", {}) or {}),
    }


def recommended_feature_for(spec: ArchitectureSpec) -> dict[str, Any] | None:
    """Recommended starting feature geometry for an architecture.

    Shown for scratch (random) initialization, where no checkpoint contract
    applies: a starting point, never a constraint — any geometry trains.
    """
    if spec.provider == "efficientat":
        base = efficientat_base_backbone(spec.source)
        for weight in spec.weights:
            if weight.label == base and weight.expected_feature_contract:
                feature = _contract_as_feature(weight.expected_feature_contract)
                upstream_window = float(feature.get("window_seconds") or 1.0)
                feature["window_seconds"] = 0.4
                feature["upstream_window_seconds"] = upstream_window
                feature["source"] = (
                    f"base weight spectral/value contract ({base}); "
                    "0.4 s task-native clip window"
                )
                return feature
        feature = _contract_as_feature(_contract())
        feature["upstream_window_seconds"] = float(feature.get("window_seconds") or 1.0)
        feature["window_seconds"] = 0.4
        feature["source"] = "EfficientAT spectral/value contract; 0.4 s task-native clip window"
        return feature
    if spec.default_feature:
        return dict(spec.default_feature)
    if spec.id == "legacy-smallervggnet":
        from sed.data.presets import builtin_preset

        preset = builtin_preset("legacy-sounddetection")
        if preset:
            keys = ("feature_type", "sample_rate", "n_fft", "win_length", "hop_length",
                    "n_mels", "fmin", "fmax", "window_seconds", "log_power", "stft_engine",
                    "db_norm", "db_min", "db_max", "colormap", "image_size")
            feature = {k: preset[k] for k in keys if k in preset}
            feature["frontend"] = preset.get("frontend", "efficientat")
            # Explicit off: this legacy dB-image feature is foreign to pcen and
            # standardize. The dict can be merged onto a project's existing
            # [features] (Model tab "Verify"), so leaving these keys out would
            # let a stale standardize=True from a prior linear preset survive
            # the merge alongside db_norm=True — an invalid FeaturesCfg combo.
            feature["pcen"] = False
            feature["standardize"] = False
            feature["standardize_mean"] = 0.0
            feature["standardize_std"] = 1.0
            feature["scaling_method"] = "none"
            feature["scaling_scope"] = "global"
            feature["scaling_stats"] = {}
            feature["standardize_stats_resolved"] = True
            feature["standardize_stats_dataset_sha256"] = ""
            feature["standardize_stats_split_seed"] = 0
            feature["standardize_stats_val_fraction"] = 0.0
            feature["standardize_stats_clips"] = 0
            feature["source"] = "legacy-sounddetection preset"
            return feature
        return None
    # timm / kws / custom: a sensible general-purpose starting geometry.
    feature = _contract_as_feature(_contract())
    feature["source"] = "generic mel-128 default"
    return feature


def constructor_contract(spec: ArchitectureSpec) -> dict[str, Any]:
    """Serializable declared-only constructor contract for the Model tab."""
    if spec.id == "clink-cnn-v2":
        standard = "fixed ClinkCNN-v2 encoder"
        note = "This topology has no constructor variants; architecture changes need a new model ID."
    elif spec.provider == "efficientat":
        standard = "EfficientAT encoder arguments"
        note = (
            "Published checkpoints are locked to their original architecture. "
            "These optional controls are available only for scratch initialization."
        )
    elif spec.provider == "timm":
        standard = "declared timm.create_model arguments"
        note = "Only the documented encoder arguments below are supported."
    elif spec.source.startswith("project:"):
        standard = "project recipe constructor arguments"
        note = "Supported keys must be declared in the project's recipe.json."
    else:
        standard = "declared encoder constructor arguments"
        note = "Only catalog-declared keys are passed to the encoder."
    return {
        "standard": standard,
        "locked": sorted(_LOCKED_CONSTRUCTOR_KEYS),
        "fixed": not bool(spec.params),
        "note": note,
        "params": [dict(param) for param in spec.params],
        # Training-only mechanisms the architecture fixes in its own code. They
        # are not constructor arguments — the Model tab shows them read-only so
        # a model whose only dropout is fixed (ClinkCNN-v2) does not look like
        # it has none at all.
        "regularization": [dict(item) for item in spec.regularization],
    }


def catalog_payload(project_id: str = "", keep_ids: set[str] | None = None,
                    train_rows: int = 0,
                    dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model-tab catalog. Archived architectures are hidden from selection but
    pass through (badged) when the project's current recipe still uses one, so
    old configs keep rendering. `keep_ids` names those pass-through ids.

    ``dataset`` (the inspect server's ``_dataset_shape``) and ``train_rows``
    resolve each architecture's ``guidance`` — the per-knob recommendation for
    *this* project. Because the initialization mode is a live dropdown on the
    page, every mode is resolved here into ``guidance_by_init`` so flipping it
    needs no refetch; ``guidance`` stays as the pretrained-mode alias for
    callers that do not care."""
    keep = {str(k) for k in (keep_ids or set()) if k}
    shape = dict(dataset or {})
    if train_rows and not shape.get("rows"):
        shape["rows"] = int(train_rows)
    architectures: list[dict[str, Any]] = []
    for spec in all_architectures(project_id):
        if spec.archived and spec.id not in keep:
            continue
        payload = spec.to_payload()
        if spec.archived and "archived" not in payload["tags"]:
            payload["tags"].append("archived")
        payload["constructor"] = constructor_contract(spec)
        by_init = {
            mode: param_guidance(spec, {}, train_rows, dataset=shape, init=mode)
            for mode in INIT_MODES
        }
        payload["guidance_by_init"] = by_init
        payload["guidance"] = by_init["upstream"]
        if not payload.get("default_feature"):
            payload["default_feature"] = recommended_feature_for(spec)
        architectures.append(payload)
    from sed.config import BACKBONES

    # Architecture-only widths with no published checkpoint (mn01…): reachable
    # from the Model tab's variant dropdown under random initialization.
    scratch_backbones = [
        {"name": name, "width": meta["width"], "feat_dim": meta["feat_dim"]}
        for name, meta in BACKBONES.items()
        if not meta["pretrained_available"]
    ]
    return {
        "architectures": architectures,
        "tag_definitions": CATALOG_TAG_DEFINITIONS,
        "providers": ["efficientat", "timm", "local"],
        "trusted_timm_prefixes": ["hf-hub:timm/"],
        "scratch_backbones": scratch_backbones,
        "train_rows": int(shape.get("rows") or train_rows or 0),
        "dataset_shape": shape,
    }


def is_verified_timm_source(source: str) -> bool:
    source = str(source or "").strip()
    if any(spec.source == source for spec in curated_architectures() if spec.provider == "timm"):
        return True
    return source.startswith("hf-hub:timm/")


def recipe_signature(recipe: ModelRecipe | dict[str, Any]) -> str:
    data = recipe.to_payload() if isinstance(recipe, ModelRecipe) else dict(recipe)
    for key in ("weight_id", "initialization", "parent_project", "parent_run"):
        data.pop(key, None)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cached_weight_provenance(weight: WeightSpec | None) -> dict[str, Any] | None:
    """Return immutable local-cache evidence once an upstream weight is loaded."""
    if weight is None:
        return None
    payload = weight.to_payload()
    filename = weight.source_url.rsplit("/", 1)[-1]
    candidates = [
        REPO_ROOT / "vendor" / "EfficientAT" / "resources" / filename,
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            payload["cached_path"] = str(path)
            payload["cached_sha256"] = sha256_file(path)
            break
    return payload


def _infer_architecture_id(backbone: str, recipe_id: str, project_id: str = "") -> str:
    """Resolve the catalog architecture for a backbone (shared by the live
    recipe card and frozen training_config hashing, so both agree)."""
    architecture_id = str(recipe_id or "")
    if not architecture_id:
        candidates = all_architectures(project_id)
        for candidate in candidates:
            if candidate.source == backbone or candidate.id == backbone:
                architecture_id = candidate.id
                break
        else:
            # Legacy configs recorded only the backbone name: map EfficientAT
            # width/checkpoint variants (mn20_as, dymn04_as, mn10_as_mels_256…)
            # onto their catalog family so live and frozen signatures agree.
            family_id = "efficientat-dymn" if backbone.startswith("dymn") else "efficientat-mn"
            if _EFFICIENTAT_WIDTH_RE.match(backbone):
                for candidate in candidates:
                    if candidate.id == family_id:
                        architecture_id = candidate.id
                        break
    return architecture_id or f"legacy:{backbone}"


def recipe_from_config(cfg: Any) -> ModelRecipe:
    model = cfg.model
    architecture_id = _infer_architecture_id(
        str(model.backbone), str(model.recipe_id or ""),
        getattr(getattr(cfg, "project", None), "id", ""),
    )
    return ModelRecipe(
        schema_version=int(getattr(model, "params_contract_version", 1) or 1),
        architecture_id=architecture_id, backbone=str(model.backbone),
        task=str(model.task), weight_id=str(model.weight_id), initialization=str(model.initialization),
        params=dict(model.params), parent_project=str(model.parent_project), parent_run=str(model.parent_run),
    )


def recipe_signature_from_payload(model: dict[str, Any], project_id: str = "") -> str:
    """``recipe_signature`` for a frozen training_config.json ``model`` block."""
    backbone = str(model.get("backbone") or "")
    return recipe_signature({
        "schema_version": int(model.get("params_contract_version") or 1),
        "architecture_id": _infer_architecture_id(
            backbone, str(model.get("recipe_id") or ""), project_id,
        ),
        "backbone": backbone,
        "task": str(model.get("task") or "clip"),
        "params": dict(model.get("params") or {}),
    })


def validate_config_recipe_params(cfg: Any) -> Any:
    """Strictly validate params before any new train/estimate pipeline starts.

    Frozen legacy bundles remain loadable for evaluation/inference because this
    preflight is called only by new-training paths.  A legacy active config with
    undeclared params gets an actionable reset/reselect error instead of writing
    another recipe whose signature may not match the model that was built.
    """
    project_id = str(getattr(getattr(cfg, "project", None), "id", "") or "")
    recipe = recipe_from_config(cfg)
    try:
        architecture = architecture_by_id(recipe.architecture_id, project_id)
    except KeyError as exc:
        if recipe.params:
            raise ValueError(
                "model.params belong to a legacy or unknown architecture. "
                "Reselect the model recipe and reset its constructor parameters before training."
            ) from exc
        return cfg
    normalized = validate_recipe_params(architecture, recipe.params)
    if (
        architecture.provider == "efficientat"
        and str(cfg.model.initialization or "upstream") == "upstream"
        and normalized
    ):
        raise ValueError(
            "EfficientAT pretrained checkpoints use their published architecture defaults; "
            "switch to scratch initialization before changing constructor parameters"
        )
    if normalized == dict(cfg.model.params):
        return cfg
    return cfg.model_copy(update={
        "model": cfg.model.model_copy(update={"params": normalized}),
    })


def effective_head_dropout(cfg: Any) -> float:
    """Resolve automatic classifier dropout against the target train depth."""
    configured = float(getattr(cfg.training, "head_dropout", 0.2) or 0.0)
    if not bool(getattr(cfg.training, "head_dropout_auto", False)):
        return configured
    if str(getattr(cfg.training, "pretraining_objective", "supervised")) == "byol":
        # BYOL ignores the supervised depth selector and trains the encoder.
        target_depth = "full"
    else:
        try:
            from sed.train.fine_tuning import resolve_fine_tune_plan

            target_depth = resolve_fine_tune_plan(cfg).target_depth
        except Exception:  # noqa: BLE001 - safe default for advisory/config paths
            target_depth = str(
                getattr(cfg.training, "fine_tune_policy", "full")
                or getattr(cfg.training, "train_depth", "full")
                or "full"
            )
    project_id = str(getattr(getattr(cfg, "project", None), "id", "") or "")
    architecture_id = _infer_architecture_id(
        str(cfg.model.backbone), str(cfg.model.recipe_id or ""), project_id,
    )
    info = architecture_dropout_info(
        architecture_id, dict(cfg.model.params or {}), project_id,
        train_depth=target_depth,
    )
    return float(info.get("recommended_head_dropout", 0.2))


def materialize_training_config(cfg: Any) -> Any:
    """Return the strict, numeric config that a new run/estimate must build."""
    cfg = validate_config_recipe_params(cfg)
    value = effective_head_dropout(cfg)
    if not bool(getattr(cfg.training, "head_dropout_auto", False)) and value == float(cfg.training.head_dropout):
        return cfg
    return cfg.model_copy(update={
        "training": cfg.training.model_copy(update={
            "head_dropout": value,
            "head_dropout_auto": False,
        }),
    })


def snapshot_project_architecture(project_id: str, recipe: ModelRecipe, destination: Path) -> str:
    """Snapshot project-local architecture code into a model bundle if used."""
    if not recipe.architecture_id.startswith(f"project:{project_id}:"):
        return ""
    slug = recipe.architecture_id.rsplit(":", 1)[-1]
    source = PROJECTS_DIR / project_id / "architectures" / slug
    if not source.exists():
        raise FileNotFoundError(f"project architecture source is missing: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    digest = hashlib.sha256()
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(destination).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "ArchitectureSpec", "WeightSpec", "ModelRecipe", "CATALOG_TAG_DEFINITIONS", "catalog_payload", "curated_architectures",
    "all_architectures", "architecture_by_id", "project_architectures", "weight_by_id", "is_verified_timm_source", "recipe_from_config",
    "recipe_signature", "snapshot_project_architecture", "efficientat_base_backbone",
    "validate_recipe_params", "validate_config_recipe_params",
    "effective_head_dropout", "materialize_training_config",
    "architecture_dropout_info", "constructor_contract", "cached_weight_provenance",
    "param_guidance", "guidance_context", "INIT_MODES",
]
