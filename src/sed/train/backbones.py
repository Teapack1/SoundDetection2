"""Pluggable backbone architectures for the SED model.

The trainer is no longer tied to EfficientAT. A *backbone adapter* knows how to:

* ``build_backbone`` — construct the feature-extractor ``nn.Module`` (no head),
* ``forward_features`` — run it and return a pooled ``(B, D)`` embedding,
* ``stages`` — return the ordered stem→…→last param groups, which drives both
  block-level freezing ("train depth") and per-stage learning rates.

Three families ship in the box:

* ``efficientat`` — the vendored MobileNet (mn*) backbones, AudioSet-pretrained.
* ``timm``        — anything ``timm.create_model`` can build (ConvNeXt-Tiny,
  MobileNetV3, …), with ``in_chans=1`` and global-avg pooling.
* ``custom``      — a user ``module:Class`` path implementing ``nn.Module`` whose
  ``forward`` returns either a ``(B, D)`` embedding or a ``(B, C, H, W)`` map we
  pool ourselves. The escape hatch for "my own net".

User-registered architectures are persisted as JSON specs under
``presets/models/<name>.json`` so the Model tab can list/select them across
sessions; built-in specs come from :data:`sed.config.BACKBONES`.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import os
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sed.config import (
    BACKBONES,
    CLINK_CNN_V2_FEATURE_CONTRACT,
    EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT,
    MN_DOWNSAMPLE,
    MN_INPUT_CHANNELS,
    MN_NATIVE_BANDS,
    MN_NATIVE_FRAMES,
    PROJECTS_DIR,
    REPO_ROOT,
)

log = logging.getLogger(__name__)

EFFICIENTAT_PATH = REPO_ROOT / "vendor" / "EfficientAT"
USER_MODEL_DIR = REPO_ROOT / "presets" / "models"


# ---------------------------------------------------------------------------
# Spec dataclasses (also the JSON shape persisted under presets/models/)
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    """One configurable backbone knob exposed to the Model tab."""

    name: str
    type: str = "float"  # float | int | bool | choice | string | int_list
    default: Any = None
    choices: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    optional: bool = False
    length: int | None = None
    item_choices: list[Any] | None = None
    item_min: float | None = None
    item_max: float | None = None
    help: str = ""
    mechanism: str = ""
    location: str = ""
    # Dataset-size-banded recommendations, ascending by ``max_train_rows``
    # (``None`` = open-ended top band). Resolved by
    # ``sed.train.model_lab.param_guidance`` for the Model tab readout.
    recommended: list[dict[str, Any]] | None = None


@dataclass
class BackboneSpec:
    name: str
    family: str  # "efficientat" | "timm" | "custom"
    pretrained_available: bool = False
    feat_dim: int | None = None  # None ⇒ inferred by a dummy forward
    input_contract: dict[str, Any] | None = None
    downsample: int = 32
    native_bands: int = MN_NATIVE_BANDS
    native_frames: int = MN_NATIVE_FRAMES
    input_channels: int = MN_INPUT_CHANNELS
    params: list[ParamSpec] = field(default_factory=list)
    source: str = ""  # timm model name or "module:Class" for custom
    description: str = ""
    builtin: bool = False
    provider: str = ""
    task_capabilities: list[str] = field(default_factory=lambda: ["clip"])
    export_supported: bool = True

    def to_payload(self) -> dict[str, Any]:
        data = asdict(self)
        # ParamSpec already serialized by asdict; ensure JSON-clean.
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackboneSpec":
        params = [ParamSpec(**p) if isinstance(p, dict) else p
                  for p in (data.get("params") or [])]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "params"}
        return cls(params=params, **kwargs)


# ---------------------------------------------------------------------------
# EfficientAT loader (moved here from train.model; re-exported there)
# ---------------------------------------------------------------------------


@contextmanager
def _chdir(path: Path):
    """Temporarily change CWD so EfficientAT's import-time relative file reads
    (``metadata/class_labels_indices.csv``) resolve."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def is_dymn(backbone_name: str) -> bool:
    """Whether ``backbone_name`` is a Dynamic MobileNet (``dymn*``) variant."""
    return str(backbone_name).startswith("dymn")


# Mirrors the vendored ``helpers.utils.NAME_TO_WIDTH`` for both families, without
# importing that module (its import-time CSV read is CWD-fragile). The width is
# keyed by the family prefix, so checkpoint suffixes (``mn10_as_mels_256``,
# ``dymn10_as``) all resolve to the right multiplier.
_MN_WIDTHS = {
    "mn01": 0.1, "mn02": 0.2, "mn04": 0.4, "mn05": 0.5, "mn06": 0.6, "mn08": 0.8,
    "mn10": 1.0, "mn12": 1.2, "mn14": 1.4, "mn16": 1.6, "mn20": 2.0, "mn30": 3.0, "mn40": 4.0,
}
_DYMN_WIDTHS = {"dymn04": 0.4, "dymn10": 1.0, "dymn20": 2.0}

# Constructor kwargs each family's ``get_model`` accepts. The Model catalog
# exposes this declared subset and the adapter rejects everything else.
_MN_KWARGS_DOC = {
    "reduced_tail": {"default": False, "help": "If True, use reduced tail (less channels in the final layers)"},
    "dilated": {"default": False, "help": "If True, use dilated convolutions for the tail"},
    "strides": {"default": [2, 2, 2, 2], "help": "List of 4 integers for per-stage downsampling (e.g. [2, 2, 2, 2])"},
    "input_dim_f": {"default": 128, "help": "Input frequency dimension (mel bands)"},
    "input_dim_t": {"default": 1000, "help": "Input time dimension (frames)"},
    "se_dims": {"default": "c", "options": ["c", "f", "t", "cf", "ct", "ft", "cft"], "help": "Squeeze-and-excitation dimensions"},
    "se_agg": {"default": "max", "options": ["max", "avg"], "help": "Squeeze-and-excitation aggregation type"},
    "se_r": {"default": 4, "help": "Squeeze-and-excitation reduction ratio"},
}

_DYMN_KWARGS_DOC = {
    "strides": {"default": [2, 2, 2, 2], "help": "List of 4 integers for per-stage downsampling"},
    "context_ratio": {"default": 4, "help": "Channel divisor for the DyMN context size (channels // ratio)"},
    "max_context_size": {"default": 128, "help": "Max context size"},
    "min_context_size": {"default": 32, "help": "Min context size"},
    "dyrelu_k": {"default": 2, "help": "Dynamic ReLU K parameter"},
    "no_dyrelu": {"default": False, "help": "If True, disable Dynamic ReLU"},
    "dyconv_k": {"default": 4, "help": "Dynamic Convolution K parameter"},
    "no_dyconv": {"default": False, "help": "If True, disable Dynamic Convolution"},
    "T_max": {"default": 30.0, "help": "Temperature max"},
    "T0_slope": {"default": 1.0, "help": "Temperature T0 slope"},
    "T1_slope": {"default": 0.02, "help": "Temperature T1 slope"},
    "T_min": {"default": 1.0, "help": "Temperature min"},
    "no_ca": {"default": False, "help": "If True, disable Coordinate Attention"},
    "use_dy_blocks": {"default": "all", "options": ["all", "replace_se"], "help": "Where to place dynamic blocks"},
}

_MN_ALLOWED_KWARGS = frozenset(_MN_KWARGS_DOC.keys())
_DYMN_ALLOWED_KWARGS = frozenset(_DYMN_KWARGS_DOC.keys())

def _efficientat_width(backbone_name: str) -> float:
    name = str(backbone_name)
    try:
        if name.startswith("dymn"):
            return _DYMN_WIDTHS[name[:6]]
        return _MN_WIDTHS[name[:4]]
    except KeyError:
        return 1.0


def _load_efficientat(backbone_name: str, pretrained: bool, **params: Any) -> nn.Module:
    """Load an ``mn*`` (MobileNet) or ``dymn*`` (Dynamic MobileNet) backbone from
    the vendored EfficientAT checkout."""
    if not EFFICIENTAT_PATH.exists():
        raise ImportError(
            f"EfficientAT not found at {EFFICIENTAT_PATH}. Run:\n"
            "  git submodule add https://github.com/fschmid56/EfficientAT "
            "vendor/EfficientAT && git submodule update --init"
        )
    metadata_csv = EFFICIENTAT_PATH / "metadata" / "class_labels_indices.csv"
    if not metadata_csv.exists():
        raise ImportError(
            f"EfficientAT submodule looks incomplete (missing {metadata_csv}). "
            "Try: git submodule update --init --recursive"
        )
    if str(EFFICIENTAT_PATH) not in sys.path:
        sys.path.insert(0, str(EFFICIENTAT_PATH))

    dymn = is_dymn(backbone_name)
    with _chdir(EFFICIENTAT_PATH):
        if dymn:
            from models.dymn.model import get_model  # type: ignore
        else:
            from models.mn.model import get_model  # type: ignore

    stdout = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Don't use ConvNormActivation directly.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore", message=r"Don't use Conv2dNormActivation directly.*",
            category=UserWarning,
        )
        with redirect_stdout(stdout):
            allowed = _DYMN_ALLOWED_KWARGS if dymn else _MN_ALLOWED_KWARGS
            unknown = sorted(k for k, value in params.items() if value is not None and k not in allowed)
            if unknown:
                raise ValueError(
                    f"unsupported {'DyMN' if dymn else 'EfficientAT'} constructor "
                    f"parameter(s): {', '.join(unknown)}"
                )
            kwargs = {k: v for k, v in params.items() if k in allowed and v is not None}
            model = get_model(
                width_mult=_efficientat_width(backbone_name),
                pretrained_name=backbone_name if pretrained else None,
                **kwargs,
            )
    printed = stdout.getvalue().strip()
    if printed:
        log.debug("suppressed EfficientAT stdout during model load (%d chars)", len(printed))
    return model


def _requested_input_channels(input_channels: int | None, default: int) -> int:
    channels = int(input_channels if input_channels is not None else default)
    if channels <= 0:
        raise ValueError("model input channels must be positive")
    return channels


def _replace_conv2d_input_channels(conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    if conv.groups != 1:
        raise ValueError("cannot adapt grouped stem convolution to a different input channel count")
    if int(conv.in_channels) == int(input_channels):
        return conv
    new_conv = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)
    return new_conv


def _adapt_efficientat_input_channels(model: nn.Module, input_channels: int) -> None:
    """Rebuild the EfficientAT scratch stem for non-1-channel features."""
    if input_channels == MN_INPUT_CHANNELS:
        return
    in_c = getattr(model, "in_c", None)
    if isinstance(in_c, nn.Sequential) and isinstance(in_c[0], nn.Conv2d):
        in_c[0] = _replace_conv2d_input_channels(in_c[0], input_channels)
        return
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential):
        first = features[0] if len(features) else None
        if isinstance(first, nn.Sequential) and isinstance(first[0], nn.Conv2d):
            first[0] = _replace_conv2d_input_channels(first[0], input_channels)
            return
        if isinstance(first, nn.Conv2d):
            features[0] = _replace_conv2d_input_channels(first, input_channels)
            return
    raise ValueError("could not locate the EfficientAT stem convolution to adapt input channels")


# ---------------------------------------------------------------------------
# Generic stage extraction (shared by timm / custom adapters)
# ---------------------------------------------------------------------------


def _ordered_stage_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Best-effort ordered (name, module) stem→…→last grouping for freezing.

    Tries the common layouts in priority order: ConvNeXt (``stem`` + ``stages``),
    MobileNet/EfficientNet (``conv_stem`` + ``blocks``), torchvision
    (``features``), else top-level ``named_children``.
    """
    explicit = getattr(model, "fine_tune_stages", None)
    if callable(explicit):
        groups = list(explicit())
        if not groups or any(not isinstance(name, str) or not isinstance(module, nn.Module)
                             for name, module in groups):
            raise ValueError("fine_tune_stages() must return non-empty (name, module) pairs")
        return groups

    groups: list[tuple[str, nn.Module]] = []
    if hasattr(model, "stem") and hasattr(model, "stages"):
        groups.append(("stem", model.stem))
        for i, stage in enumerate(model.stages):  # type: ignore[arg-type]
            groups.append((f"stage{i}", stage))
        return groups
    # Vendored DyMN: in_c (stem) → layers (ModuleList of blocks) → out_c (tail).
    # Without this each block would collapse into one giant ``layers`` group, so
    # block-level freezing / layer-wise LR would have almost no granularity.
    if isinstance(getattr(model, "layers", None), nn.ModuleList):
        if isinstance(getattr(model, "in_c", None), nn.Module):
            groups.append(("in_c", model.in_c))
        for i, layer in enumerate(model.layers):  # type: ignore[arg-type]
            groups.append((f"layer{i}", layer))
        if isinstance(getattr(model, "out_c", None), nn.Module):
            groups.append(("out_c", model.out_c))
        return groups
    if hasattr(model, "blocks"):
        stem_mods = [getattr(model, a) for a in ("conv_stem", "bn1", "stem")
                     if hasattr(model, a) and isinstance(getattr(model, a), nn.Module)]
        if stem_mods:
            groups.append(("stem", nn.ModuleList(stem_mods)))
        for i, block in enumerate(model.blocks):  # type: ignore[arg-type]
            groups.append((f"block{i}", block))
        return groups
    if hasattr(model, "features") and isinstance(model.features, nn.Module):
        children = list(model.features.children())
        if children:
            for i, child in enumerate(children):
                groups.append((f"features{i}", child))
            return groups
    for name, child in model.named_children():
        groups.append((name, child))
    if not groups and list(model.parameters(recurse=False)):
        groups.append(("backbone", model))
    return groups


def _stage_param_groups(model: nn.Module) -> list[tuple[str, list[nn.Parameter]]]:
    """Ordered (name, params) groups covering ALL backbone params.

    Any parameter not captured by an ``_ordered_stage_modules`` group (e.g. a
    trailing norm) is appended to the LAST group so it trains whenever the final
    stage does and freezes when the whole backbone is frozen.
    """
    modules = _ordered_stage_modules(model)
    groups: list[tuple[str, list[nn.Parameter]]] = []
    seen: set[int] = set()
    for name, mod in modules:
        params = [p for p in mod.parameters() if id(p) not in seen]
        for p in params:
            seen.add(id(p))
        if params:
            groups.append((name, params))
    leftover = [p for p in model.parameters() if id(p) not in seen]
    if leftover:
        if groups:
            groups[-1] = (groups[-1][0], groups[-1][1] + leftover)
        else:
            groups.append(("backbone", leftover))
    return groups


def _pool_features(out: Any) -> torch.Tensor:
    """Reduce a backbone output to a ``(B, D)`` embedding."""
    if isinstance(out, (tuple, list)):
        out = out[-1]
    if out.dim() == 4:
        return torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), start_dim=1)
    if out.dim() == 3:
        return out.mean(dim=-1)
    return out


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class BackboneAdapter:
    """Base adapter. Subclasses build a backbone and know its stages."""

    def __init__(self, spec: BackboneSpec):
        self.spec = spec

    def build_backbone(self, *, pretrained: bool, n_bands: int, n_frames: int,
                       input_channels: int | None = None,
                       **params: Any) -> nn.Module:
        raise NotImplementedError

    def forward_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return _pool_features(backbone(x))

    def forward_frame_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, D, T)`` features for optional frame-level SED.

        Adapters that expose a feature map inherit a reliable frequency mean.
        A plain pooled-only encoder gets a single frame and is intentionally not
        advertised as frame-capable by the catalog.
        """
        if hasattr(backbone, "forward_features"):
            out = backbone.forward_features(x)
        else:
            out = backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.dim() == 4:
            return out.mean(dim=2)
        if out.dim() == 3:
            # timm transformers commonly return (B, tokens, channels).
            return out.transpose(1, 2)
        if out.dim() == 2:
            return out.unsqueeze(-1)
        raise ValueError(f"unsupported frame feature rank {out.dim()}")

    def stages(self, backbone: nn.Module) -> list[tuple[str, list[nn.Parameter]]]:
        return _stage_param_groups(backbone)

    def stage_modules(self, backbone: nn.Module) -> list[tuple[str, nn.Module]]:
        return _ordered_stage_modules(backbone)

    def feature_map_size(self, backbone: nn.Module, x: torch.Tensor) -> tuple[int, int] | None:
        """Spatial (H, W) of the pre-pool feature map, or None if not 4D."""
        try:
            if hasattr(backbone, "forward_features"):
                out = backbone.forward_features(x)
            else:
                out = backbone(x)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.dim() == 4:
            return int(out.shape[-2]), int(out.shape[-1])
        if out.dim() == 3:  # token sequence (transformer) — no 2D map
            return int(out.shape[1]), 1
        return None


class EfficientATAdapter(BackboneAdapter):
    family = "efficientat"

    def build_backbone(self, *, pretrained, n_bands, n_frames, input_channels=None, **params):
        channels = _requested_input_channels(input_channels, int(self.spec.input_channels or 1))
        load_pretrained = bool(pretrained and self.spec.pretrained_available)
        if channels != MN_INPUT_CHANNELS and load_pretrained:
            raise ValueError(
                "EfficientAT pretrained weights require a 1-channel spectrogram input. "
                "Use a non-colormap feature, or switch initialization to scratch/random "
                f"to train a {channels}-channel EfficientAT stem."
            )
        model = _load_efficientat(self.spec.source or self.spec.name, load_pretrained, **params)
        _adapt_efficientat_input_channels(model, channels)
        return model

    @staticmethod
    def _feature_map(backbone, x):
        """The pre-pool ``(B, C, F, T)`` map for both families.

        DyMN exposes ``_feature_forward`` (it has no ``features`` Sequential);
        MN exposes ``features``. Keying off the wrong one collapsed DyMN to a
        single frame and hid its feature map — the bug this unifies away.
        """
        if hasattr(backbone, "_feature_forward"):
            return backbone._feature_forward(x)
        features = getattr(backbone, "features", None)
        if features is not None:
            return features(x)
        return None

    def forward_features(self, backbone, x):
        fmap = self._feature_map(backbone, x)
        if fmap is not None:
            return torch.flatten(F.adaptive_avg_pool2d(fmap, (1, 1)), start_dim=1)
        out = backbone(x)
        if isinstance(out, tuple):
            _logits, emb = out
            return emb
        return out

    def feature_map_size(self, backbone, x):
        fmap = self._feature_map(backbone, x)
        if fmap is not None:
            return int(fmap.shape[-2]), int(fmap.shape[-1])
        return super().feature_map_size(backbone, x)

    def forward_frame_features(self, backbone, x):
        fmap = self._feature_map(backbone, x)
        if fmap is not None:
            return fmap.mean(dim=2)
        return super().forward_frame_features(backbone, x)


class TimmAdapter(BackboneAdapter):
    family = "timm"

    def build_backbone(self, *, pretrained, n_bands, n_frames, input_channels=None, **params):
        import timm

        model_name = self.spec.source or self.spec.name
        channels = _requested_input_channels(input_channels, int(self.spec.input_channels or 1))
        kwargs: dict[str, Any] = dict(
            pretrained=bool(pretrained and self.spec.pretrained_available),
            num_classes=0,
            in_chans=channels,
            global_pool="avg",
        )
        # ``create_model`` is the canonical timm configuration interface. The
        # Model Lab build-probes its declared controls and never lets them
        # override the invariants that make this an encoder.
        protected = {"pretrained", "num_classes", "in_chans", "global_pool", "pretrained_cfg"}
        for key, value in params.items():
            if key not in protected and value is not None:
                kwargs[key] = value
        return timm.create_model(model_name, **kwargs)

    def forward_features(self, backbone, x):
        # num_classes=0 + global_pool="avg" ⇒ forward returns (B, D).
        return _pool_features(backbone(x))


class CustomAdapter(BackboneAdapter):
    family = "custom"

    def build_backbone(self, *, pretrained, n_bands, n_frames, input_channels=None, **params):
        channels = _requested_input_channels(input_channels, int(self.spec.input_channels or 1))
        if (self.spec.source or "").startswith("project:"):
            _, project_id, slug = self.spec.source.split(":", 2)
            path = PROJECTS_DIR / project_id / "architectures" / slug / "model.py"
            if not path.exists():
                raise FileNotFoundError(f"project architecture source is missing: {path}")
            module_name = f"sed_project_arch_{project_id}_{slug}"
            module_spec = importlib.util.spec_from_file_location(module_name, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"could not load project architecture from {path}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            cls = getattr(module, "CustomEncoder")
            return self._construct(cls, pretrained=pretrained, n_bands=n_bands,
                                   n_frames=n_frames, params=params,
                                   input_channels=channels)
        module_path, _, cls_name = (self.spec.source or "").partition(":")
        if not module_path or not cls_name:
            raise ValueError(
                f"custom backbone {self.spec.name!r} needs source 'module:Class', "
                f"got {self.spec.source!r}"
            )
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
        return self._construct(cls, pretrained=pretrained, n_bands=n_bands,
                               n_frames=n_frames, params=params,
                               input_channels=channels)

    def _construct(self, cls, *, pretrained, n_bands, n_frames, params, input_channels):
        # User parameters are a declared recipe contract and must never be
        # silently discarded.  Geometry/pretrained values are adapter hints,
        # however, so custom constructors may legitimately omit those names.
        import inspect

        signature = inspect.signature(cls)
        accepted = set(signature.parameters)
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        # A catch-all **kwargs cannot prove that a declared value is consumed;
        # requiring an explicit signature entry makes the build probe a real
        # end-to-end contract check instead of another silent sink.
        unsupported = sorted(k for k in params if k not in accepted)
        if unsupported:
            raise ValueError(
                "declared constructor parameter(s) not accepted by "
                f"{cls.__name__}: {', '.join(unsupported)}"
            )
        kwargs = dict(params)
        hints = {
            "in_chans": int(input_channels),
            "n_bands": n_bands,
            "n_frames": n_frames,
            "pretrained": bool(pretrained),
        }
        for key, value in hints.items():
            if accepts_kwargs or key in accepted:
                kwargs.setdefault(key, value)
        return cls(**kwargs)


_ADAPTERS: dict[str, type[BackboneAdapter]] = {
    "efficientat": EfficientATAdapter,
    "timm": TimmAdapter,
    "custom": CustomAdapter,
}


# ---------------------------------------------------------------------------
# Train-depth (block-level freezing)
# ---------------------------------------------------------------------------

def _train_depth_selection(
    model: nn.Module, train_depth: str,
) -> tuple[int, int, list[tuple[str, nn.Module]], list[tuple[str, nn.Module]]]:
    mods = model.backbone_stage_modules()  # type: ignore[attr-defined]
    n = len(mods)
    keep = select_trainable_stages(n, train_depth)
    frozen = mods[: n - keep] if keep < n else []
    trainable = mods[n - keep:] if keep > 0 else []
    return n, keep, frozen, trainable


def _wants_trainable_encoder(train_depth: str) -> bool:
    """Whether ``train_depth`` asks for ANY encoder training.

    ``keep`` alone cannot answer this when an adapter reports no stages: a
    backbone whose structure defeats stage discovery yields ``n = 0``, so even
    ``full`` computes ``keep = 0`` and would freeze the entire encoder while
    reporting a full fine-tune. Deciding from the request instead means such a
    model trains — the fallback is "train it", not "silently train only the head".
    """
    return select_trainable_stages(1, train_depth) > 0


def describe_train_depth(
    model: nn.Module, train_depth: str, *, freeze_batchnorm: bool = False,
) -> dict[str, Any]:
    """Return the trainability summary for ``train_depth`` without mutation."""
    n, keep, frozen, trainable = _train_depth_selection(model, train_depth)

    trainable_ids: set[int] = set()
    for _, mod in trainable:
        trainable_ids.update(id(p) for p in mod.parameters())

    # Backbone params not under any stage module follow the deepest stage.
    backbone = getattr(model, "backbone", model)
    stage_modules = model.backbone_stage_modules()  # type: ignore[attr-defined]
    covered = {id(p) for _, mod in stage_modules for p in mod.parameters()}
    if keep > 0 if n else _wants_trainable_encoder(train_depth):
        trainable_ids.update(id(p) for p in backbone.parameters() if id(p) not in covered)

    if freeze_batchnorm:
        for _, mod in trainable:
            for sub in mod.modules():
                if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    for p in sub.parameters():
                        trainable_ids.discard(id(p))

    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        trainable_ids.update(id(p) for p in head.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if id(p) in trainable_ids)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "train_depth": str(train_depth),
        "n_stages": n,
        "trainable_stages": [name for name, _ in trainable],
        "frozen_stages": [name for name, _ in frozen],
        "freeze_batchnorm": bool(freeze_batchnorm),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_fraction": (trainable_params / total_params) if total_params else 1.0,
    }


def select_trainable_stages(n_stages: int, train_depth: str) -> int:
    """How many trailing backbone stages stay trainable for ``train_depth``.

    ``head`` ⇒ 0 (freeze the whole backbone, train only the head),
    ``last_block`` ⇒ 1, ``last_n:<k>`` ⇒ k, ``half`` ⇒ the deeper half,
    ``full`` ⇒ all.
    """
    d = str(train_depth or "full").strip().lower()
    if d in ("full", "all"):
        return n_stages
    if d in ("head", "head_only", "none"):
        return 0
    if d in ("last_block", "last", "last1"):
        return min(1, n_stages)
    if d == "half":
        return n_stages - n_stages // 2
    if ":" in d or "=" in d:
        try:
            k = int(d.replace("=", ":").split(":")[-1])
        except ValueError:
            return n_stages
        return max(0, min(k, n_stages))
    if d.startswith("last") and d[4:].isdigit():
        return max(0, min(int(d[4:]), n_stages))
    return n_stages


def apply_train_depth(
    model: nn.Module, train_depth: str, *, freeze_batchnorm: bool = False
) -> tuple[dict[str, Any], list[nn.Module]]:
    """Freeze a leading prefix of the backbone per ``train_depth``.

    Returns ``(info, eval_modules)`` where ``info`` summarizes trainable vs
    frozen stages + param counts. ``eval_modules`` contains every frozen stage
    plus, when ``freeze_batchnorm`` is enabled, BatchNorm modules in trainable
    stages too. The caller reapplies ``eval()`` after Lightning's epoch-level
    ``train()`` reset so no running statistics drift. The classification head
    is always left trainable.
    """
    n, keep, frozen, trainable = _train_depth_selection(model, train_depth)
    mods = frozen + trainable
    # With no discoverable stages there is nothing to freeze a prefix of, so the
    # request itself decides whether the encoder trains at all.
    encoder_trainable = keep > 0 if n else _wants_trainable_encoder(train_depth)

    for _, mod in mods:
        for p in mod.parameters():
            p.requires_grad = False
    for _, mod in trainable:
        for p in mod.parameters():
            p.requires_grad = True

    # Backbone params not under any stage module follow the deepest stage.
    backbone = getattr(model, "backbone", model)
    covered = {id(p) for _, mod in mods for p in mod.parameters()}
    for p in backbone.parameters():
        if id(p) not in covered:
            p.requires_grad = encoder_trainable

    # Head-only training freezes the *entire* encoder, including parameterless
    # modules such as Dropout/DropPath that may sit outside the stage parameter
    # groups.  Keeping only parameter-bearing stages in eval mode would leave
    # pooling dropout active in an otherwise frozen backbone.
    eval_modules = [backbone] if not encoder_trainable else [mod for _, mod in frozen]
    if freeze_batchnorm:
        # Freeze both affine parameters and running-stat updates across the
        # whole encoder. Frozen stages are already present above; listing BN
        # modules explicitly also covers trainable stages and any parameters
        # an adapter could not map into its public stage list.
        for sub in backbone.modules():
            if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for p in sub.parameters():
                    p.requires_grad = False
                eval_modules.append(sub)

    for p in model.head.parameters():  # type: ignore[attr-defined]
        p.requires_grad = True

    info = describe_train_depth(
        model, train_depth, freeze_batchnorm=freeze_batchnorm,
    )
    return info, eval_modules


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _builtin_specs() -> dict[str, BackboneSpec]:
    specs: dict[str, BackboneSpec] = {}
    for name, meta in BACKBONES.items():
        specs[name] = BackboneSpec(
            name=name,
            family="efficientat",
            pretrained_available=bool(meta.get("pretrained_available")),
            feat_dim=int(meta.get("feat_dim")) if meta.get("feat_dim") else None,
            input_contract=dict(EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT)
            if meta.get("pretrained_available") else None,
            downsample=MN_DOWNSAMPLE,
            native_bands=MN_NATIVE_BANDS,
            native_frames=MN_NATIVE_FRAMES,
            input_channels=MN_INPUT_CHANNELS,
            source=name,
            description="EfficientAT MobileNet (AudioSet)" if meta.get("pretrained_available")
            else "EfficientAT MobileNet (from scratch)",
            builtin=True,
            provider="efficientat",
            # Every MN/DyMN width exposes its pre-pool temporal map through
            # EfficientATAdapter.forward_frame_features.  Keep scratch-only
            # widths (notably mn01/mn30) aligned with the dual-task family shown
            # by Model Lab instead of rejecting them only when training starts.
            task_capabilities=["clip", "frame"],
            export_supported=not is_dymn(name),
        )
    # The compact static config map predates the Model Lab.  Add every curated
    # architecture/weight source so selecting it in the UI also makes it
    # resolvable by the trainer and feature-fit probe.
    try:
        from sed.train.model_lab import curated_architectures

        for architecture in curated_architectures():
            if architecture.provider == "efficientat":
                for weight in architecture.weights:
                    name = weight.label
                    contract = weight.expected_feature_contract
                    specs[name] = BackboneSpec(
                        name=name, family="efficientat", source=name,
                        pretrained_available=True, input_contract=dict(contract or {}),
                        native_bands=int((contract or {}).get("features.n_mels", MN_NATIVE_BANDS)),
                        native_frames=MN_NATIVE_FRAMES, input_channels=1,
                        description=f"EfficientAT AudioSet checkpoint ({name})",
                        builtin=True, provider="efficientat",
                        task_capabilities=list(architecture.capabilities),
                        export_supported=architecture.export_supported,
                    )
            elif architecture.provider == "local":
                contract = None
                if architecture.id == "clink-cnn-v2":
                    contract = dict(CLINK_CNN_V2_FEATURE_CONTRACT)
                if architecture.weights:
                    for weight in architecture.weights:
                        if weight.expected_feature_contract:
                            contract = dict(weight.expected_feature_contract)
                            break
                native_frames = MN_NATIVE_FRAMES
                if contract:
                    try:
                        from sed.audio.features import feature_frame_count

                        native_frames = feature_frame_count(
                            sample_rate=int(contract.get("model.sample_rate", 32000)),
                            window_seconds=float(contract.get("features.window_seconds", 1.0)),
                            hop_length=int(contract.get("features.hop_length", 320)),
                            n_fft=int(contract.get("features.n_fft", 1024)),
                            frontend=str(contract.get("features.frontend", "efficientat")),
                            win_length=int(contract.get("features.win_length", 0) or 0) or None,
                        )
                    except Exception:  # noqa: BLE001 - metadata only
                        native_frames = MN_NATIVE_FRAMES
                specs[architecture.id] = BackboneSpec(
                    name=architecture.id, family="custom", source=architecture.source,
                    pretrained_available=bool(architecture.weights),
                    input_contract=contract,
                    native_bands=int((contract or {}).get("features.n_mels", MN_NATIVE_BANDS)),
                    native_frames=int(native_frames),
                    input_channels=int((contract or {}).get("model.channels", architecture.input_channels)),
                    params=[ParamSpec(**p) for p in architecture.params],
                    description=architecture.description, builtin=True, provider="local",
                    task_capabilities=list(architecture.capabilities),
                    export_supported=architecture.export_supported,
                )
            elif architecture.provider == "timm":
                specs[architecture.source] = BackboneSpec(
                    name=architecture.source, family="timm", source=architecture.source,
                    pretrained_available=True, input_channels=architecture.input_channels,
                    params=[ParamSpec(**p) for p in architecture.params],
                    description=architecture.description, builtin=True, provider="timm",
                    task_capabilities=list(architecture.capabilities),
                    export_supported=architecture.export_supported,
                )
    except Exception as exc:  # noqa: BLE001 - catalog must not break edge installs
        log.debug("unable to load Model Lab builtins: %s", exc)
    return specs


def _user_specs() -> dict[str, BackboneSpec]:
    specs: dict[str, BackboneSpec] = {}
    if not USER_MODEL_DIR.exists():
        return specs
    for path in sorted(USER_MODEL_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            spec = BackboneSpec.from_dict(data)
            spec.builtin = False
            specs[spec.name] = spec
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping malformed model spec %s: %s", path, exc)
    return specs


def all_specs() -> dict[str, BackboneSpec]:
    """Built-in specs overlaid with user-registered ones (user wins on name)."""
    specs = _builtin_specs()
    specs.update(_user_specs())
    return specs


def get_spec(name: str) -> BackboneSpec:
    specs = all_specs()
    if name in specs:
        return specs[name]
    # Unregistered timm name still resolvable on the fly.
    raise KeyError(name)


def resolve_adapter(name: str) -> BackboneAdapter:
    """Return a ready adapter for ``name`` (built-in, user-registered, or a bare
    timm model name as a last resort)."""
    specs = all_specs()
    if name in specs:
        spec = specs[name]
    elif str(name).startswith(("mn", "dymn")):
        # Defensive: an mn*/dymn* name not in BACKBONES still loads via EfficientAT.
        spec = BackboneSpec(name=name, family="efficientat", source=name)
    else:
        spec = BackboneSpec(name=name, family="timm", source=name)
    adapter_cls = _ADAPTERS.get(spec.family)
    if adapter_cls is None:
        raise ValueError(f"unknown backbone family {spec.family!r} for {name!r}")
    return adapter_cls(spec)


def registry_payload() -> list[dict[str, Any]]:
    """Serializable spec list for the Model tab / Audio Features dropdown."""
    return [spec.to_payload() for spec in all_specs().values()]


def save_user_spec(spec: BackboneSpec) -> Path:
    USER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_MODEL_DIR / f"{spec.name}.json"
    spec.builtin = False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_payload(), f, indent=2, sort_keys=True)
    return path


def delete_user_spec(name: str) -> bool:
    path = USER_MODEL_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def _build_and_probe(
    adapter: BackboneAdapter, *, n_bands: int, n_frames: int,
    params: dict[str, Any] | None, pretrained: bool, input_channels: int | None = None,
) -> dict[str, Any]:
    """One real build + forward; the shared core of both probe entry points."""
    in_ch = _requested_input_channels(input_channels, int(adapter.spec.input_channels or 1))
    backbone = adapter.build_backbone(pretrained=bool(pretrained), n_bands=n_bands,
                                      n_frames=n_frames, input_channels=in_ch,
                                      **(params or {}))
    # eval mode so the single zeros probe neither pollutes BatchNorm running
    # stats nor trips BatchNorm's "expected >1 value per channel" guard for
    # encoders that pool to a vector then apply BatchNorm1d (legacy SmallerVGGNet).
    backbone.eval()
    x = torch.zeros(1, in_ch, n_bands, n_frames)
    with torch.no_grad():
        fmap = adapter.feature_map_size(backbone, x)
        emb = adapter.forward_features(backbone, x)
    stage_groups = adapter.stages(backbone)
    return {
        "feat_dim": int(emb.shape[-1]),
        "feature_map": [int(fmap[0]), int(fmap[1])] if fmap else None,
        "input_channels": in_ch,
        "total_params": sum(p.numel() for p in backbone.parameters()),
        "batchnorm_layers": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in backbone.modules()
        ),
        "n_stages": len(stage_groups),
        "stage_names": [name for name, _ in stage_groups],
    }


def measure_model_fit(name: str, *, n_bands: int, n_frames: int,
                      params: dict[str, Any] | None = None,
                      pretrained: bool = False,
                      input_channels: int | None = None) -> dict[str, Any]:
    """Build a registered backbone once and measure its real feat_dim + pre-pool map.

    Used for the feature↔model compatibility check on non-mn (timm/custom)
    backbones, where the spatial reduction is not analytically known, and for the
    Model tab's "Verify model" probe. With ``pretrained=True`` it builds the real
    checkpoint (downloading it on first use), so it doubles as a load/download check.
    """
    adapter = resolve_adapter(name)
    probe = _build_and_probe(adapter, n_bands=n_bands, n_frames=n_frames,
                             params=params, pretrained=pretrained,
                             input_channels=input_channels)
    probe.pop("stage_names", None)
    return {
        "name": name,
        "family": adapter.spec.family,
        "pretrained_available": bool(adapter.spec.pretrained_available),
        "input_contract": adapter.spec.input_contract,
        **probe,
    }


def model_input_geometry(cfg) -> dict[str, Any]:
    """Resolved model input tensor geometry for a run config.

    Channels come from the configured feature; bands/frames from the feature helpers,
    which already account for
    colormap-rendered image inputs (legacy jet → image_size × image_size).
    """
    from sed.audio.features import feature_band_count, feature_channels, feature_frame_count_for_cfg

    n_bands = int(feature_band_count(cfg.features, cfg.model.sample_rate))
    n_frames = int(feature_frame_count_for_cfg(cfg))
    channels = int(feature_channels(cfg.features))
    return {
        "input_channels": channels,
        "n_bands": n_bands,
        "n_frames": n_frames,
        "feature_type": str(cfg.features.feature_type),
        "colormap": str(getattr(cfg.features, "colormap", "none") or "none"),
    }


@lru_cache(maxsize=32)
def _efficientat_param_count_cached(base_backbone: str, input_channels: int = 1) -> int | None:
    try:
        adapter = resolve_adapter(base_backbone)
        model = adapter.build_backbone(
            pretrained=False, n_bands=128, n_frames=100, input_channels=input_channels,
        )
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # noqa: BLE001 — count is informational; never break the fit report
        return None


def efficientat_param_count(backbone: str, input_channels: int = 1) -> int | None:
    """Scratch-build parameter count for an EfficientAT backbone.

    Shape-independent (the adapter ignores n_bands/n_frames — global pooling),
    so checkpoint variants (``mn10_as_mels_256`` …) are normalized to the base
    width first and share one cache entry per width.
    """
    from sed.train.model_lab import efficientat_base_backbone

    return _efficientat_param_count_cached(
        efficientat_base_backbone(str(backbone)),
        int(input_channels),
    )


def probe_spec(spec: BackboneSpec, *, n_bands: int = 128, n_frames: int = 100,
               params: dict[str, Any] | None = None,
               input_channels: int | None = None) -> dict[str, Any]:
    """Probe an (optionally not-yet-registered) spec for the register/validate flow."""
    adapter = _ADAPTERS[spec.family](spec)
    probe = _build_and_probe(adapter, n_bands=n_bands, n_frames=n_frames,
                             params=params, pretrained=False,
                             input_channels=input_channels)
    probe.pop("feature_map", None)
    probe.pop("input_channels", None)
    return {"downsample": spec.downsample, **probe}
