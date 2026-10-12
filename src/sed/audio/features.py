"""Log-mel spectrogram feature extraction.

This module is the single source of truth for training, export calibration,
calibration, inspection, and live inference. Supported frontends are
``efficientat`` (pre-emphasis, STFT, Kaldi mel banks, natural log, and fast
normalization for EfficientAT checkpoints) and ``ced`` (16 kHz / 64-mel dB
features for CED checkpoints). The old ``legacy_torchaudio_db`` frontend was
removed — bundles trained with it need an older repo revision or a retrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

try:
    import torch
    import torchaudio
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Shared display range for visualizing efficientat fast-normalized features
# ((ln(mel) + 4.5) / 5): -1.40 is digital silence (the true floor) and +1.6
# covers p99.9 of the loudest library transients (~65 dB span). Used by the
# inspect previews and the augmentation mix maps so both render comparably.
EFFICIENTAT_DISPLAY_RANGE: tuple[float, float] = (-1.4, 1.6)

# Display range for the linear (STFT log-power) preview. Values are
# ln(|STFT|^2 + 1e-5): ~-11.5 is the digital-silence floor and ~+2 covers the
# loudest transients. Absolute (not per-image autoscale) so quiet/loud clips
# stay comparable, matching the mel preview convention.
LINEAR_DISPLAY_RANGE: tuple[float, float] = (-11.5, 2.0)

# Audio feature types selectable in the Audio Features lab. Both are trainable
# (mel + efficientat-norm matches the pretrained contract; linear and PCEN need a
# from-scratch backbone).
FEATURE_TYPES: tuple[str, ...] = ("mel", "linear")

# Magnitude compression — the feature-generation axis. Mirrors
# sed.config.FEATURE_COMPRESSIONS, duplicated here so this module stays
# importable in torch-free / config-free deployments.
FEATURE_COMPRESSIONS: tuple[str, ...] = ("none", "log", "db", "pcen")

# STFT engines for the linear feature. "librosa" is the standard frontend
# (pre-emphasis + Hann + centered frames); "scipy" reproduces the legacy
# soundDetection system's scipy.signal.spectrogram exactly (Tukey 0.25 window,
# per-segment mean detrend, no pre-emphasis/centering, PSD density scaling,
# int16-full-scale amplitudes).
STFT_ENGINES: tuple[str, ...] = ("librosa", "scipy")
FEATURE_FRONTENDS: tuple[str, ...] = ("efficientat", "ced")

# The legacy system read WAVs as raw int16, so its PSD (and the -20..60 dB
# normalization range) lives at int16 full-scale. Our float waveforms are
# int16/32768, so the scipy engine scales amplitudes back up to reproduce the
# legacy values bit-for-bit.
LEGACY_INT16_SCALE: float = 32768.0

# Colormap image render (the legacy soundDetection model input). "jet" mirrors
# OpenCV's COLORMAP_JET.
COLORMAPS: tuple[str, ...] = ("none", "jet")

# OpenCV's COLORMAP_JET lookup table (256 entries, BGR byte order), dumped
# verbatim from cv2.applyColorMap — the analytic jet formula differs from
# OpenCV's interpolated table by ±1 on ~120 entries, so the exact bytes are
# embedded to reproduce the legacy colors bit-for-bit.
_JET_LUT_B64 = (
    "gAAAhAAAiAAAjAAAkAAAlAAAmAAAnAAAoAAApAAAqAAArAAAsAAAtAAAuAAAvAAAwAAAxAAAyAAA"
    "zAAA0AAA1AAA2AAA3AAA4AAA5AAA6AAA7AAA8AAA9AAA+AAA/AAA/wAA/wQA/wgA/wwA/xAA/xQA"
    "/xgA/xwA/yAA/yQA/ygA/ywA/zAA/zQA/zgA/zwA/0AA/0QA/0gA/0wA/1AA/1QA/1gA/1wA/2AA"
    "/2QA/2gA/2wA/3AA/3QA/3gA/3wA/4AA/4QA/4gA/4wA/5AA/5QA/5gA/5wA/6AA/6QA/6gA/6wA"
    "/7AA/7QA/7gA/7wA/8AA/8QA/8gA/8wA/9AA/9QA/9gA/9wA/+AA/+QA/+gA/+wA//AA//QA//gA"
    "//wA/v8C+v8G9v8K8v8O7v8S6v8W5v8a4v8e3v8i2v8m1v8q0v8uzv8yyv82xv86wv8+vv9Cuv9G"
    "tv9Ksv9Orv9Sqv9Wpv9aov9env9imv9mlv9qkv9ujv9yiv92hv96gv9+fv+Cev+Gdv+Kcv+Obv+S"
    "av+WZv+aYv+eXv+iWv+mVv+qUv+uTv+ySv+2Rv+6Qv++Pv/COv/GNv/KMv/OLv/SKv/WJv/aIv/e"
    "Hv/iGv/mFv/qEv/uDv/yCv/2Bv/6Af/+APz/APj/APT/APD/AOz/AOj/AOT/AOD/ANz/ANj/ANT/"
    "AND/AMz/AMj/AMT/AMD/ALz/ALj/ALT/ALD/AKz/AKj/AKT/AKD/AJz/AJj/AJT/AJD/AIz/AIj/"
    "AIT/AID/AHz/AHj/AHT/AHD/AGz/AGj/AGT/AGD/AFz/AFj/AFT/AFD/AEz/AEj/AET/AED/ADz/"
    "ADj/ADT/ADD/ACz/ACj/ACT/ACD/ABz/ABj/ABT/ABD/AAz/AAj/AAT/AAD/AAD8AAD4AAD0AADw"
    "AADsAADoAADkAADgAADcAADYAADUAADQAADMAADIAADEAADAAAC8AAC4AAC0AACwAACsAACoAACk"
    "AACgAACcAACYAACUAACQAACMAACIAACEAACA"
)
_JET_LUT_CACHE: np.ndarray | None = None


def jet_lut() -> np.ndarray:
    """The exact cv2.COLORMAP_JET table as float32 in [0, 1], shape (256, 3), RGB.

    The embedded bytes are OpenCV's native BGR order; they are reversed to RGB
    here so a low value renders blue and a high value red when displayed/saved as
    RGB (matplotlib/PIL). Using the raw BGR directly swaps blue↔red, which made
    quiet (mostly-low) spectrograms look red instead of blue.
    """
    global _JET_LUT_CACHE
    if _JET_LUT_CACHE is None:
        import base64
        raw = np.frombuffer(base64.b64decode(_JET_LUT_B64), dtype=np.uint8)
        bgr = raw.reshape(256, 3).astype(np.float32) / 255.0
        _JET_LUT_CACHE = np.ascontiguousarray(bgr[:, ::-1])  # BGR → RGB
    return _JET_LUT_CACHE


def _bilinear_resize_hw(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Half-pixel bilinear resize of an (H, W, C) image to (out_h, out_w, C).

    Uses the half-pixel-center convention (source = (dst + 0.5)·scale − 0.5),
    matching cv2.resize(INTER_LINEAR) and torch F.interpolate(align_corners=False)
    so training, inference, and the preview all resize identically.
    """
    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32, copy=False)

    def _axis(n_out: int, n_in: int):
        scale = n_in / float(n_out)
        centers = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
        lo = np.floor(centers).astype(np.intp)
        frac = (centers - lo).astype(np.float32)
        hi = lo + 1
        lo = np.clip(lo, 0, n_in - 1)
        hi = np.clip(hi, 0, n_in - 1)
        return lo, hi, frac

    ry_lo, ry_hi, wy = _axis(out_h, in_h)
    rx_lo, rx_hi, wx = _axis(out_w, in_w)
    arr = img.astype(np.float32, copy=False)
    # Interpolate rows then columns.
    top = arr[ry_lo] * (1.0 - wy)[:, None, None] + arr[ry_hi] * wy[:, None, None]
    out = top[:, rx_lo] * (1.0 - wx)[None, :, None] + top[:, rx_hi] * wx[None, :, None]
    return out.astype(np.float32)


def render_spectrogram_image(
    spec01: np.ndarray, colormap: str = "jet", image_size: int = 96
) -> np.ndarray:
    """Normalized [0,1] spectrogram (F, T) → (3, image_size, image_size) image.

    The legacy soundDetection colormap render, in the conventional spectrogram
    orientation: clamp to [0,1], map to a 3-channel colormap (OpenCV COLORMAP_JET
    via the embedded LUT — RGB, so low=blue/high=red — indexed by floor(v·255)),
    flip so the highest frequency is the top row, and resize to a square
    image_size. Frequency stays on the rows and time on the columns (the legacy
    cv2.transpose that put time on the rows is dropped — it only rotated the image
    90°, which is irrelevant to a from-scratch CNN but made the preview look
    tilted). This exact image is the model's input under colormap mode, so the
    preview, training, and inference all see the same thing.
    """
    v = np.clip(np.asarray(spec01, dtype=np.float32), 0.0, 1.0)
    if str(colormap).lower() == "jet":
        idx = np.floor(v * 255.0).astype(np.intp)
        np.clip(idx, 0, 255, out=idx)
        img = jet_lut()[idx]                      # (F, T, 3), RGB
    else:
        img = np.repeat(v[:, :, None], 3, axis=2)  # grayscale × 3
    img = np.flipud(img)                          # highest frequency on the top row
    size = int(image_size)
    img = _bilinear_resize_hw(img, size, size)    # (image_size, image_size, 3)
    return np.transpose(img, (2, 0, 1)).astype(np.float32)  # (3, H, W)


def feature_channels(features) -> int:
    """Model-input channel count for a FeaturesCfg: 3 for a colormap image, else 1."""
    return 3 if str(getattr(features, "colormap", "none")).lower() != "none" else 1

# Default librosa.pcen parameters; overridable per FeaturesCfg.
PCEN_DEFAULTS: dict = {
    "time_constant": 0.4,
    "gain": 0.98,
    "bias": 2.0,
    "power": 0.5,
    "eps": 1e-6,
}


@dataclass(frozen=True)
class ScalingSpec:
    """Resolved value normalization for one feature (see FeaturesCfg).

    This is the feature pipeline's ONLY normalization stage. ``a``/``b`` are the
    constants — mean/std for ``zscore``, lo/hi for ``minmax`` — shaped ``(1,)``
    for global scope or ``(F,)`` per band, and ``source`` records where they
    came from: ``fixed`` (a checkpoint contract or hand-entered values, e.g.
    EfficientAT's (x + 4.5) / 5 or the legacy [-20, 60] dB map) or ``dataset``
    (fitted on the training split). A ``declared`` dataset spec without
    constants is not fitted yet and applies as the identity, so previews and the
    statistics fit see the true compressed domain.
    """

    method: str = "none"           # none | zscore | minmax
    a: np.ndarray | None = None    # zscore: mean; minmax: lo
    b: np.ndarray | None = None    # zscore: std;  minmax: hi
    scope: str = "global"          # global | per_band
    source: str = "dataset"        # fixed | dataset

    @property
    def declared(self) -> bool:
        return self.method != "none"

    @property
    def active(self) -> bool:
        return self.declared and self.a is not None and self.b is not None

    def describe(self) -> str:
        if not self.declared:
            return "none"
        name = "z-score" if self.method == "zscore" else "min-max"
        return f"{name} · {self.scope} · {self.source}"


def scaling_spec_from_stats(
    method: str, stats: dict | None, scope: str = "global", source: str = "dataset"
) -> ScalingSpec:
    """Build a ScalingSpec from the ``features.scaling_*`` fields."""
    method = str(method or "none").lower()
    scope = str(scope or "global").lower()
    source = str(source or "dataset").lower()
    if method == "none":
        return ScalingSpec(scope=scope, source=source)
    keys = ("mean", "std") if method == "zscore" else ("lo", "hi")
    stats = stats or {}
    a = np.asarray(stats.get(keys[0]) or [], dtype=np.float32)
    b = np.asarray(stats.get(keys[1]) or [], dtype=np.float32)
    if a.size == 0 or a.shape != b.shape:
        return ScalingSpec(method=method, scope=scope, source=source)
    return ScalingSpec(method=method, a=a, b=b, scope=scope, source=source)


def efficientat_fast_norm_spec() -> ScalingSpec:
    """EfficientAT's published (x + 4.5) / 5 affine, as an ordinary fixed spec.

    Kept here so torch-free/direct callers of :class:`MelExtractor` (which
    predate the explicit config axis) still get the AudioSet contract by default
    instead of unnormalized log-mel.
    """
    from sed.config import EFFICIENTAT_FAST_NORM_MEAN, EFFICIENTAT_FAST_NORM_STD

    return ScalingSpec(
        method="zscore",
        a=np.asarray([EFFICIENTAT_FAST_NORM_MEAN], dtype=np.float32),
        b=np.asarray([EFFICIENTAT_FAST_NORM_STD], dtype=np.float32),
        scope="global",
        source="fixed",
    )


def _is_bare_efficientat_logmel(features) -> bool:
    """Whether a feature is plain log-mel on the efficientat frontend."""
    return (
        str(getattr(features, "frontend", "efficientat") or "efficientat").lower()
        == "efficientat"
        and str(getattr(features, "feature_type", "mel") or "mel") == "mel"
        and str(getattr(features, "colormap", "none") or "none").lower() == "none"
        and feature_compression(features) == "log"
    )


def resolve_feature_scaling(features, n_bands: int | None = None) -> ScalingSpec:
    """ScalingSpec for a FeaturesCfg (or compatible object).

    When ``n_bands`` is known, per-band vector lengths are validated against
    the feature geometry — a stale vector from a previous geometry must fail
    loudly instead of scaling the wrong rows.
    """
    spec = scaling_spec_from_stats(
        getattr(features, "scaling_method", "none"),
        getattr(features, "scaling_stats", None),
        getattr(features, "scaling_scope", "global"),
        getattr(features, "scaling_source", "dataset"),
    )
    if not spec.declared and _is_bare_efficientat_logmel(features):
        # Plain log-mel on the efficientat frontend IS the AudioSet contract —
        # its affine is a fixed z-score, not the absence of one. Config
        # validation records that explicitly; objects built without the field
        # (bare namespaces, older snapshots) get it resolved here.
        spec = efficientat_fast_norm_spec()
    if (
        spec.active
        and spec.a is not None
        and spec.a.size > 1
        and n_bands is not None
        and spec.a.size != int(n_bands)
    ):
        raise ValueError(
            f"features.scaling_stats carries {spec.a.size} bands but this feature "
            f"produces {int(n_bands)} — re-estimate the statistics for this geometry"
        )
    return spec


def apply_feature_scaling(x: np.ndarray, spec: ScalingSpec) -> np.ndarray:
    """Apply a ScalingSpec to a ``(..., F, T)`` numpy feature.

    Per-band constants broadcast over the band axis; an inactive (declared but
    unfitted) spec passes values through unchanged.
    """
    if not spec.active:
        return x
    a, b = spec.a, spec.b
    if a.size > 1:
        a = a.reshape((-1, 1))
        b = b.reshape((-1, 1))
    else:
        a = float(a[0])
        b = float(b[0])
    if spec.method == "zscore":
        return ((x - a) / b).astype(np.float32, copy=False)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32, copy=False)


def apply_feature_scaling_torch(x, spec: ScalingSpec):
    """Torch twin of :func:`apply_feature_scaling` for ``(B, F, T)`` tensors."""
    if not spec.active:
        return x
    a = torch.as_tensor(spec.a, dtype=x.dtype, device=x.device)
    b = torch.as_tensor(spec.b, dtype=x.dtype, device=x.device)
    if a.numel() > 1:
        a = a.reshape(1, -1, 1)
        b = b.reshape(1, -1, 1)
    if spec.method == "zscore":
        return (x - a) / b
    return ((x - a) / (b - a)).clamp(0.0, 1.0)


# Floors for the compression stage. LOG_EPS is EfficientAT's own 1e-5 (the mel
# contract depends on it byte-for-byte); DB_FLOOR is -120 dB, low enough that any
# sane dB window clips it to the bottom exactly as -inf would, while keeping the
# values finite for everything downstream.
LOG_EPS: float = 1e-5
DB_FLOOR_POWER: float = 1e-12


def compress_power(
    power: np.ndarray,
    compression: str = "log",
    *,
    sample_rate: int = 32000,
    hop_length: int = 320,
    pcen_params: dict | None = None,
) -> np.ndarray:
    """Power spectrogram → the compressed values, for any feature type.

    This is the feature-generation stage: it decides what the numbers mean.
    Mapping those numbers into the model's range is a separate stage
    (:func:`apply_feature_scaling`), so mel and linear are compressed and
    normalized by exactly the same two steps.
    """
    kind = str(compression or "log").lower()
    arr = np.asarray(power, dtype=np.float32)
    if kind == "none":
        return arr
    if kind == "log":
        return np.log(arr + LOG_EPS).astype(np.float32)
    if kind == "db":
        return (10.0 * np.log10(np.maximum(arr, DB_FLOOR_POWER))).astype(np.float32)
    if kind == "pcen":
        return _apply_pcen(arr, sample_rate, hop_length, pcen_params)
    raise ValueError(f"unsupported features.compression {compression!r}")


def compress_power_torch(x, compression: str = "log"):
    """Torch twin of :func:`compress_power` for the batched extractors.

    PCEN has no torch implementation here — the extractors that offer it route
    through the shared numpy path instead.
    """
    kind = str(compression or "log").lower()
    if kind == "none":
        return x
    if kind == "log":
        return (x + LOG_EPS).log()
    if kind == "db":
        return 10.0 * torch.log10(x.clamp_min(DB_FLOOR_POWER))
    raise ValueError(f"features.compression {compression!r} has no torch path")


def preview_display_range(
    feature_type: str, log_power: bool = True, pcen: bool = False,
    db_norm: bool = False,
) -> tuple[float | None, float | None]:
    """Fixed (vmin, vmax) display range for a feature's preview.

    Mel and log-power linear use an absolute range so clips stay comparable;
    raw-power linear and PCEN (already an automatic gain control) autoscale per
    image (return None, None). The legacy dB normalization is already mapped to
    [0, 1], so its display range is exactly that.
    """
    if pcen:
        return (None, None)
    if str(feature_type) == "linear":
        if db_norm:
            return (0.0, 1.0)
        return LINEAR_DISPLAY_RANGE if log_power else (None, None)
    return EFFICIENTAT_DISPLAY_RANGE


def robust_display_range(
    spec: np.ndarray, lo: float = 1.0, hi: float = 99.5
) -> tuple[float | None, float | None]:
    """Per-image (vmin, vmax) from robust percentiles — high-contrast previews.

    Linear log/raw power and PCEN have wide, clip-dependent dynamic ranges that a
    fixed absolute range washes out; percentile clipping gives a consistently
    legible image (the quantitative comparison comes from the analysis metrics,
    which are computed from the true power separately)."""
    arr = np.asarray(spec, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (None, None)
    vmin = float(np.percentile(arr, lo))
    vmax = float(np.percentile(arr, hi))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return (None, None)
    return (vmin, vmax)


def display_range_for_spec(
    spec: np.ndarray, feature_type: str, log_power: bool = True, pcen: bool = False,
    db_norm: bool = False, scaling_method: str = "none",
    scaling_source: str = "dataset", scaling_stats: dict | None = None,
) -> tuple[float | None, float | None]:
    """Display range for a rendered spec, decided by the normalization stage.

    A min-max normalization always lands in exactly [0, 1] — including the
    legacy dB image map, which is one. The AudioSet fast-norm is a known fixed
    z-score, so it keeps its calibrated absolute range and quiet/loud clips stay
    comparable; any other z-score, and every unnormalized feature with a wide
    dynamic range, autoscales by robust percentiles.
    """
    method = str(scaling_method or "none").lower()
    if method == "minmax":
        return (0.0, 1.0)
    if method == "zscore":
        stats = scaling_stats or {}
        mean = (stats.get("mean") or [None])[0]
        std = (stats.get("std") or [None])[0]
        if (
            str(scaling_source or "").lower() == "fixed"
            and mean is not None and std is not None
            and abs(float(mean) + 4.5) < 1e-9 and abs(float(std) - 5.0) < 1e-9
        ):
            return EFFICIENTAT_DISPLAY_RANGE
        return robust_display_range(spec)
    if str(feature_type) == "linear" and db_norm and not pcen:
        return (0.0, 1.0)
    if pcen or str(feature_type) == "linear":
        return robust_display_range(spec)
    return preview_display_range(feature_type, log_power, pcen)


def legacy_implied_compression(features) -> str:
    """The compression the old per-feature-type flags describe.

    ``pcen`` and ``db_norm`` are opt-in booleans, so they win where set;
    ``log_power=False`` means raw power on a linear feature; the CED frontend
    is a dB-mel by construction.
    """
    if bool(getattr(features, "pcen", False)):
        return "pcen"
    if bool(getattr(features, "db_norm", False)):
        return "db"
    if str(getattr(features, "frontend", "efficientat")).lower() == "ced":
        return "db"
    if (str(getattr(features, "feature_type", "mel")) == "linear"
            and not bool(getattr(features, "log_power", True))):
        return "none"
    return "log"


def feature_compression(features) -> str:
    """The compression stage of a FeaturesCfg (or any compatible object).

    The canonical ``compression`` field wins, except when it still holds the
    default "log" while the older flags name something else — that is either a
    config written before this field existed, or a live object whose ``pcen`` /
    ``db_norm`` was flipped in place without re-validating.
    """
    value = str(getattr(features, "compression", "") or "").lower()
    implied = legacy_implied_compression(features)
    if value not in FEATURE_COMPRESSIONS or (value == "log" and implied != "log"):
        return implied
    return value


def pcen_params_from_features(features) -> dict:
    """Extract the librosa.pcen kwargs from a FeaturesCfg (or compatible object)."""
    return {
        "time_constant": getattr(features, "pcen_time_constant", PCEN_DEFAULTS["time_constant"]),
        "gain": getattr(features, "pcen_gain", PCEN_DEFAULTS["gain"]),
        "bias": getattr(features, "pcen_bias", PCEN_DEFAULTS["bias"]),
        "power": getattr(features, "pcen_power", PCEN_DEFAULTS["power"]),
        "eps": getattr(features, "pcen_eps", PCEN_DEFAULTS["eps"]),
    }


def feature_band_count(features, sample_rate: int) -> int:
    """Number of feature rows (the model-input height) for a FeaturesCfg.

    Colormap image → image_size; mel → n_mels; linear → rfft bins in [fmin, fmax].
    """
    if str(getattr(features, "colormap", "none")).lower() != "none":
        return int(getattr(features, "image_size", 96))
    if str(getattr(features, "feature_type", "mel")) == "linear":
        return int(_linear_freq_mask(
            int(features.n_fft), int(sample_rate), float(features.fmin), float(features.fmax),
        ).sum())
    return int(features.n_mels)


# The mn/dymn backbone's 5 stride-2 layers (kernel, stride, padding) use SAME
# padding (pad = kernel // 2). Replaying the vendored cnn_out_size over these
# reproduces the real pre-pool feature map exactly: 128→4, 100→4, 1000→32, 64→2
# (verified against an actual forward through the vendored MN/DyMN backbones).
_MN_STRIDE2_LAYERS: tuple[tuple[int, int, int], ...] = (
    (3, 2, 1), (3, 2, 1), (5, 2, 2), (3, 2, 1), (5, 2, 2),
)


def _mn_reduce(size: int) -> int:
    """Spatial size after the mn backbone's 5 stride-2 layers (one axis)."""
    import math

    out = int(size)
    for kernel, stride, pad in _MN_STRIDE2_LAYERS:
        out = math.floor((out + 2 * pad - (kernel - 1) - 1) / stride + 1)
        if out <= 0:
            return 0
    return out


def _task_verdicts(task: str, fmap_t: int | None) -> list[dict]:
    """Task-aware verdicts: frame-level SED needs enough temporal cells (before
    the 32× reduction) to localize events; clip classification does not."""
    if str(task or "clip").lower() != "frame":
        return []
    if fmap_t is None:
        return [{"level": "error", "text":
                 "This encoder pools to a single embedding, so per-frame timestamps "
                 "are unavailable — frame-level SED needs a spatial feature map."}]
    if fmap_t <= 1:
        return [{"level": "error", "text":
                 f"Only {fmap_t} temporal cell before pooling — frame-level SED cannot "
                 "localize events. Use a longer window (≥1 s) or a smaller hop."}]
    if fmap_t < 4:
        return [{"level": "warn", "text":
                 f"Only {fmap_t} temporal cells before the 32× reduction — coarse for "
                 "frame-level SED (a 0.4 s window gives ~2 cells). A ≥1 s window or a "
                 "smaller hop gives finer event timing."}]
    return [{"level": "ok", "text":
             f"{fmap_t} temporal cells before pooling — enough resolution for "
             "frame-level event timing."}]


def _contract_scaling_spec(contract: dict) -> ScalingSpec:
    """The normalization a checkpoint contract pins, as a ScalingSpec.

    Older contract dicts spell it with ``features.standardize`` + mean/std;
    current ones name the method, scope, source and constants outright.
    """
    method = str(contract.get("features.scaling_method", "") or "").lower()
    stats = dict(contract.get("features.scaling_stats", {}) or {})
    if not method and bool(contract.get("features.standardize", False)):
        method, stats = "zscore", {
            "mean": [float(contract.get("features.standardize_mean", 0.0))],
            "std": [float(contract.get("features.standardize_std", 1.0))],
        }
    return scaling_spec_from_stats(
        method or "none",
        stats,
        str(contract.get("features.scaling_scope", "global") or "global"),
        str(contract.get("features.scaling_source", "fixed") or "fixed"),
    )


def _contract_mismatch_verdicts(
    features,
    sample_rate: int,
    n_bands: int,
    input_channels: int = 1,
    contract: dict | None = None,
) -> list[dict]:
    """Per-parameter compatibility of a feature with the AudioSet-pretrained
    EfficientAT weights, each tagged with a severity (high / medium / low).

    The weights were trained on a fixed mel contract; the further a feature
    departs from it, the worse the pretrained filters transfer. Used by the Model
    tab's "Model & feature fit" compatibility report. Never blocks training.
    """
    if contract is None:
        from sed.config import EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT as default_contract

        contract = default_contract

    checks: list[tuple[str, object, object, str, str]] = []
    ftype = str(getattr(features, "feature_type", "mel"))
    if ftype != "mel":
        checks.append(("feature_type", ftype, "mel", "high",
            "a linear/other spectrogram is a different basis than the mel input the weights "
            "learned — the pretrained filters do not transfer"))
    # Axis 1: compression — what the numbers mean.
    have_compression = feature_compression(features)
    want_compression = str(contract.get("features.compression", "log") or "log").lower()
    if have_compression != want_compression:
        checks.append(("compression", have_compression, want_compression, "high",
            "the checkpoint learned on a different magnitude compression, so the same sound "
            "arrives with a different value distribution and dynamic range"))
    # Axis 2: normalization — what range those numbers arrive in.
    want_scaling = _contract_scaling_spec(contract)
    have_scaling = resolve_feature_scaling(features)
    if (have_scaling.method, have_scaling.scope, have_scaling.source) != (
        want_scaling.method, want_scaling.scope, want_scaling.source,
    ):
        checks.append(("normalization",
            have_scaling.describe(), want_scaling.describe(), "high",
            "the checkpoint expects one specific value normalization — swapping its fixed "
            "affine for dataset-fitted constants (or vice versa) moves every input value"))
    elif want_scaling.active and have_scaling.active:
        same = (
            have_scaling.a.shape == want_scaling.a.shape
            and bool(np.allclose(have_scaling.a, want_scaling.a, atol=1e-3))
            and bool(np.allclose(have_scaling.b, want_scaling.b, atol=1e-3))
        )
        if not same:
            def _fmt(spec: ScalingSpec) -> str:
                if spec.a.size == 1:
                    return f"{float(spec.a[0]):.4g} / {float(spec.b[0]):.4g}"
                return f"{spec.a.size} fitted band pairs"
            checks.append(("normalization constants", _fmt(have_scaling), _fmt(want_scaling),
                "medium",
                "different constants shift/scale every input value relative to what the "
                "checkpoint saw in pretraining"))
    if int(sample_rate) != int(contract["model.sample_rate"]):
        checks.append(("sample_rate", int(sample_rate), int(contract["model.sample_rate"]), "high",
            "changes the Hz each mel bin covers, so pretrained frequency filters land on "
            "different content"))
    expected_channels = int(contract.get("model.channels", 1))
    if int(input_channels) != expected_channels:
        checks.append(("channels", int(input_channels), expected_channels, "high",
            "the first convolution was trained for a different number of input channels"))
    if int(n_bands) != int(contract["features.n_mels"]):
        checks.append(("n_mels", int(n_bands), int(contract["features.n_mels"]), "high",
            "the input band count differs from the 128 the first conv learned over"))
    fmin = int(getattr(features, "fmin", 0))
    fmax = int(getattr(features, "fmax", 0))
    if fmin != int(contract["features.fmin"]) or fmax != int(contract["features.fmax"]):
        checks.append(("fmin–fmax", f"{fmin}–{fmax}",
            f"{int(contract['features.fmin'])}–{int(contract['features.fmax'])}", "medium",
            "shifts the mel band edges, so each band's frequency content drifts from pretraining"))
    if (int(getattr(features, "n_fft", 0)) != int(contract["features.n_fft"])
            or int(getattr(features, "win_length", 0)) != int(contract["features.win_length"])):
        checks.append(("n_fft / win_length",
            f"{int(getattr(features, 'n_fft', 0))} / {int(getattr(features, 'win_length', 0))}",
            f"{int(contract['features.n_fft'])} / {int(contract['features.win_length'])}", "medium",
            "changes time–frequency resolution; tolerated but off the training distribution"))
    if int(getattr(features, "hop_length", 0)) != int(contract["features.hop_length"]):
        checks.append(("hop_length", int(getattr(features, "hop_length", 0)),
            int(contract["features.hop_length"]), "low",
            "only changes the frame rate / frame count, which clip-level global pooling absorbs"))

    rank = {"high": 0, "medium": 1, "low": 2}
    checks.sort(key=lambda c: rank.get(c[3], 3))
    verdicts: list[dict] = []
    for label, actual, expected, severity, why in checks:
        verdicts.append({
            "level": "warn", "severity": severity, "kind": "contract_mismatch",
            "text": f"{severity.upper()} · {label} = {actual} (contract {expected}) — {why}.",
        })
    return verdicts


def _efficientat_static_meta(backbone: str) -> dict:
    """Analytic EfficientAT metadata for curated variants outside BACKBONES."""
    from sed.config import BACKBONES

    name = str(backbone)
    if name in BACKBONES:
        return dict(BACKBONES[name])
    for key, meta in sorted(BACKBONES.items(), key=lambda item: len(item[0]), reverse=True):
        stem = key.split("_as", 1)[0]
        if name.startswith(stem) and name.startswith(("mn", "dymn")):
            out = dict(meta)
            out["pretrained_available"] = "_as" in name or bool(meta.get("pretrained_available"))
            return out
    return {}


def _generic_model_fit_report(
    backbone: str, pretrained: bool, n_bands: int, n_frames: int,
    input_channels: int, task: str = "clip", params: dict | None = None,
) -> dict:
    """Feature-fit report for a non-mn (timm/custom) backbone, measured by an
    actual dummy forward (the spatial reduction isn't analytically known)."""
    from sed.train.backbones import measure_model_fit

    try:
        fit = measure_model_fit(
            backbone,
            n_bands=n_bands,
            n_frames=n_frames,
            params=params,
            pretrained=bool(pretrained),
            input_channels=input_channels,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "backbone": backbone,
            "input_channels": input_channels,
            "input_shape": [1, input_channels, n_bands, n_frames],
            "n_bands": n_bands, "n_frames": n_frames,
            "verdicts": [{"level": "error",
                          "text": f"Could not build {backbone!r}: {exc}"}],
        }
    fmap = fit.get("feature_map")
    feat_dim = int(fit.get("feat_dim") or 0)
    in_ch = int(fit.get("input_channels") or input_channels)
    verdicts: list[dict] = []
    if fmap is None:
        verdicts.append({"level": "ok", "kind": "vector_embedding", "text":
            f"{backbone} returns a {feat_dim}-d vector embedding directly; "
            "clip training and inference use that embedding without spatial pooling."})
    elif fmap[0] < 1 or fmap[1] < 1:
        verdicts.append({"level": "error", "text":
            f"Feature map collapses to {fmap[0]}×{fmap[1]} — too small to train. "
            "Increase n_bands and/or the window."})
    elif fmap[0] == 1 or fmap[1] == 1:
        verdicts.append({"level": "warn", "text":
            f"Thin {fmap[0]}×{fmap[1]} feature map before global pooling — usable "
            "but with little spatial context."})
    else:
        verdicts.append({"level": "ok", "text":
            f"{fmap[0]}×{fmap[1]} feature map before global pooling → {feat_dim}-d embedding."})
    if pretrained and fit.get("pretrained_available"):
        verdicts.append({"level": "ok", "text":
            f"{backbone} ships pretrained weights — loaded for transfer."})
    elif pretrained:
        verdicts.append({"level": "warn", "text":
            f"{backbone} has no bundled pretrained weights; training from scratch."})
    verdicts.extend(_task_verdicts(task, fmap[1] if fmap else None))
    return {
        "backbone": backbone,
        "family": fit.get("family"),
        "input_channels": in_ch,
        "input_shape": [1, in_ch, n_bands, n_frames],
        "n_bands": n_bands,
        "n_frames": n_frames,
        "feature_map": fmap,
        "feat_dim": feat_dim,
        "total_params": fit.get("total_params"),
        "n_stages": fit.get("n_stages"),
        "head": (
            f"vector embedding ({feat_dim},) → Linear(num_classes)"
            if fmap is None else
            f"global avg-pool → ({feat_dim},) → Linear(num_classes)"
        ),
        "on_contract": False,
        "verdicts": verdicts,
    }


def model_fit_report(features, backbone: str, pretrained: bool, sample_rate: int,
                     task: str = "clip", params: dict | None = None,
                     weight_id: str = "") -> dict:
    """Describe how a feature shape fits the mn/dymn backbone — for the "Align model
    to features" panel and the Model tab. Pure/analytical (no model load): input
    tensor, 32× downsample, pre-pool feature map, embedding dim, native shape, and
    compatibility verdicts (incl. task-aware frame-level resolution checks).

    ``weight_id`` (when known) pins the exact checkpoint contract — hop/mels
    variants share a backbone name prefix, so resolving by backbone alone can
    pick the wrong geometry.
    """
    from sed.config import (
        EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT,
        MN_DOWNSAMPLE,
        MN_INPUT_CHANNELS,
        MN_NATIVE_BANDS,
        MN_NATIVE_FRAMES,
    )

    meta = _efficientat_static_meta(str(backbone))
    n_bands = feature_band_count(features, sample_rate)
    in_ch = feature_channels(features)
    if str(getattr(features, "colormap", "none")).lower() != "none":
        # Colormap image: model input is the fixed square, not the spectrogram.
        n_frames = int(getattr(features, "image_size", 96))
    else:
        n_frames = feature_frame_count(
            sample_rate=sample_rate,
            window_seconds=float(getattr(features, "window_seconds", 1.0)),
            hop_length=int(features.hop_length),
            n_fft=int(features.n_fft),
            frontend=str(getattr(features, "frontend", "efficientat")),
            win_length=int(getattr(features, "win_length", 0) or 0) or None,
            stft_engine=str(getattr(features, "stft_engine", "librosa")),
        )
    mel_verdict, mel_diag = mel_geometry_warning(features, sample_rate)
    if not meta:
        # Non-mn backbone (timm/custom): the spatial reduction isn't analytic, so
        # build the model once and measure the real feature map + embedding dim.
        report = _generic_model_fit_report(
            str(backbone), bool(pretrained), n_bands, n_frames, in_ch, task, params
        )
        if mel_verdict is not None:
            report.setdefault("verdicts", []).append(mel_verdict)
        report["mel_diagnostics"] = mel_diag
        return report
    feat_dim = int(meta.get("feat_dim", 0))
    fmap_f, fmap_t = _mn_reduce(n_bands), _mn_reduce(n_frames)

    try:
        from sed.train.model_lab import expected_contract_for_backbone, expected_contract_for_weight

        contract = expected_contract_for_weight(weight_id) or expected_contract_for_backbone(str(backbone))
    except Exception:  # noqa: BLE001
        contract = None
    contract = dict(contract or EFFICIENTAT_PRETRAINED_FEATURE_CONTRACT)
    # Geometry, then the two value axes — a feature is on-contract only when the
    # checkpoint would see the same numbers in the same range.
    on_contract = (
        str(getattr(features, "feature_type", "mel")) == "mel"
        and n_bands == contract["features.n_mels"]
        and int(sample_rate) == contract["model.sample_rate"]
        and int(getattr(features, "n_fft", 0)) == int(contract["features.n_fft"])
        and int(getattr(features, "win_length", 0)) == int(contract["features.win_length"])
        and int(getattr(features, "hop_length", 0)) == int(contract["features.hop_length"])
        and int(getattr(features, "fmin", 0)) == int(contract["features.fmin"])
        and int(getattr(features, "fmax", 0)) == int(contract["features.fmax"])
        and in_ch == int(contract.get("model.channels", MN_INPUT_CHANNELS))
        and not _contract_mismatch_verdicts(features, sample_rate, n_bands, in_ch, contract)
    )

    verdicts: list[dict] = []
    if fmap_f < 1 or fmap_t < 1:
        verdicts.append({"level": "error", "text":
            f"Feature map collapses to {fmap_f}×{fmap_t} after the 32× reduction — "
            f"too small to train. Increase n_bands (≥64) and/or the window so each axis stays ≥1."})
    elif fmap_f == 1 or fmap_t == 1:
        verdicts.append({"level": "warn", "text":
            f"Thin {fmap_f}×{fmap_t} feature map — one axis is a single cell after the 32× "
            "reduction; usable but with little spatial context. More bands/frames give the net more to pool."})
    else:
        verdicts.append({"level": "ok", "text":
            f"{fmap_f}×{fmap_t} feature map before global pooling "
            f"(native AudioSet is {_mn_reduce(MN_NATIVE_BANDS)}×{_mn_reduce(MN_NATIVE_FRAMES)} from 128×{MN_NATIVE_FRAMES})."})
    if pretrained and meta.get("pretrained_available"):
        if on_contract:
            verdicts.append({"level": "ok", "text":
                "Matches the AudioSet-pretrained contract for this variant — "
                "weights transfer directly."})
        else:
            mismatch = _contract_mismatch_verdicts(
                features, sample_rate, n_bands, in_ch, contract
            )
            worst = mismatch[0].get("severity", "low") if mismatch else "low"
            verdicts.append({"level": "warn", "severity": worst, "kind": "contract_summary", "text":
                f"Off the AudioSet-pretrained contract — {len(mismatch)} "
                f"difference{'s' if len(mismatch) != 1 else ''} (worst: {worst.upper()}). "
                "The model still builds (conv + global pool), but the pretrained weights may "
                "transfer poorly. To train this feature directly, set initialization to "
                "scratch (random) in step 1."})
            verdicts.extend(mismatch)
        if in_ch != MN_INPUT_CHANNELS:
            verdicts.append({"level": "error", "text":
                "EfficientAT pretrained checkpoints can only consume a 1-channel "
                "spectrogram. This feature produces a 3-channel image; switch "
                "initialization to scratch/random or choose a 1-channel feature."})
    elif pretrained:
        verdicts.append({"level": "warn", "text":
            f"{backbone} has no AudioSet-pretrained weights — it will train from scratch, "
            "so any feature geometry is valid."})
    else:
        verdicts.append({"level": "ok", "text":
            "Scratch initialization — any feature geometry is valid; the model learns "
            "directly from this feature."})
        if in_ch != MN_INPUT_CHANNELS:
            verdicts.append({"level": "ok", "text":
                f"Scratch EfficientAT stem will be built for {in_ch} input channels."})
    if mel_verdict is not None:
        verdicts.append(mel_verdict)
    verdicts.extend(_task_verdicts(task, fmap_t))

    return {
        "backbone": backbone,
        "input_channels": in_ch,
        "mel_diagnostics": mel_diag,
        "input_shape": [1, in_ch, n_bands, n_frames],
        "n_bands": n_bands,
        "n_frames": n_frames,
        "downsample": MN_DOWNSAMPLE,
        "feature_map": [fmap_f, fmap_t],
        "feat_dim": feat_dim,
        "head": f"global avg-pool → ({feat_dim},) → Linear(num_classes)",
        "native_shape": [MN_INPUT_CHANNELS, MN_NATIVE_BANDS, MN_NATIVE_FRAMES],
        "native_feature_map": [_mn_reduce(MN_NATIVE_BANDS), _mn_reduce(MN_NATIVE_FRAMES)],
        "on_contract": on_contract,
        "total_params": _efficientat_param_count_safe(str(backbone), input_channels=in_ch),
        "verdicts": verdicts,
    }


def _efficientat_param_count_safe(backbone: str, *, input_channels: int = 1) -> int | None:
    """Cached scratch-build count; None when torch is unavailable so the
    otherwise-analytic report stays usable in slim environments."""
    try:
        from sed.train.backbones import efficientat_param_count

        return efficientat_param_count(backbone, input_channels=input_channels)
    except Exception:  # noqa: BLE001
        return None


def _require_efficientat(frontend: str) -> str:
    value = str(frontend or "efficientat").lower()
    if value != "efficientat":
        raise ValueError(
            f"unsupported features.frontend {frontend!r} — only 'efficientat' is "
            "supported (the legacy_torchaudio_db frontend was removed; retrain "
            "the model or pin an older repo revision)"
        )
    return value


def _require_supported_frontend(frontend: str) -> str:
    value = str(frontend or "efficientat").lower()
    if value not in FEATURE_FRONTENDS:
        raise ValueError(
            f"unsupported features.frontend {frontend!r} — supported frontends are "
            "'efficientat' and 'ced' (the legacy_torchaudio_db frontend was "
            "removed; retrain the model or pin an older repo revision)"
        )
    return value


def window_sample_count(sample_rate: int, window_seconds: float) -> int:
    """Samples in one model window.

    Single source of truth (round, not int) so the training crop, export
    tracing, calibration windows, and the live ring buffer agree even when
    ``window_seconds * sample_rate`` is not an integer.
    """
    return int(round(float(window_seconds) * int(sample_rate)))


def feature_frame_count(
    *,
    sample_rate: int = 32000,
    window_seconds: float = 1.0,
    hop_length: int = 320,
    n_fft: int = 1024,
    frontend: str = "efficientat",
    win_length: int | None = None,
    stft_engine: str = "librosa",
) -> int:
    """Return the number of time frames produced for one model window."""
    frontend_value = _require_supported_frontend(frontend)
    _ = n_fft
    n = window_sample_count(sample_rate, window_seconds)
    if frontend_value == "ced":
        return 1 + max(0, n // hop_length)
    if str(stft_engine or "librosa").lower() == "scipy":
        # scipy.signal.spectrogram frames without centering/padding:
        # 1 + floor((len - nperseg) / hop) full segments. A window shorter than
        # one segment is zero-padded up to nperseg by `_scipy_psd_power_np`, so
        # it still yields exactly one frame — reporting 0 here would hand export
        # and the shape validators a model input of width 0.
        win = int(win_length or n_fft)
        if n < win:
            return 1
        return 1 + (n - win) // hop_length
    # EfficientAT first applies a two-tap pre-emphasis conv without padding,
    # shortening the waveform by one sample. torch.stft(center=True) then
    # yields 1 + floor(len / hop) frames.
    n = max(0, n - 1)
    return 1 + max(0, n // hop_length)


def feature_frame_count_for_cfg(cfg) -> int:
    """``feature_frame_count`` for a full Config — reads the engine + geometry.

    The one place training, export, inference, and diagnostics should call so
    the legacy scipy engine's (uncentered) frame count is honored everywhere the
    model input width is derived, not just the librosa/efficientat default.
    """
    f = cfg.features
    # A colormap image is resized to a fixed square, so the model-input width is
    # image_size regardless of the underlying spectrogram frame count.
    if str(getattr(f, "colormap", "none")).lower() != "none":
        return int(getattr(f, "image_size", 96))
    return feature_frame_count(
        sample_rate=int(cfg.model.sample_rate),
        window_seconds=float(f.window_seconds),
        hop_length=int(f.hop_length),
        n_fft=int(f.n_fft),
        frontend=str(f.frontend),
        win_length=int(getattr(f, "win_length", 0) or 0) or None,
        stft_engine=str(getattr(f, "stft_engine", "librosa")),
    )


def _kaldi_mel_banks_np(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Numpy equivalent of torchaudio.compliance.kaldi.get_mel_banks.

    EfficientAT pads Kaldi's ``n_fft/2`` filters with one zero Nyquist column
    before multiplying by the power spectrum. VTLN is unused by EfficientAT
    inference, so this implements the straight triangular mel-bank path.
    """
    num_fft_bins = n_fft // 2
    nyquist = 0.5 * sample_rate
    if fmax <= 0.0:
        fmax += nyquist
    if not (0.0 <= fmin < nyquist and 0.0 < fmax <= nyquist and fmin < fmax):
        raise ValueError(f"bad mel limits: fmin={fmin}, fmax={fmax}, nyquist={nyquist}")

    def mel_scale(freq):
        return 1127.0 * np.log1p(freq / 700.0)

    mel_low = mel_scale(fmin)
    mel_high = mel_scale(fmax)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)

    bins = np.arange(n_mels, dtype=np.float32)[:, None]
    left = mel_low + bins * mel_delta
    center = mel_low + (bins + 1.0) * mel_delta
    right = mel_low + (bins + 2.0) * mel_delta

    fft_bin_width = sample_rate / n_fft
    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float32))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    bank = np.maximum(0.0, np.minimum(up, down))
    return np.pad(bank, ((0, 0), (0, 1)), mode="constant").astype(np.float32)


def kaldi_mel_center_freqs(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Center frequencies (Hz) of the Kaldi triangular mel banks above.

    The efficientat frontend uses the HTK/Kaldi mel scale (1127·ln(1+f/700)),
    NOT librosa's default Slaney scale — labeling mel rows with
    ``librosa.mel_frequencies`` drifts hundreds of Hz off the real filter
    centers in the upper bands.
    """
    mel_low = 1127.0 * np.log1p(float(fmin) / 700.0)
    mel_high = 1127.0 * np.log1p(float(fmax) / 700.0)
    mel_delta = (mel_high - mel_low) / (int(n_mels) + 1)
    centers = mel_low + (np.arange(int(n_mels), dtype=np.float64) + 1.0) * mel_delta
    return 700.0 * np.expm1(centers / 1127.0)


def mel_filter_diagnostics(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float, fmax: float
) -> dict:
    """Detect empty (all-zero) mel filters for a mel geometry.

    When ``n_mels`` is high relative to ``n_fft`` the narrowest (lowest) HTK
    triangles fall between FFT bin centers and capture zero power; those rows
    sit exactly at the feature floor and render as solid black strips at the
    bottom of the spectrogram. This is a *design* diagnostic (warn, never
    block): reports how many rows are empty and the largest ``n_mels`` that
    keeps every filter non-empty for the same n_fft/sr/fmin/fmax.
    """
    def _empty_rows(k: int) -> np.ndarray:
        bank = _kaldi_mel_banks_np(int(k), int(n_fft), int(sample_rate), float(fmin), float(fmax))
        return np.flatnonzero(bank.sum(axis=1) <= 0.0)

    empty = _empty_rows(int(n_mels))
    max_usable = int(n_mels)
    if empty.size:
        lo, hi = 1, int(n_mels)   # bisect the largest k with zero empty filters
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _empty_rows(mid).size:
                hi = mid - 1
            else:
                lo = mid
        max_usable = lo
    return {
        "n_mels": int(n_mels),
        "empty_rows": int(empty.size),
        "empty_row_indices": [int(i) for i in empty[:8]],
        "max_usable_n_mels": int(max_usable),
    }


def mel_geometry_warning(features, sample_rate: int) -> tuple[dict | None, dict | None]:
    """(verdict, diagnostics) for empty-mel-filter geometries; (None, diag) when clean.

    Warn-only by project convention — training still runs; the black rows are
    truthful (those filters capture zero power) and this names the cause.
    """
    if str(getattr(features, "feature_type", "mel")) != "mel":
        return None, None
    if str(getattr(features, "frontend", "efficientat")).lower() != "efficientat":
        return None, None  # the CED frontend pins a known-good 64-mel geometry
    try:
        diag = mel_filter_diagnostics(
            int(features.n_mels), int(features.n_fft), int(sample_rate),
            float(features.fmin), float(features.fmax),
        )
    except Exception:  # noqa: BLE001 — invalid limits already fail config validation
        return None, None
    if not diag["empty_rows"]:
        return None, diag
    plural = "s" if diag["empty_rows"] != 1 else ""
    verdict = {
        "level": "warn", "severity": "medium", "kind": "mel_empty_filters",
        "text": (
            f"n_mels={diag['n_mels']} is too high for n_fft={int(features.n_fft)} @ "
            f"{int(sample_rate)} Hz — {diag['empty_rows']} mel filter{plural} capture zero "
            "power and render as solid black rows (mostly at the low-frequency bottom). "
            f"Max n_mels with no empty filters here: {diag['max_usable_n_mels']} "
            "(or raise n_fft / fmin)."
        ),
    }
    return verdict, diag


class MelExtractor:
    """Waveform → mel features, as compression then normalization.

    Everything up to the mel power spectrogram is byte-identical to the vendored
    EfficientAT preprocessing (pre-emphasis, STFT, Kaldi mel banks). After that
    come the two configurable stages: ``compression`` (log by default, the
    AudioSet contract) and ``scaling``, the single normalization stage.

    Callers that pass neither get the full EfficientAT contract — log plus its
    published (x + 4.5) / 5 affine, expressed as an ordinary fixed z-score. The
    legacy ``standardize`` arguments remain accepted and build the equivalent
    dataset-sourced z-score spec.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 800,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: int = 15000,
        frontend: str = "efficientat",
        device: str = "cpu",
        standardize: bool = False,
        standardize_mean: float = 0.0,
        standardize_std: float = 1.0,
        scaling: ScalingSpec | None = None,
        compression: str = "log",
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.frontend = _require_efficientat(frontend)
        self.compression = str(compression or "log").lower()
        self.standardize = bool(standardize)
        self.standardize_mean = float(standardize_mean)
        self.standardize_std = float(standardize_std)
        if self.standardize and self.standardize_std <= 0:
            raise ValueError("standardize_std must be positive")
        if scaling is not None:
            self.scaling = scaling
        elif self.standardize:
            self.scaling = ScalingSpec(
                method="zscore",
                a=np.asarray([self.standardize_mean], dtype=np.float32),
                b=np.asarray([self.standardize_std], dtype=np.float32),
                scope="global", source="dataset",
            )
        elif self.compression == "log":
            self.scaling = efficientat_fast_norm_spec()
        else:
            self.scaling = ScalingSpec()

        if _HAS_TORCH:
            self.window = torch.hann_window(win_length, periodic=False).to(device)
            mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
                n_mels,
                n_fft,
                sample_rate,
                float(fmin),
                float(fmax),
                vtln_low=100.0,
                vtln_high=-500.0,
                vtln_warp_factor=1.0,
            )
            mel_basis = torch.nn.functional.pad(
                mel_basis, (0, 1), mode="constant", value=0
            )
            self.mel_basis = mel_basis.to(device)

    def __call__(self, waveform):
        """waveform: 1-d numpy array or torch tensor (samples,) or (B, samples).

        Returns fast-normalized log-mel of shape (B, n_mels, T).
        """
        if not _HAS_TORCH:
            return self._efficientat_numpy(np.asarray(waveform))
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return self._efficientat_torch(waveform)

    def _efficientat_torch(self, waveform):
        waveform = waveform[:, 1:] - 0.97 * waveform[:, :-1]
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel_basis = self.mel_basis.to(device=waveform.device, dtype=power.dtype)
        melspec = torch.matmul(mel_basis, power)
        return apply_feature_scaling_torch(
            compress_power_torch(melspec, self.compression), self.scaling
        )

    # --- numpy fallback for inference-only (torch-free) deployments ------

    def _efficientat_numpy(self, waveform: np.ndarray) -> np.ndarray:
        import librosa
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        window = hann(self.win_length, sym=True).astype(np.float32)
        mel_basis = _kaldi_mel_banks_np(
            self.n_mels,
            self.n_fft,
            self.sample_rate,
            float(self.fmin),
            float(self.fmax),
        )
        out = []
        for w in waveform:
            w = w.astype(np.float32, copy=False)
            if len(w) >= 2:
                w = w[1:] - 0.97 * w[:-1]
            else:
                w = np.zeros((0,), dtype=np.float32)
            stft = librosa.stft(
                w,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
            )
            power = np.abs(stft).astype(np.float32) ** 2
            mel = mel_basis @ power
            out.append(apply_feature_scaling(
                compress_power(
                    mel, self.compression,
                    sample_rate=self.sample_rate, hop_length=self.hop_length,
                ),
                self.scaling,
            ))
        return np.stack(out, axis=0).astype(np.float32)


class CedFeatureExtractor:
    """CED dB-mel frontend used by the ``mispeech/ced-*`` family.

    Returns ``(B, n_mels, T)`` with a 1-channel model input supplied later by
    the datamodule. The default geometry follows the shared CED HF preprocessor:
    16 kHz, 64 mel, n_fft=512, win_length=512, hop_length=160.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        fmin: int = 0,
        fmax: int = 8000,
        device: str = "cpu",
        scaling: ScalingSpec | None = None,
        **_ignored,
    ):
        # CED's power→dB mapping IS its compression stage (compression="db",
        # pinned to the checkpoint's own top_db=120 implementation). Whatever
        # normalization is configured then applies on top of it, exactly as for
        # log-mel. Silently dropping it would train on one distribution and
        # deploy on another.
        self.compression = "db"
        self.scaling = scaling if scaling is not None else ScalingSpec()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax else None
        if _HAS_TORCH:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=self.fmin,
                f_max=self.fmax,
                n_mels=self.n_mels,
                center=True,
                power=2.0,
                normalized=False,
                mel_scale="htk",
            ).to(device)
            self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=120).to(device)

    def __call__(self, waveform):
        if not _HAS_TORCH:
            return self._numpy(np.asarray(waveform))
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)
        return apply_feature_scaling_torch(self.db(mel), self.scaling)

    def _numpy(self, waveform: np.ndarray) -> np.ndarray:
        import librosa

        if waveform.ndim == 1:
            waveform = waveform[None, :]
        out = []
        for w in waveform:
            mel = librosa.feature.melspectrogram(
                y=np.asarray(w, dtype=np.float32),
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0,
                center=True,
                # torchaudio's Spectrogram pads with "reflect"; librosa defaults
                # to "constant". Left unmatched, the torch-free fallback drifts
                # up to ~20 dB from the trained frontend on the frames touching
                # the clip edges — exactly where an onset-centred transient sits.
                pad_mode="reflect",
                htk=True,
                norm=None,
            )
            out.append(apply_feature_scaling(
                librosa.power_to_db(mel, ref=1.0, top_db=120).astype(np.float32),
                self.scaling,
            ))
        return np.stack(out, axis=0).astype(np.float32)


def waveform_to_logmel(
    wave: np.ndarray,
    sample_rate: int = 32000,
    **kwargs,
) -> np.ndarray:
    """Convenience for one-shot mel extraction in pure numpy land."""
    extractor = MelExtractor(sample_rate=sample_rate, **kwargs)
    feats = extractor(wave)
    if _HAS_TORCH and isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    return feats


def waveform_to_ced_dbmel(
    wave: np.ndarray,
    sample_rate: int = 16000,
    **kwargs,
) -> np.ndarray:
    """Convenience for one-shot CED dB-mel extraction."""
    extractor = CedFeatureExtractor(sample_rate=sample_rate, **kwargs)
    feats = extractor(wave)
    if _HAS_TORCH and isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()
    return feats


def _stft_power_np(
    wave: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> np.ndarray:
    """Pre-emphasis + STFT magnitude² (full rfft bins, T). Shared by linear + mel-power."""
    import librosa

    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    if len(wave) >= 2:
        wave = wave[1:] - 0.97 * wave[:-1]
    else:
        wave = np.zeros((0,), dtype=np.float32)
    window = hann(win_length, sym=True).astype(np.float32)
    stft = librosa.stft(
        wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, pad_mode="reflect",
    )
    return np.abs(stft).astype(np.float32) ** 2


def _scipy_psd_power_np(
    wave: np.ndarray,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> np.ndarray:
    """Legacy-exact scipy PSD spectrogram (full rfft bins, T).

    Byte-for-byte the legacy soundDetection preprocessing:
    ``scipy.signal.spectrogram(int16_wave, fs, nperseg=(spec-1)*2)`` with the
    scipy defaults that call implied — Tukey(0.25) window, ``noverlap =
    nperseg - hop``, per-segment constant detrend, one-sided PSD density
    scaling, no pre-emphasis, no centering. Amplitudes are scaled back to int16
    full-scale (the legacy system read WAVs as raw int16), so the fixed
    -20..60 dB normalization range keeps its original meaning.
    """
    from scipy import signal as sps

    wave = np.asarray(wave)
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    wave = wave.astype(np.float64) * LEGACY_INT16_SCALE
    if len(wave) < int(win_length):
        wave = np.pad(wave, (0, int(win_length) - len(wave)))
    _, _, sxx = sps.spectrogram(
        wave,
        fs=int(sample_rate),
        window=("tukey", 0.25),
        nperseg=int(win_length),
        noverlap=int(win_length) - int(hop_length),
        nfft=int(n_fft),
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    return sxx.astype(np.float32)


def _linear_freq_mask(n_fft: int, sample_rate: int, fmin: float, fmax: float) -> np.ndarray:
    """Boolean rfft-bin mask for [fmin, fmax]; the linear feature's band selection."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    lo = float(fmin) if fmin else 0.0
    hi = float(fmax) if fmax and fmax > 0 else float(freqs[-1])
    keep = (freqs >= lo) & (freqs <= hi)
    return keep if keep.any() else np.ones_like(freqs, dtype=bool)


def _linear_power_np(
    wave: np.ndarray,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    stft_engine: str = "librosa",
) -> np.ndarray:
    """Linear STFT power cropped to [fmin, fmax] → (n_bins, T)."""
    if str(stft_engine or "librosa").lower() == "scipy":
        power = _scipy_psd_power_np(wave, sample_rate, n_fft, win_length, hop_length)
    else:
        power = _stft_power_np(wave, n_fft, win_length, hop_length)
    return power[_linear_freq_mask(n_fft, sample_rate, fmin, fmax), :]


def _mel_power_np(
    wave: np.ndarray,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Mel power (no log) using the same Kaldi banks as the efficientat frontend."""
    power = _stft_power_np(wave, n_fft, win_length, hop_length)
    mel_basis = _kaldi_mel_banks_np(n_mels, n_fft, sample_rate, float(fmin), float(fmax))
    return (mel_basis @ power).astype(np.float32)


def band_power_per_row(
    wave: np.ndarray,
    sample_rate: int,
    *,
    feature_type: str,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    stft_engine: str = "librosa",
) -> np.ndarray:
    """True per-band signal energy (summed over time) for the displayed feature's
    rows — mel-band power for mel, cropped linear power for linear. Used for
    band-energy readouts and noise-overlay survival (independent of log/PCEN)."""
    if str(feature_type) == "linear":
        power = _linear_power_np(
            wave, sample_rate, n_fft, win_length, hop_length, fmin, fmax,
            stft_engine=stft_engine,
        )
    else:
        power = _mel_power_np(wave, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax)
    return power.sum(axis=1).astype(np.float64)


def _apply_pcen(
    power: np.ndarray,
    sample_rate: int,
    hop_length: int,
    params: dict | None = None,
) -> np.ndarray:
    """Per-Channel Energy Normalization (librosa.pcen) over a power spectrogram."""
    import librosa

    p = dict(PCEN_DEFAULTS)
    if params:
        p.update({k: params[k] for k in PCEN_DEFAULTS if k in params})
    return librosa.pcen(
        np.asarray(power, dtype=np.float32),
        sr=sample_rate,
        hop_length=hop_length,
        gain=float(p["gain"]),
        bias=float(p["bias"]),
        power=float(p["power"]),
        time_constant=float(p["time_constant"]),
        eps=float(p["eps"]),
    ).astype(np.float32)


def waveform_to_linear(
    wave: np.ndarray,
    sample_rate: int = 32000,
    *,
    n_fft: int = 1024,
    win_length: int = 800,
    hop_length: int = 320,
    fmin: float = 0,
    fmax: float = 15000,
    log_power: bool = True,
    stft_engine: str = "librosa",
    db_norm: bool = False,
    db_min: float = -20.0,
    db_max: float = 60.0,
    standardize: bool = False,
    standardize_mean: float = 0.0,
    standardize_std: float = 1.0,
    scaling: ScalingSpec | None = None,
    compression: str | None = None,
    pcen_params: dict | None = None,
    **_ignored,
) -> np.ndarray:
    """Linear STFT spectrogram (n_bins, T), compressed then normalized.

    With the default librosa engine it mirrors the mel frontend's pre-emphasis +
    STFT so the two views are directly comparable; the scipy engine reproduces
    the legacy soundDetection PSD instead (see ``_scipy_psd_power_np``).
    Frequency bins are cropped to [fmin, fmax], the values are compressed
    (``compression``), then the single normalization stage is applied.

    The legacy ``db_norm`` / ``log_power`` / ``standardize`` arguments are still
    accepted and are translated into that same pair, so old call sites produce
    identical output.
    """
    power = _linear_power_np(
        wave, sample_rate, n_fft, win_length, hop_length, fmin, fmax,
        stft_engine=stft_engine,
    )
    kind = str(compression).lower() if compression else (
        "db" if db_norm else ("log" if log_power else "none")
    )
    values = compress_power(
        power, kind, sample_rate=sample_rate, hop_length=hop_length,
        pcen_params=pcen_params,
    )
    if scaling is None:
        if db_norm:
            # The legacy fixed-range dB image map is a fixed min-max, and
            # apply_feature_scaling's minmax already clips to [0, 1].
            scaling = ScalingSpec(
                method="minmax",
                a=np.asarray([float(db_min)], dtype=np.float32),
                b=np.asarray([float(db_max)], dtype=np.float32),
                scope="global", source="fixed",
            )
        elif standardize:
            if float(standardize_std) <= 0.0:
                raise ValueError("standardize_std must be positive")
            scaling = ScalingSpec(
                method="zscore",
                a=np.asarray([float(standardize_mean)], dtype=np.float32),
                b=np.asarray([float(standardize_std)], dtype=np.float32),
                scope="global", source="dataset",
            )
        else:
            scaling = ScalingSpec()
    return apply_feature_scaling(values, scaling)


def waveform_to_feature(
    wave: np.ndarray,
    sample_rate: int = 32000,
    *,
    feature_type: str = "mel",
    pcen: bool = False,
    pcen_params: dict | None = None,
    n_fft: int = 1024,
    win_length: int = 800,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 0,
    fmax: float = 15000,
    log_power: bool = True,
    stft_engine: str = "librosa",
    db_norm: bool = False,
    db_min: float = -20.0,
    db_max: float = 60.0,
    frontend: str = "efficientat",
    standardize: bool = False,
    standardize_mean: float = 0.0,
    standardize_std: float = 1.0,
    scaling: ScalingSpec | None = None,
    scaling_method: str = "none",
    scaling_scope: str = "global",
    scaling_source: str = "dataset",
    scaling_stats: dict | None = None,
    compression: str | None = None,
    **_ignored,
) -> np.ndarray:
    """One clip → (n_bands, T). Single numpy source of truth, shared by the
    preview, the trainable FeatureExtractor, and analysis.

    Three stages, identical for every feature type:

    1. **basis** — linear STFT bins (librosa or the legacy scipy PSD engine) or
       Kaldi mel banks, cropped to [fmin, fmax]
    2. **compression** — ``none`` | ``log`` | ``db`` | ``pcen``
    3. **normalization** — the ``scaling`` spec: nothing, a z-score, or a
       min-max, with constants either fixed (a checkpoint contract, e.g. the
       AudioSet fast-norm or the legacy [-20, 60] dB map) or fitted on the
       dataset.

    The legacy ``pcen`` / ``log_power`` / ``db_norm`` / ``standardize`` flags are
    still accepted and map onto those stages unchanged.
    """
    feature_type = str(feature_type)
    if scaling is None:
        scaling = scaling_spec_from_stats(
            scaling_method, scaling_stats, scaling_scope, scaling_source
        )
    if str(frontend or "efficientat").lower() == "ced" and feature_type != "linear":
        return waveform_to_ced_dbmel(
            wave, sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax,
            scaling=scaling,
        )[0]
    kind = str(compression).lower() if compression else (
        "pcen" if pcen else ("db" if db_norm else ("log" if log_power else "none"))
    )
    if feature_type == "linear":
        return waveform_to_linear(
            wave, sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, fmin=fmin, fmax=fmax,
            stft_engine=stft_engine, db_min=db_min, db_max=db_max,
            compression=kind, pcen_params=pcen_params,
            # A legacy db_norm call carries no explicit spec; waveform_to_linear
            # then builds the fixed [db_min, db_max] min-max itself.
            scaling=None if (scaling is None or not scaling.declared) and db_norm else scaling,
            db_norm=db_norm,
        )
    if kind == "pcen":
        # PCEN has no torch path, so it works from the shared numpy mel power.
        power = _mel_power_np(
            wave, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax
        )
        return apply_feature_scaling(
            compress_power(
                power, "pcen", sample_rate=sample_rate, hop_length=hop_length,
                pcen_params=pcen_params,
            ),
            scaling,
        )
    # Everything else stays on MelExtractor so training, export and this preview
    # share one implementation of the EfficientAT mel pipeline.
    return waveform_to_logmel(
        wave, sample_rate, frontend=frontend, n_fft=n_fft, win_length=win_length,
        hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax,
        compression=kind,
        standardize=standardize, standardize_mean=standardize_mean,
        standardize_std=standardize_std,
        scaling=scaling if scaling.declared else None,
    )[0]


class LinearExtractor:
    """Batched linear-STFT frontend with a NumPy fallback.

    The operation order matches the EfficientAT-style mel frontend through the
    STFT: pre-emphasis 0.97, centered symmetric-Hann analysis, power, frequency
    cropping — then the same two configurable stages every feature has,
    ``compression`` and ``scaling``. The legacy scipy PSD engine, PCEN and the
    colormap render stay in :class:`FeatureExtractor`.
    """

    def __init__(
        self,
        sample_rate: int = 44_100,
        n_fft: int = 1024,
        win_length: int = 882,
        hop_length: int = 220,
        fmin: int = 80,
        fmax: int = 20_000,
        *,
        standardize: bool = False,
        standardize_mean: float = 0.0,
        standardize_std: float = 1.0,
        scaling: ScalingSpec | None = None,
        compression: str = "log",
        device: str = "cpu",
    ):
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.fmin = int(fmin)
        self.fmax = int(fmax)
        self.compression = str(compression or "log").lower()
        self.standardize = bool(standardize)
        self.standardize_mean = float(standardize_mean)
        self.standardize_std = float(standardize_std)
        if self.standardize and self.standardize_std <= 0.0:
            raise ValueError("standardize_std must be positive")
        if scaling is not None:
            self.scaling = scaling
        elif self.standardize:
            self.scaling = ScalingSpec(
                method="zscore",
                a=np.asarray([self.standardize_mean], dtype=np.float32),
                b=np.asarray([self.standardize_std], dtype=np.float32),
                scope="global", source="dataset",
            )
        else:
            self.scaling = ScalingSpec()
        self._mask_np = _linear_freq_mask(
            self.n_fft, self.sample_rate, self.fmin, self.fmax,
        )
        if _HAS_TORCH:
            self.window = torch.hann_window(
                self.win_length, periodic=False, device=device,
            )
            self.frequency_mask = torch.from_numpy(self._mask_np).to(device=device)

    def __call__(self, waveform):
        """Return ``(B, linear_bins, frames)`` for one waveform or a batch."""
        if not _HAS_TORCH:
            return self._numpy(np.asarray(waveform))
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[:, 1:] - 0.97 * waveform[:, :-1]
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            pad_mode="reflect",
            normalized=False,
            window=window,
            return_complex=True,
        )
        mask = self.frequency_mask.to(device=waveform.device)
        power = spectrum.abs().square()[:, mask, :]
        return apply_feature_scaling_torch(
            compress_power_torch(power, self.compression), self.scaling
        )

    def _numpy(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        return np.stack([
            waveform_to_linear(
                w,
                self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                stft_engine="librosa",
                compression=self.compression,
                scaling=self.scaling,
            )
            for w in waveform
        ], axis=0).astype(np.float32)


class FeatureExtractor:
    """Trainable extractor for the non-pretrained-contract features (linear / PCEN).

    Mirrors ``MelExtractor.__call__`` semantics — input ``(samples,)`` or
    ``(B, samples)``, output ``(B, n_bands, T)`` — but computes the configured
    mel|linear × log|PCEN feature via the shared numpy ``waveform_to_feature`` so
    training and inference are byte-identical. The pretrained mel+efficientat
    path keeps using ``MelExtractor`` (see ``build_feature_extractor``).
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 800,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: int = 15000,
        *,
        feature_type: str = "mel",
        pcen: bool = False,
        pcen_params: dict | None = None,
        log_power: bool = True,
        stft_engine: str = "librosa",
        db_norm: bool = False,
        db_min: float = -20.0,
        db_max: float = 60.0,
        colormap: str = "none",
        image_size: int = 96,
        frontend: str = "efficientat",
        standardize: bool = False,
        standardize_mean: float = 0.0,
        standardize_std: float = 1.0,
        scaling: ScalingSpec | None = None,
        compression: str | None = None,
        device: str = "cpu",  # accepted for drop-in parity; numpy path is CPU
    ):
        self.sample_rate = sample_rate
        self.colormap = str(colormap or "none").lower()
        self.image_size = int(image_size)
        self.kwargs = dict(
            feature_type=str(feature_type),
            compression=compression,
            pcen=bool(pcen),
            pcen_params=dict(pcen_params) if pcen_params else None,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            log_power=log_power,
            stft_engine=str(stft_engine or "librosa"),
            db_norm=bool(db_norm),
            db_min=float(db_min),
            db_max=float(db_max),
            frontend=frontend,
            standardize=bool(standardize),
            standardize_mean=float(standardize_mean),
            standardize_std=float(standardize_std),
            scaling=scaling if scaling is not None else ScalingSpec(),
        )

    def __call__(self, waveform):
        """Input ``(samples,)`` or ``(B, samples)``. Output ``(B, n_bands, T)`` for
        a spectrogram feature, or ``(B, 3, image_size, image_size)`` when a colormap
        image render is configured (the legacy soundDetection model input)."""
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        out = []
        for w in arr:
            spec = waveform_to_feature(w, self.sample_rate, **self.kwargs)
            if self.colormap != "none":
                spec = render_spectrogram_image(spec, self.colormap, self.image_size)
            out.append(spec)
        return np.stack(out, axis=0).astype(np.float32)


def build_feature_extractor(cfg):
    """Return the right extractor for a Config: the verified ``MelExtractor`` for
    the mel+efficientat pretrained contract, else the general ``FeatureExtractor``.

    Used by training, inference, calibration, and export so one feature definition
    drives the whole pipeline.
    """
    # Config validation catches normal load/snapshot paths. This second guard
    # also covers programmatic callers that mutated a Config in place before
    # entering training, evaluation, calibration, export, or live inference.
    from sed.config import validate_clink_cnn_v2_feature_contract

    validate_clink_cnn_v2_feature_contract(cfg, phase="feature extraction")
    f = cfg.features
    sr = int(cfg.model.sample_rate)
    # Resolve the one normalization stage once, validating per-band vector
    # lengths against the actual feature geometry.
    scaling = resolve_feature_scaling(f, n_bands=feature_band_count(f, sr))
    compression = feature_compression(f)
    if str(getattr(f, "frontend", "efficientat")).lower() == "ced":
        return CedFeatureExtractor(
            sample_rate=sr,
            n_fft=f.n_fft,
            win_length=f.win_length,
            hop_length=f.hop_length,
            n_mels=f.n_mels,
            fmin=f.fmin,
            fmax=f.fmax,
            device="cpu",
            scaling=scaling,
        )
    # A colormap image render is always a FeatureExtractor (never the plain
    # Mel/Linear extractor), even for a mel base, so the image stage is applied.
    # Every non-PCEN mel stays on MelExtractor whatever its compression and
    # normalization: those are the two stages after a shared STFT/mel pipeline.
    if (str(f.feature_type) == "mel" and compression != "pcen"
            and str(getattr(f, "colormap", "none")).lower() == "none"):
        return MelExtractor(
            sample_rate=sr,
            n_fft=f.n_fft,
            win_length=f.win_length,
            hop_length=f.hop_length,
            n_mels=f.n_mels,
            fmin=f.fmin,
            fmax=f.fmax,
            frontend=f.frontend,
            device="cpu",
            compression=compression,
            scaling=scaling,
        )
    if (
        str(f.feature_type) == "linear"
        and compression in ("none", "log", "db")
        and str(getattr(f, "stft_engine", "librosa")) == "librosa"
        and str(getattr(f, "colormap", "none")) == "none"
    ):
        return LinearExtractor(
            sample_rate=sr,
            n_fft=f.n_fft,
            win_length=f.win_length,
            hop_length=f.hop_length,
            fmin=f.fmin,
            fmax=f.fmax,
            compression=compression,
            scaling=scaling,
            device="cpu",
        )
    return FeatureExtractor(
        sample_rate=sr,
        n_fft=f.n_fft,
        win_length=f.win_length,
        hop_length=f.hop_length,
        n_mels=f.n_mels,
        fmin=f.fmin,
        fmax=f.fmax,
        feature_type=f.feature_type,
        compression=compression,
        pcen_params=pcen_params_from_features(f),
        stft_engine=str(getattr(f, "stft_engine", "librosa")),
        db_min=float(getattr(f, "db_min", -20.0)),
        db_max=float(getattr(f, "db_max", 60.0)),
        colormap=str(getattr(f, "colormap", "none")),
        image_size=int(getattr(f, "image_size", 96)),
        frontend=f.frontend,
        scaling=scaling,
    )
