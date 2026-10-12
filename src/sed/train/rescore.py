"""Re-derive frozen-test metrics from stored scores, at any operating point.

`sed.train.eval` scores a run once and writes the probability matrix to the
score store. Everything a reader might want to know afterwards — metrics at a
different confidence, a confusion matrix at a different margin, recall at a
different false-alarm budget — is a re-cut of that matrix, not another forward
pass. This module is that re-cut.

It is the only path the Runs & compare tab uses, and it calls the same
`sed.train.metrics` functions the evaluator called, so a number moved by a
slider is the same measurement as the number frozen in the manifest.

Torch-free and cached: the browser asks for 60+ runs at once whenever a control
changes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sed.train import score_store
from sed.train.metrics import (
    argmax_metrics,
    closed_set_decisions,
    confusion_matrix,
    cross_target_confusion,
    decision_metrics,
    fit_thresholds_for_precision,
    fit_thresholds_from_val,
    runtime_pred_indices,
)
from sed.train.operating_point import (
    band_pauc,
    operating_window,
    recall_at_budget,
    sweep,
)

log = logging.getLogger(__name__)

NEGATIVE_CLASS = "_negative"

# How a stored score matrix becomes a pass/fail. `shared` is the benchmark — one
# confidence for every model, so the column compares models. The others put each
# model at its own operating point, so they compare deployments.
DECISION_MODES: tuple[str, ...] = ("shared", "deployed", "val_f1", "val_precision", "argmax")

# The recall a venue product is expected to hold; the operating window is
# measured against it.
DEFAULT_RECALL_FLOOR = 0.90


@dataclass(frozen=True)
class Decision:
    """One operating point, as the UI's control bar expresses it."""

    confidence: float = 0.90
    negative_max: float = 0.50
    margin: float = 0.08
    mode: str = "shared"
    target_precision: float = 0.90
    budget_per_1000: float = 1.0
    recall_floor: float = DEFAULT_RECALL_FLOOR

    @classmethod
    def from_payload(cls, data: dict | None) -> "Decision":
        """Build from a request body, clamping rather than rejecting.

        A slider cannot send something meaningful out of range, and failing a
        whole table because one field arrived as a string is worse than using
        the default for it.
        """
        data = data if isinstance(data, dict) else {}

        def _num(key: str, fallback: float, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(data.get(key, fallback))))
            except (TypeError, ValueError):
                return fallback

        mode = str(data.get("mode") or "shared").strip().lower()
        return cls(
            confidence=_num("confidence", 0.90, 0.0, 1.0),
            negative_max=_num("negative_max", 0.50, 0.0, 1.0),
            margin=_num("margin", 0.08, 0.0, 1.0),
            mode=mode if mode in DECISION_MODES else "shared",
            target_precision=_num("target_precision", 0.90, 0.05, 0.999),
            budget_per_1000=_num("budget_per_1000", 1.0, 0.0, 1000.0),
            recall_floor=_num("recall_floor", DEFAULT_RECALL_FLOOR, 0.0, 1.0),
        )

    @property
    def fingerprint(self) -> tuple:
        """Cache key — every field that can change a number."""
        return (
            round(self.confidence, 6), round(self.negative_max, 6), round(self.margin, 6),
            self.mode, round(self.target_precision, 6),
            round(self.budget_per_1000, 6), round(self.recall_floor, 6),
        )


# --- Score cache ----------------------------------------------------------
# Scores are immutable once written; a re-score writes a new file, so the file's
# size+mtime is enough to invalidate. Bounded because a project can hold dozens
# of runs and each matrix is a few hundred kB.

_CACHE: dict[tuple, tuple] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 96


def _load(run_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
    npz = score_store.scores_dir(run_dir) / f"{key}.npz"
    try:
        stat = npz.stat()
    except OSError:
        return None
    cache_key = (str(run_dir), key, stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
    if hit is not None:
        return hit
    loaded = score_store.read_scores(run_dir, key)
    if loaded is None:
        return None
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[cache_key] = loaded
    return loaded


# --- Threshold vectors ----------------------------------------------------

def _shared_thresholds(
    classes: list[str], decision: Decision, *, negative_classes: set[str] | None = None,
) -> dict[str, float]:
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    return {
        c: (decision.negative_max if c in negative_classes else decision.confidence)
        for c in classes
    }


def _deployed_thresholds(run_dir: Path, classes: list[str], decision: Decision) -> dict | None:
    """The run's own saved decision layer, if it has one."""
    from sed.config import load_runtime_settings

    try:
        settings = load_runtime_settings(run_dir / "runtime_settings.json")
    except Exception:
        return None
    saved = settings.get("thresholds")
    if not isinstance(saved, dict) or not saved:
        return None
    negative_classes = _run_negative_classes(run_dir, classes) or {NEGATIVE_CLASS}
    out = _shared_thresholds(classes, decision, negative_classes=negative_classes)
    for cls, value in saved.items():
        if cls in out and cls not in negative_classes:
            try:
                out[cls] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _run_config_snapshot(run_dir: Path):
    """This run's frozen config, or ``None`` when its snapshot can't be read."""
    from sed.config import load_config_snapshot

    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return None
    try:
        return load_config_snapshot(path)
    except Exception:
        return None


def _run_firing_classes(run_dir: Path, classes: list[str]) -> set[str] | None:
    """Which of this run's outputs may fire, read from its frozen config.

    The re-cut has to model the same decision layer the run deploys, so a
    suppressor class competes here too. A run whose snapshot cannot be read
    keeps the historical all-targets behaviour rather than guessing.
    """
    from sed.train.metrics import firing_classes

    cfg = _run_config_snapshot(run_dir)
    if cfg is None:
        return None
    try:
        return firing_classes(cfg, classes)
    except Exception:
        return None


def _run_class_roles(run_dir: Path, classes: list[str]) -> dict[str, str] | None:
    """This run's `Config.class_roles`, or ``None`` when its snapshot can't be read.

    Callers fall back to the historical "only `_negative` is non-target"
    default when this is ``None`` — a run whose snapshot cannot be read keeps
    scoring exactly as it did before roles existed.
    """
    cfg = _run_config_snapshot(run_dir)
    if cfg is None:
        return None
    try:
        return cfg.class_roles(list(classes))
    except Exception:
        return None


def _run_negative_classes(run_dir: Path, classes: list[str]) -> set[str] | None:
    """This run's negative-role classes, or ``None`` when roles can't be read."""
    roles = _run_class_roles(run_dir, classes)
    if roles is None:
        return None
    return {cls for cls, role in roles.items() if role == "negative"}


def _val_scores(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """The run's stored validation pass, if one exists."""
    for key, sidecar in score_store.available(run_dir).items():
        if str(sidecar.get("split") or "") == score_store.VAL_SPLIT:
            return _load(run_dir, key)
    return None


def _val_fitted_thresholds(
    run_dir: Path, classes: list[str], decision: Decision
) -> dict | None:
    """Per-class thresholds fitted on the run's stored val scores.

    Same fit the evaluator performs, but over the stored matrix rather than a
    fresh forward pass — so a val-fitted operating point is available for every
    run that has a val score store, with no GPU.
    """
    loaded = _val_scores(run_dir)
    if loaded is None:
        return None
    probs, true_idx, sidecar = loaded
    val_classes = list(sidecar.get("classes") or [])
    negative_classes = _run_negative_classes(run_dir, val_classes)
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    if val_classes != list(classes) or not (negative_classes & set(val_classes)):
        return None
    targets = score_store.one_hot(true_idx, len(val_classes))
    base = _shared_thresholds(val_classes, decision, negative_classes=negative_classes)
    try:
        if decision.mode == "val_precision":
            fitted = fit_thresholds_for_precision(
                probs, targets, val_classes,
                target_precision=decision.target_precision,
                negative_max=decision.negative_max,
                margin_to_second_best=decision.margin,
                base_thresholds=base,
                negative_classes=negative_classes,
            )
        else:
            fitted = fit_thresholds_from_val(
                probs, targets, val_classes,
                negative_max=decision.negative_max,
                margin_to_second_best=decision.margin,
                base_thresholds=base,
                negative_classes=negative_classes,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("val threshold fit failed for %s: %s", run_dir.name, exc)
        return None
    out = _shared_thresholds(classes, decision)
    out.update({k: float(v) for k, v in fitted.items() if k in out})
    return out


def resolve_thresholds(
    run_dir: Path, classes: list[str], decision: Decision
) -> tuple[dict[str, float], str]:
    """The per-class threshold vector for one run, plus where it came from.

    Falls back to the shared confidence when a per-model mode is unavailable for
    this run, and says so — a silently substituted decision layer would make two
    rows in the same column mean different things.
    """
    negative_classes = _run_negative_classes(run_dir, classes)
    if decision.mode == "deployed":
        found = _deployed_thresholds(run_dir, classes, decision)
        if found is not None:
            return found, "deployed"
        return (
            _shared_thresholds(classes, decision, negative_classes=negative_classes),
            "shared_fallback_no_deployed",
        )
    if decision.mode in ("val_f1", "val_precision"):
        found = _val_fitted_thresholds(run_dir, classes, decision)
        if found is not None:
            return found, decision.mode
        return (
            _shared_thresholds(classes, decision, negative_classes=negative_classes),
            "shared_fallback_no_val_scores",
        )
    return _shared_thresholds(classes, decision, negative_classes=negative_classes), "shared"


# --- The re-cut -----------------------------------------------------------

def rescore(run_dir: Path, key: str, decision: Decision) -> dict | None:
    """Full metric block for one run at one operating point, or None.

    Returns the same shape `sed.train.eval` stores, plus the operating-point
    summary the runs table ranks on.
    """
    loaded = _load(Path(run_dir), key)
    if loaded is None:
        return None
    probs, true_idx, sidecar = loaded
    classes = list(sidecar.get("classes") or [])
    if not classes or probs.shape[1] != len(classes):
        return None
    targets = score_store.one_hot(true_idx, len(classes))

    roles = _run_class_roles(Path(run_dir), classes)
    negative_classes = (
        {NEGATIVE_CLASS} if roles is None
        else {cls for cls, role in roles.items() if role == "negative"}
    )
    thresholds, thresholds_source = resolve_thresholds(Path(run_dir), classes, decision)
    if decision.mode == "argmax" or not (negative_classes & set(classes)):
        pred_idx, _decisions = closed_set_decisions(probs, classes)
        prediction_mode = "closed_set_argmax"
        margin = 0.0
    else:
        pred_idx = runtime_pred_indices(
            probs, classes, thresholds, decision.negative_max, decision.margin,
            trigger_classes=_run_firing_classes(Path(run_dir), classes),
            negative_classes=negative_classes,
        )
        prediction_mode = "runtime_thresholds"
        margin = decision.margin

    cm = confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    metrics = decision_metrics(
        probs, targets, classes, cm,
        thresholds=thresholds,
        threshold=decision.confidence if decision.mode == "shared" else None,
        margin_to_second_best=margin,
        budget_per_1000=decision.budget_per_1000,
        roles=roles,
    )

    curve = sweep(
        probs, targets, classes, negative_max=decision.negative_max, margin=margin,
        roles=roles,
    )
    at_budget = recall_at_budget(curve, decision.budget_per_1000)
    window = operating_window(curve, decision.budget_per_1000, decision.recall_floor)

    metrics.update({
        "classes": classes,
        "confusion": cm.tolist(),
        "prediction_mode": prediction_mode,
        "thresholds_source": thresholds_source,
        "negative_max": decision.negative_max,
        "margin_to_second_best": margin,
        "test_set_hash": sidecar.get("test_set_hash") or "",
        "engine": sidecar.get("engine") or "",
        "ir_precision": sidecar.get("ir_precision"),
        "score_source": sidecar.get("source") or "scored",
        "budget_per_1000": decision.budget_per_1000,
        "recall_floor": decision.recall_floor,
        # The headline: recall reachable inside the false-alarm budget, and the
        # confidence that reaches it. `None` means no threshold qualifies.
        "recall_at_budget": (at_budget or {}).get("recall_macro"),
        "threshold_at_budget": (at_budget or {}).get("threshold"),
        "false_at_budget": (at_budget or {}).get("false_per_1000"),
        "recall_at_budget_per_class": (at_budget or {}).get("recall_per_class") or {},
        "band_pauc": band_pauc(curve),
        "operating_window": window,
        "operating_curve": curve["points"],
        "separation": separation(
            probs, targets, classes, pred_idx, negative_classes=negative_classes,
        ),
        "margin_abstention": margin_abstention(
            probs, classes, thresholds, decision.negative_max, margin,
            negative_classes=negative_classes,
        ),
        # The one reading of this matrix that no control on the bar can move.
        # Stored alongside the gated metrics rather than behind a mode switch,
        # so "what the model knows" and "what the decision layer delivers" are
        # visible in the same row — their gap is the calibration headroom.
        "argmax": argmax_metrics(probs, targets, classes, roles=roles),
    })
    # The curve carries every threshold; the table only ever plots it, so drop
    # the per-class breakdown from the wire form.
    metrics["operating_curve"] = [
        {k: v for k, v in p.items() if k != "recall_per_class"}
        for p in metrics["operating_curve"]
    ]
    metrics.pop("cross_target_confusion", None)
    metrics["cross_target_confusion"] = cross_target_confusion(cm, classes, roles=roles)
    return metrics


def separation(
    probs, targets, classes: list[str], pred_idx, *,
    negative_classes: set[str] | None = None,
) -> dict | None:
    """How far apart the fired and the rejected actually sit.

    Two models can share an F1 while one puts its targets at 0.98 and its
    background at 0.02 and the other has both straddling the threshold. Only the
    first survives a venue whose noise floor moves, and no point metric
    distinguishes them.
    """
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    target_cols = [i for i, c in enumerate(classes) if c not in negative_classes]
    negative_idx = [i for i, c in enumerate(classes) if c in negative_classes]
    if not target_cols or not negative_idx or probs.shape[0] == 0:
        return None
    true_idx = np.argmax(targets, axis=1)
    is_negative = np.isin(true_idx, negative_idx)
    tgt = probs[:, target_cols]
    win = tgt.max(axis=1)

    correct = (~is_negative) & (np.asarray(pred_idx) == true_idx)
    if tgt.shape[1] > 1:
        second = np.sort(tgt, axis=1)[:, -2]
        margins = (win - second)[correct]
    else:
        margins = win[correct]
    return {
        # Median headroom between the winning target and its nearest rival on
        # the clips the model got right.
        "target_margin": float(np.median(margins)) if margins.size else None,
        # Median best-target score on background: how close the room gets to
        # firing. Lower is calmer.
        "negative_headroom": float(np.median(win[is_negative])) if is_negative.any() else None,
    }


def margin_abstention(
    probs, classes: list[str], thresholds: dict, negative_max: float, margin: float, *,
    negative_classes: set[str] | None = None,
) -> float | None:
    """Share of clips suppressed by the margin gate alone.

    With several genuinely different targets this gate does real work — it is
    what stops an ambiguous glass firing the wrong event — but nothing reports
    how often it triggers, so its cost in recall is invisible.
    """
    if margin <= 0 or probs.shape[0] == 0:
        return None
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    target_cols = [i for i, c in enumerate(classes) if c not in negative_classes]
    if len(target_cols) < 2:
        return None
    with_margin = runtime_pred_indices(
        probs, classes, thresholds, negative_max, margin, negative_classes=negative_classes,
    )
    without = runtime_pred_indices(
        probs, classes, thresholds, negative_max, 0.0, negative_classes=negative_classes,
    )
    return float(np.mean(with_margin != without))
