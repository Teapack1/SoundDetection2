"""Group training runs that differ only by random seed.

Architectures on this project were being ranked on frozen-test gaps of 1-3
points from **single runs**. A run is one draw from a distribution: without
replicates there is nothing to compare a gap against, so those rankings were
not measurable differences. (All 60 runs on disk use seed 42 — the replicate
count is currently 1 everywhere, which this makes visible rather than implied.)

`experiment_key` fingerprints everything that defines a run except the seed, so
runs sharing a key are replicates and can be reported as mean ± spread with an
explicit n. Torch-free: the inspect server reads this on page load.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# Config blocks that shape the experiment: data, features, model, schedule, and
# the decision knobs that drive checkpoint selection (`lit_module` scores its
# selection metric through the runtime gates, so `inference` belongs here).
EXPERIMENT_BLOCKS = (
    "audio", "augmentation", "dataset", "features", "inference", "model", "onset", "training",
)

# Deployment, transport and UI state — changing any of them re-runs nothing.
# `targets`/`thresholds` are the post-training decision layer, recalibrated
# independently of the weights.
_IGNORED_BLOCKS = frozenset({
    "artnet", "deploy", "inspect", "logging", "mining", "project", "targets", "thresholds", "zmq",
})

# Per-key exclusions inside kept blocks: the seed itself, plus provenance that
# records *how* a run resolved rather than what was asked for. Feature
# scaling statistics (the fitted values and their provenance) are fitted from
# the seeded split, so replicates would otherwise disagree on them and never
# group; the *method/scope* stay in the key — they are the asked-for design.
_IGNORED_KEYS = {
    "training": frozenset({"seed"}),
    "features": frozenset({
        "standardize_mean", "standardize_std", "scaling_stats",
        "standardize_stats_resolved", "standardize_stats_dataset_sha256",
        "standardize_stats_split_seed", "standardize_stats_val_fraction",
        "standardize_stats_clips",
    }),
    "model": frozenset({"classes"}),  # derived from dataset.target_classes
}


def _canonical(config: dict | None) -> dict:
    out: dict[str, Any] = {}
    for block, value in sorted((config or {}).items()):
        if block in _IGNORED_BLOCKS or not isinstance(value, dict):
            continue
        if block not in EXPERIMENT_BLOCKS:
            continue
        drop = _IGNORED_KEYS.get(block, frozenset())
        out[block] = {k: v for k, v in sorted(value.items()) if k not in drop}
    return out


def experiment_key(config: dict | None) -> str:
    """Stable hash of a run's configuration with the seed removed.

    Two runs with the same key are replicates of one experiment. Deliberately
    conservative: anything that might have changed the outcome is included, so
    the failure mode is reporting two replicates separately rather than
    averaging two runs that were never comparable.
    """
    blob = json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def run_seed(config: dict | None) -> int | None:
    seed = ((config or {}).get("training") or {}).get("seed")
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def summarize(values: list[float | None]) -> dict:
    """mean / sample sd / min / max / n over the values that are present.

    `sd` is the sample standard deviation (n-1) and is None for a single run —
    a lone value has no spread, and reporting 0.0 would read as "reproducible".
    """
    present = [float(v) for v in values if v is not None and not _isnan(v)]
    n = len(present)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    mean = sum(present) / n
    sd = None
    if n > 1:
        var = sum((v - mean) ** 2 for v in present) / (n - 1)
        sd = math.sqrt(var)
    return {"n": n, "mean": mean, "sd": sd, "min": min(present), "max": max(present)}


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def group_runs(runs: list[dict]) -> list[dict]:
    """Collapse runs into seed groups, preserving input order of first sighting.

    Each run needs ``config`` (its frozen training config) and ``id``. The
    result carries the member ids and seeds so a group can be expanded, plus
    ``replicated`` — false when n == 1, which is the state to make loud.
    """
    groups: dict[str, dict] = {}
    for run in runs:
        key = experiment_key(run.get("config"))
        group = groups.setdefault(key, {
            "experiment_key": key, "run_ids": [], "seeds": [], "runs": [],
        })
        group["run_ids"].append(run.get("id"))
        group["seeds"].append(run_seed(run.get("config")))
        group["runs"].append(run)
    for group in groups.values():
        distinct = {s for s in group["seeds"] if s is not None}
        group["n"] = len(group["run_ids"])
        # Two runs at the SAME seed are re-runs, not independent samples.
        group["n_distinct_seeds"] = len(distinct)
        group["replicated"] = len(distinct) > 1
    return list(groups.values())
