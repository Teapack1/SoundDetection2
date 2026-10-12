"""Frozen-test decision rules and the stored-score readers.

Deliberately free of torch, lightning and the dataset stack: the web UI reads
these on every Runs-tab page load, and pulling the training runtime in for what
is dict manipulation cost a multi-second stall on first paint.

`sed.train.eval` re-exports everything here, so `from sed.train.eval import
TestRule` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass

FROZEN_TEST_SECTION = "frozen_test"
DEPLOYED_TEST_SECTION = "deployed_test"


# --- Frozen-test decision rules ------------------------------------------
# A "rule" is how a model's per-class probabilities become a pass/fail on the
# frozen test. Every rule scores the SAME clips with the SAME checkpoint; only
# the decision step differs, so two rules are directly comparable statements
# about one model and one rule is directly comparable across models. The rule
# id is the storage key (manifest + reports), so it has to be stable.

RULE_KINDS: tuple[str, ...] = (
    "fixed",          # target class must reach a shared fixed confidence
    "argmax",         # highest-scoring class wins, no threshold at all
    "val_f1",         # per-class thresholds fitted for max F1 on the run's val
    "val_precision",  # per-class thresholds fitted to a val precision floor
    "deployed",       # the run's own runtime_settings.json decision layer
)

# Which kinds carry a numeric parameter, and its default.
RULE_PARAM_DEFAULTS: dict[str, float] = {"fixed": 0.90, "val_precision": 0.90}

# The ONE description of every rule: its label, what the decision actually does,
# and whether it puts all models on even ground. `benchmark=True` means the rule
# is identical for every model, so the resulting column compares architectures;
# `benchmark=False` means each model is scored at its own operating point, so
# the column compares deployments, not models. The browser renders the batch
# panel, the column headers, the run detail and the metric guide from this table
# (shipped as `rule_catalog()`), which is why no rule wording is written twice.
#
# `label_template` takes `{param}`, already formatted to two decimals.
# `on_exported` marks the rules that also make sense against an exported IR: the
# val-fitted ones fit thresholds on CHECKPOINT probabilities, so applying them to
# a quantized IR would silently compare two different decision layers.
RULE_KIND_INFO: dict[str, dict] = {
    "fixed": {
        "label_template": "≥ {param}",
        "benchmark": True,
        "on_exported": True,
        "param_label": "confidence",
        "how": (
            "a target class must reach this confidence to count as detected — "
            "the same rule for every model, per-run calibration ignored"
        ),
    },
    "argmax": {
        "label_template": "argmax",
        "benchmark": True,
        "on_exported": True,
        "param_label": "",
        "how": (
            "the highest-scoring class wins, no threshold at all — "
            "discriminative quality, immune to calibration"
        ),
    },
    "val_f1": {
        "label_template": "F1-opt (val)",
        "benchmark": False,
        "on_exported": False,
        "param_label": "",
        "how": (
            "per-class thresholds fitted for max F1 on THIS run's val split, then "
            "applied to the frozen test — every model at its own best operating point"
        ),
    },
    "val_precision": {
        "label_template": "P≥{param} (val)",
        "benchmark": False,
        "on_exported": False,
        "param_label": "val precision",
        "how": (
            "per-class thresholds fitted to reach this precision floor on val, then "
            "applied to the frozen test — every model at one shared operating point"
        ),
    },
    "deployed": {
        "label_template": "deployed thresholds",
        "benchmark": False,
        "on_exported": True,
        "param_label": "",
        "how": "this run's own saved decision layer (runtime_settings.json)",
    },
}


assert set(RULE_KIND_INFO) == set(RULE_KINDS), "RULE_KIND_INFO must describe every RULE_KINDS entry"


def rule_catalog() -> list[dict]:
    """Every decision rule, in presentation order, for UI and docs.

    The single source of rule wording. Served to the browser by
    `/api/train/runs` so the batch panel, the after-training panel, the runs
    table headers, the run detail and the metric guide cannot drift apart.
    """
    return [
        {
            "kind": kind,
            "label_template": info["label_template"],
            "label": TestRule(kind).label,
            "benchmark": info["benchmark"],
            "on_exported": info["on_exported"],
            "param": RULE_PARAM_DEFAULTS.get(kind),
            "param_label": info["param_label"],
            "how": info["how"],
        }
        for kind, info in RULE_KIND_INFO.items()
    ]


@dataclass(frozen=True)
class TestRule:
    """One frozen-test decision rule: identity, storage slug and UI label."""

    kind: str
    param: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"unknown test rule kind: {self.kind!r} (expected one of {RULE_KINDS})")
        if self.kind in RULE_PARAM_DEFAULTS:
            value = RULE_PARAM_DEFAULTS[self.kind] if self.param is None else float(self.param)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{self.kind} rule parameter must be in (0, 1), got {self.param!r}")
            object.__setattr__(self, "param", value)
        else:
            object.__setattr__(self, "param", None)

    @property
    def id(self) -> str:
        """Stable identity, e.g. 'fixed@0.90' — the manifest key."""
        return f"{self.kind}@{self.param:.2f}" if self.param is not None else self.kind

    @property
    def slug(self) -> str:
        """Filesystem-safe form of `id`, used for report filenames."""
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in self.id)

    @property
    def label(self) -> str:
        """Short human label, rendered from the one template in RULE_KIND_INFO."""
        template = RULE_KIND_INFO[self.kind]["label_template"]
        return template.format(param=f"{self.param:.2f}" if self.param is not None else "")

    @property
    def benchmark(self) -> bool:
        """True when this rule is identical for every model (even-ground comparison)."""
        return bool(RULE_KIND_INFO[self.kind]["benchmark"])

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "param": self.param,
            "label": self.label,
            "benchmark": self.benchmark,
        }


def parse_rule_id(rule_id: str) -> TestRule:
    """Inverse of `TestRule.id` — 'val_precision@0.90' → TestRule(...)."""
    raw = str(rule_id or "").strip()
    if not raw:
        raise ValueError("empty test rule id")
    kind, _, param = raw.partition("@")
    return TestRule(kind, float(param) if param else None)



def _is_non_target_row(name: str, row: dict) -> bool:
    """Whether one `per_class` row is background/suppressor, not a real target.

    Reads the `role` key `decision_metrics` annotates each row with; falls
    back to the literal `_negative` name for manifests stored before that key
    existed, so old runs keep scoring exactly as they did.
    """
    role = row.get("role")
    if role is not None:
        return role != "target"
    return name == "_negative"


def _non_target_rows(per_class: dict | None) -> list[dict]:
    if not isinstance(per_class, dict):
        return []
    return [
        row for name, row in per_class.items()
        if isinstance(row, dict) and _is_non_target_row(name, row)
    ]


def false_fire_rate(per_class: dict | None) -> float | None:
    """Fraction of non-target (background/suppressor) clips predicted as a target.

    A non-target class's false negatives ARE the false fires: a clip that
    trips no target threshold is predicted as its own (non-firing) class, so
    anything else it was predicted as is a target that fired on background.
    Sums over every non-target class — `_negative` by default, plus any class
    a project split its background pool into (see `_is_non_target_row`).
    `None` when the set has no non-target row at all (closed-set / target-only
    models).
    """
    rows = _non_target_rows(per_class)
    if not rows:
        return None
    support = sum(int(row.get("support") or 0) for row in rows)
    if not support:
        return None
    fires = sum(int(row["fn"]) for row in rows if row.get("fn") is not None)
    return float(fires) / float(support)


def _negative_support(per_class: dict | None) -> int | None:
    """Background-clip count — the denominator of every false-alarm rate."""
    rows = [row for row in _non_target_rows(per_class) if row.get("support") is not None]
    if not rows:
        return None
    return int(sum(row["support"] for row in rows))


def _negative_prior(per_class: dict | None, n_samples: object) -> float | None:
    """Share of the scored set that is background.

    Precision/F1/accuracy/AP are only comparable between blocks with the same
    prior, so it is recovered for older manifests rather than left unknown.
    """
    support = _negative_support(per_class)
    try:
        total = int(n_samples)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if support is None or total <= 0:
        return None
    return support / total


def false_per_1000_negatives(per_class: dict | None) -> float | None:
    """False fires per 1000 background clips.

    The same count as `false_fire_rate`, expressed as a rate. Precision, F1 and
    AP all shift when a test set's negative:positive balance changes, so they
    only compare within one test set. A per-negative rate does not — it is the
    clip-tier sibling of false-alarms-per-1000-onsets and of FP/hour, and it is
    what should be compared across differently-balanced frozen tests.
    """
    rate = false_fire_rate(per_class)
    return None if rate is None else rate * 1000.0


def macro_targets_from_per_class(per_class: dict | None, key: str) -> float | None:
    """Macro-average one per-class metric over the target classes.

    Recovers `precision_macro_targets` / `recall_macro_targets` for manifests
    written before those keys existed — every stored block carries `per_class`,
    so old runs can fill the P/R/prod-precision columns without re-evaluation.
    """
    if not isinstance(per_class, dict):
        return None
    values = [
        float(row[key])
        for name, row in per_class.items()
        if isinstance(row, dict)
        and not _is_non_target_row(name, row)
        and row.get(key) is not None
    ]
    return (sum(values) / len(values)) if values else None



def read_frozen_test_evals(manifest: dict) -> dict[str, dict]:
    """Every frozen-test score in a manifest, keyed by rule id.

    Bridges the pre-rule manifest layout so old runs show up in the rule columns
    without being re-evaluated:

      * ``benchmark_metrics``            → ``fixed@<its threshold>``
      * ``benchmark_calibrated_metrics`` → ``deployed``
      * ``test_metrics``                 → ``deployed`` (or ``fixed@…``)

    Legacy blocks are tagged ``origin`` so callers can flag them: the train-time
    ``test_metrics`` is scored against the run's OWN dataset snapshot, not the
    current frozen test, and is therefore not comparable across runs.

    Every returned block is completed from ``per_class`` where the stored one
    predates a metric, so old runs fill the macro-precision / macro-recall /
    false-fire columns without being re-scored.
    """
    evals: dict[str, dict] = {}

    def _complete(block: dict) -> dict:
        per_class = block.get("per_class")
        out = dict(block)
        if out.get("precision_macro_targets") is None:
            out["precision_macro_targets"] = macro_targets_from_per_class(per_class, "precision")
        if out.get("recall_macro_targets") is None:
            out["recall_macro_targets"] = macro_targets_from_per_class(per_class, "recall")
        if out.get("false_fire_rate") is None:
            out["false_fire_rate"] = false_fire_rate(per_class)
        if out.get("false_per_1000_negatives") is None:
            out["false_per_1000_negatives"] = false_per_1000_negatives(per_class)
        if out.get("n_negatives") is None:
            out["n_negatives"] = _negative_support(per_class)
        if out.get("negative_prior") is None:
            out["negative_prior"] = _negative_prior(per_class, out.get("n_samples"))
        return out

    def _legacy(block: object, origin: str, report: str, force_kind: str | None = None) -> None:
        if not isinstance(block, dict) or block.get("f1_macro_targets") is None:
            return
        if force_kind == "deployed" or block.get("runtime_settings_applied"):
            rule = TestRule("deployed")
        elif block.get("threshold") is not None:
            try:
                rule = TestRule("fixed", float(block["threshold"]))
            except (TypeError, ValueError):
                return
        else:
            return
        if rule.id in evals:
            return  # a real frozen_test entry always wins over a reconstructed one
        # `report_name` locates this block's report files, which for legacy
        # blocks are NOT named after the rule slug.
        evals[rule.id] = {
            **_complete(block), "rule": rule.as_dict(), "origin": origin, "report_name": report,
        }

    stored = manifest.get(FROZEN_TEST_SECTION)
    if isinstance(stored, dict):
        for rule_id, block in stored.items():
            if not isinstance(block, dict):
                continue
            try:
                rule = parse_rule_id(rule_id)
            except ValueError:
                continue
            evals[rule.id] = {
                **_complete(block), "rule": rule.as_dict(), "origin": "rule", "report_name": rule.slug,
            }

    _legacy(manifest.get("benchmark_metrics"), "legacy_benchmark", "benchmark")
    _legacy(
        manifest.get("benchmark_calibrated_metrics"), "legacy_benchmark",
        "benchmark_calibrated", force_kind="deployed",
    )
    # test_source "model" means the run's own snapshot (the train-time auto-test)
    # rather than the current frozen test — a real cross-run comparability trap.
    tm = manifest.get("test_metrics")
    _legacy(
        tm,
        "train_auto" if (isinstance(tm, dict) and tm.get("test_source") == "model") else "legacy_test",
        "test",
    )
    return evals


def read_deployed_test_evals(manifest: dict) -> dict[str, dict]:
    """Every exported-IR score in a manifest, keyed ``"<precision>:<rule id>"``.

    The flat key mirrors the frozen-test keying, so both tiers of the runs table
    discover their column bands the same way. A pre-section manifest carries one
    ``test_metrics_exported`` block; it is bridged in under the precision its
    ``model_file`` names, so old runs keep their deployed columns.
    """
    evals: dict[str, dict] = {}

    def _complete(block: dict, precision: str, rule: TestRule, origin: str) -> dict:
        per_class = block.get("per_class")
        out = dict(block)
        for key, metric in (("precision_macro_targets", "precision"), ("recall_macro_targets", "recall")):
            if out.get(key) is None:
                out[key] = macro_targets_from_per_class(per_class, metric)
        if out.get("false_fire_rate") is None:
            out["false_fire_rate"] = false_fire_rate(per_class)
        if out.get("false_per_1000_negatives") is None:
            out["false_per_1000_negatives"] = false_per_1000_negatives(per_class)
        if out.get("n_negatives") is None:
            out["n_negatives"] = _negative_support(per_class)
        if out.get("negative_prior") is None:
            out["negative_prior"] = _negative_prior(per_class, out.get("n_samples"))
        out["ir_precision"] = precision
        out["rule"] = rule.as_dict()
        out["origin"] = origin
        return out

    section = manifest.get(DEPLOYED_TEST_SECTION)
    if isinstance(section, dict):
        for precision, by_rule in section.items():
            if not isinstance(by_rule, dict):
                continue
            for rule_id, block in by_rule.items():
                if not isinstance(block, dict):
                    continue
                try:
                    rule = parse_rule_id(rule_id)
                except ValueError:
                    continue
                evals[f"{precision}:{rule.id}"] = _complete(block, str(precision), rule, "rule")

    legacy = manifest.get("test_metrics_exported")
    if isinstance(legacy, dict) and legacy.get("f1_macro_targets") is not None:
        precision = str(legacy.get("ir_precision") or "")
        if not precision:
            name = str(legacy.get("model_file") or "").lower()
            precision = "int8" if "int8" in name else ("fp16" if "fp16" in name else "ir")
        rule = TestRule("deployed")
        key = f"{precision}:{rule.id}"
        if key not in evals:  # a real section entry always wins
            evals[key] = _complete(legacy, precision, rule, "legacy_exported")
    return evals




# Rules that can be applied to an exported IR. The val-fitted rules are excluded
# on purpose: they fit thresholds on CHECKPOINT probabilities, so scoring an IR
# with them would silently compare two different decision layers.
EXPORTED_RULE_KINDS = tuple(k for k, v in RULE_KIND_INFO.items() if v["on_exported"])
