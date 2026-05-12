"""Compose a project dataset.

Reads `projects/<id>/project.toml` to figure out which classes the model needs,
then assembles three CSVs (train.csv, val.csv, test.csv) listing the source
audio files plus their class labels and a session_id (used for leakage-free
splitting).

Sources, in order of precedence:
  1. On-site recordings under `projects/<id>/recordings/positives/<class>/`
     and `.../hardneg/<class>/` — registered automatically.
  2. Library clips, optionally pulled in based on `[dataset]` rules in
     project.toml (e.g. "use all library clips with label='wine_thin'").
  3. Ambience: union of project ambience + library ambience.

The composer DOES NOT itself perform augmentation or domain-mixing — that is
done at training time by the DataModule, so we can vary it across epochs
without re-writing files.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from sed.config import Config, ProjectLayout
from sed.data.library import Library
from sed.data.splits import session_split, session_split_adaptive

log = logging.getLogger(__name__)


@dataclass
class SourceClip:
    path: str
    class_name: str
    session_id: str
    sample_rate: int
    duration_s: float
    is_target: bool                # whether this class is a positive target
    origin: str                    # "project" | "library"


def _read_sidecar(wav_path: Path) -> dict | None:
    side = wav_path.with_suffix(".json")
    if not side.exists():
        return None
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _scan_project_class(
    class_dir: Path, class_name: str, is_target: bool
) -> list[SourceClip]:
    out: list[SourceClip] = []
    if not class_dir.exists():
        return out
    for wav in sorted(class_dir.rglob("*.wav")):
        meta = _read_sidecar(wav)
        if meta and meta.get("flagged"):
            continue                                # quarantined by inspector
        try:
            info = sf.info(str(wav))
        except Exception as e:  # noqa: BLE001
            log.warning("skipping unreadable %s: %s", wav, e)
            continue
        session_id = meta.get("session_id", wav.stem) if meta else wav.stem
        out.append(
            SourceClip(
                path=str(wav.resolve()),
                class_name=class_name,
                session_id=session_id,
                sample_rate=info.samplerate,
                duration_s=info.frames / info.samplerate if info.samplerate else 0.0,
                is_target=is_target,
                origin="project",
            )
        )
    return out


def _scan_library_for_class(
    library: Library, class_name: str, mapping: dict
) -> list[SourceClip]:
    """`mapping` is e.g. {"category": "glasses_clean", "label": "wine_thin"}.

    If `label` is None or absent, all labels under the category are pulled.
    Each library clip gets a UNIQUE session_id (derived from its UUID) so
    the splitter can distribute them freely — library clips never have the
    session-leakage problem that on-site multi-take recordings do.
    """
    out: list[SourceClip] = []
    clips = library.query(
        category=mapping.get("category"),
        label=mapping.get("label"),                     # None = all labels
        tags_all=mapping.get("tags_all"),
    )
    for c in clips:
        wav = c.absolute_path()
        if not wav.exists():
            continue
        # Treat each library clip as its own session for splitting purposes.
        # Reasoning: library clips were recorded as independent samples
        # (often deliberately, varying force/angle/glass), so there's no
        # session-coupling between two clips of the same label. This lets
        # the splitter distribute them randomly across train/val/test.
        out.append(
            SourceClip(
                path=str(wav.resolve()),
                class_name=class_name,
                session_id=f"lib-{c.uuid[:12]}",
                sample_rate=c.sample_rate,
                duration_s=c.duration_s,
                is_target=class_name != "_negative",
                origin="library",
            )
        )
    return out


def _strip_common_prefixes(name: str) -> str:
    """Try common naming conventions to map a class name to a library label.

    Examples:
        clink_wine_thin     → wine_thin
        glass_champagne     → champagne
        wine_thin           → wine_thin   (no change)
    """
    for prefix in ("clink_", "glass_", "snd_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _auto_discover_library(
    library: Library,
    classes: list[str],
    explicit_rules: dict,
) -> dict[str, list[dict]]:
    """For each class in the model, figure out which library labels to pull.

    Rules (lowest priority first):
      1. Auto-discover by name matching the class to category/label in library.
      2. project.toml [dataset.library] block overrides (highest priority).

    Returns a dict {class_name: [list of {category, label} dicts]} —
    plus a meta dict per class describing what auto-matched.
    """
    rules: dict[str, list[dict]] = {}

    # ---- Auto-discover defaults ----
    for class_name in classes:
        if class_name == "_negative":
            # All hard-negatives in the library are fair game for _negative.
            # Library ambience is NOT included here — it's used for domain
            # mixing augmentation, not as direct training negative samples.
            categories_to_scan = ["negatives_external"]
            class_rules: list[dict] = []
            for cat in categories_to_scan:
                # Wildcard: any label. We pass label=None to pick all.
                if library.query(category=cat):
                    class_rules.append({"category": cat})
            if class_rules:
                rules[class_name] = class_rules
            continue

        # Target classes: try to match to `glasses_clean` library labels
        candidates = [class_name, _strip_common_prefixes(class_name)]
        # De-dupe but preserve order
        seen = set(); ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c); ordered.append(c)

        matched: list[dict] = []
        for label_candidate in ordered:
            hits = library.query(category="glasses_clean", label=label_candidate)
            if hits:
                matched.append({"category": "glasses_clean", "label": label_candidate})
                break                        # first match wins; don't double-include
        if matched:
            rules[class_name] = matched

    # ---- Apply explicit overrides from [dataset.library] ----
    for class_name, mapping in explicit_rules.items():
        normalized = mapping if isinstance(mapping, list) else [mapping]
        rules[class_name] = normalized           # explicit completely replaces auto

    return rules


def compose_dataset(cfg: Config, project_id: str, split_seed: int = 42) -> dict:
    """Build train.csv / val.csv / test.csv for a project.

    The composer assembles clips from THREE sources, in order:

      A. On-site project recordings under projects/<id>/recordings/
           positives/<class>/ → that class
           hardneg/<label>/   → _negative
           ambience/          → _negative
      B. Library content, AUTO-DISCOVERED by class-name matching:
           target class 'clink_wine_thin' → glasses_clean/wine_thin/*
           target class 'champagne'       → glasses_clean/champagne/*
           '_negative'                    → negatives_external/* (all labels)
      C. Library content explicitly listed in project.toml [dataset.library]
         (overrides the auto-discovery — set this when you want non-default
         mappings, e.g. pulling only specific labels).

    Library AMBIENCE is intentionally NOT included as CSV rows. It's used at
    training time for domain mixing instead (overlaid on positives), which
    is a different thing. Project ambience IS included as _negative rows.

    Returns a summary dict for printing.
    """
    layout = ProjectLayout(project_id)
    library = Library()
    classes = list(cfg.model.classes)
    if "_negative" not in classes:
        raise ValueError("model.classes must include '_negative'")

    # ---- A. Discover project on-site clips ----
    by_class_project: dict[str, list[SourceClip]] = {c: [] for c in classes}
    for class_name in classes:
        if class_name == "_negative":
            for hn_dir in (layout.recordings_dir / "hardneg").glob("*/"):
                by_class_project["_negative"].extend(
                    _scan_project_class(hn_dir, "_negative", is_target=False)
                )
            if layout.ambience_dir.exists():
                by_class_project["_negative"].extend(
                    _scan_project_class(layout.ambience_dir, "_negative", is_target=False)
                )
        else:
            pos_dir = layout.positives_dir(class_name)
            by_class_project[class_name].extend(
                _scan_project_class(pos_dir, class_name, is_target=True)
            )

    # ---- B + C. Library content (auto-discovered, with explicit overrides) ----
    proj_toml_path = layout.project_toml
    explicit_rules: dict = {}
    if proj_toml_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(proj_toml_path, "rb") as f:
            explicit_rules = tomllib.load(f).get("dataset", {}).get("library", {})

    library_rules = _auto_discover_library(library, classes, explicit_rules)
    by_class_library: dict[str, list[SourceClip]] = {c: [] for c in classes}
    for class_name, rules in library_rules.items():
        for r in rules:
            by_class_library[class_name].extend(
                _scan_library_for_class(library, class_name, r)
            )

    # ---- Combine and report ----
    all_clips: list[SourceClip] = []
    counts_by_origin: dict[str, dict[str, int]] = {}
    for class_name in classes:
        proj = by_class_project[class_name]
        lib = by_class_library[class_name]
        all_clips.extend(proj)
        all_clips.extend(lib)
        counts_by_origin[class_name] = {
            "project": len(proj), "library": len(lib), "total": len(proj) + len(lib),
        }

    log.info("--- composed dataset summary ---")
    for class_name, c in counts_by_origin.items():
        log.info(
            "  %-22s  project=%3d  library=%3d  total=%4d",
            class_name, c["project"], c["library"], c["total"],
        )
    if counts_by_origin.get("_negative", {}).get("total", 0) == 0:
        log.warning(
            "_negative is empty. Domain mixing during training will still "
            "provide negative-like samples via ambience augmentation, but "
            "the model benefits from explicit negatives — record hardnegs or "
            "import some into the library with `clinkd library record-hardneg`."
        )
    for class_name in classes:
        if class_name == "_negative":
            continue
        if counts_by_origin[class_name]["total"] == 0:
            log.warning(
                "class '%s' has ZERO clips. Either record some "
                "(`clinkd record positives --class %s`) or add a "
                "[dataset.library] mapping in project.toml.",
                class_name, class_name,
            )

    # ---- Split (adaptive: per-class session-aware OR random fallback) ----
    train, val, test, split_modes = session_split_adaptive(
        all_clips, val_frac=0.15, test_frac=0.15, seed=split_seed,
    )
    log.info("--- split modes per class ---")
    for class_name, mode in split_modes.items():
        log.info("  %-22s  %s", class_name, mode)

    # ---- Write CSVs ----
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", train), ("val", val), ("test", test)]:
        out_csv = layout.dataset_dir / f"{name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["path", "class_name", "session_id", "sample_rate", "duration_s",
                 "is_target", "origin"]
            )
            for c in items:
                w.writerow([c.path, c.class_name, c.session_id, c.sample_rate,
                            f"{c.duration_s:.4f}", int(c.is_target), c.origin])

    class_index = {name: i for i, name in enumerate(classes)}
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_index": class_index}, f, indent=2)

    return {
        "total_clips": len(all_clips),
        "by_class": {c: counts_by_origin[c]["total"] for c in classes},
        "by_origin": counts_by_origin,
        "split_modes": split_modes,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }
