"""Local HTTP server for the interactive inspect UI.

`sed dataset inspect --serve` opens this on http://127.0.0.1:PORT. The
HTML report is rendered live and each clip carries a checkbox. Pressing
"Done" writes the ticked dataset/augmented clips to the project's soft
exclusion manifest for the next compose/train run. The "Delete selected"
button POSTs the chosen paths to /api/delete, which removes the WAV + sidecar
from disk. The "Promote selected" button on the mined tab POSTs to
/api/promote-mined, which turns mined clips into accepted library sounds.

No external deps — uses stdlib http.server. Single-process, localhost-only;
no auth needed because the server only binds to 127.0.0.1.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import time
import threading
import uuid
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sed.config import Config, ProjectLayout

log = logging.getLogger(__name__)


_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_AUDIO_SUFFIXES: frozenset[str] = frozenset({".wav"})


def _safe_resolve_asset_path(
    layout: ProjectLayout, raw_path: str, allowed_suffixes: frozenset[str],
) -> Path | None:
    """Resolve a path requested via /audio?path=... or /image?path=... and
    only allow files inside the library dir, the project dataset report dir,
    or the legacy project's recordings dir. The suffix must be in
    `allowed_suffixes`. This keeps the server from being abused as a
    file-read primitive."""
    from sed.config import LIBRARY_DIR
    try:
        candidate = Path(raw_path).resolve()
    except Exception:  # noqa: BLE001
        return None
    allowed_roots = [
        (layout.dataset_inspect_dir / "augmented").resolve(),
        LIBRARY_DIR.resolve(),
        layout.recordings_dir.resolve(),
    ]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
                return candidate
        except ValueError:
            continue
    return None


def _safe_resolve_audio_path(layout: ProjectLayout, raw_path: str) -> Path | None:
    return _safe_resolve_asset_path(layout, raw_path, _AUDIO_SUFFIXES)


def _safe_label(label: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label.strip())
    return cleaned or "mined_false_positive"


def _promote_mined_clip(
    layout: ProjectLayout,
    cfg: Config | None,
    wav: Path,
    label: str,
) -> Path | None:
    from sed.data.library import Library

    lib = Library()
    dest = lib.promote_mined(wav, label=_safe_label(label), project_id=layout.id)
    if dest is None:
        return None
    side = dest.with_suffix(".json")
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        meta = {}
    meta["promoted_to"] = "_negative"
    meta["promoted_negative_label"] = _safe_label(label)
    meta["promoted_at"] = datetime.now(UTC).isoformat()
    meta["promoted_from"] = str(wav)
    side.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lib.rebuild_manifest()
    return dest


def _cleanup_after_file_changes(layout: ProjectLayout) -> None:
    try:
        from sed.data.compose import prune_project_metadata
        prune_project_metadata(layout)
    except Exception as exc:  # noqa: BLE001
        log.warning("metadata cleanup failed: %s", exc)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dedupe_strings(raw: object) -> list[str]:
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    elif isinstance(raw, list):
        parts = [str(item) for item in raw]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        val = str(part).strip().lstrip("#")
        if not val or val in seen:
            continue
        out.append(val)
        seen.add(val)
    return out


def _selection_rule_id() -> str:
    return uuid.uuid4().hex[:12]


def _normalise_filter_rule(raw: object, *, default_split: str = "train") -> dict:
    if not isinstance(raw, dict):
        raw = {}
    row: dict = {}
    labels = _dedupe_strings(raw.get("labels", raw.get("label")))
    if labels:
        row["labels"] = labels
    project_ids = raw.get("project_ids", raw.get("project_id"))
    projects = _dedupe_strings(project_ids)
    if project_ids == "*":
        row["project_ids"] = "*"
    elif projects:
        row["project_ids"] = projects
    for key in (
        "density", "dataset_role", "source_bucket", "session", "search",
        "clip_kind", "mic_filter",
    ):
        value = str(raw.get(key) or "").strip()
        if value:
            row[key] = value
    try:
        from sed.data.library import coerce_channel_list
        for key in ("triggered_channels", "recorded_channels", "mic_channels"):
            channels = coerce_channel_list(raw.get(key))
            if channels:
                row[key] = channels
    except Exception:  # noqa: BLE001
        pass
    if raw.get("channel_index") not in (None, ""):
        try:
            channel_index = int(raw.get("channel_index"))
            if channel_index >= 0:
                row["channel_index"] = channel_index
        except Exception:  # noqa: BLE001
            pass
    tags = _dedupe_strings(raw.get("tags_all", raw.get("tags")))
    if tags:
        row["tags_all"] = tags
    split = str(raw.get("split") or default_split or "train").strip().lower()
    if split not in {"train", "test"}:
        split = "train"
    row["split"] = split
    if split == "test":
        row["dataset_role"] = "test"
    if "count" in raw:
        try:
            row["count"] = int(raw.get("count") or 0)
        except Exception:  # noqa: BLE001
            pass
    row["id"] = str(raw.get("id") or _selection_rule_id())
    row["updated_at"] = _now_iso()
    return row


def _single_label_from_rule(rule: dict) -> str:
    labels = rule.get("labels")
    if isinstance(labels, list) and len(labels) == 1 and labels[0] != "*":
        return str(labels[0])
    label = rule.get("label")
    if isinstance(label, str) and label and label != "*":
        return label
    return "*"


def _normalise_filter_rules(raw: object, *, default_split: str = "train") -> list[dict]:
    rows = raw if isinstance(raw, list) else [raw]
    return [
        _normalise_filter_rule(row, default_split=default_split)
        for row in rows
        if isinstance(row, dict)
    ]


def _normalise_augmentation_recipe(raw: object) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    base_filters = _normalise_filter_rules(
        raw.get("base_filters", raw.get("base_filter") or {}),
        default_split="train",
    ) or [_normalise_filter_rule({}, default_split="train")]
    overlay_raw = raw.get("overlay_filters", raw.get("overlay_filter") or {})
    overlay_filters = _normalise_filter_rules(
        overlay_raw,
        default_split="train",
    )
    if not overlay_filters and not (isinstance(overlay_raw, list) and len(overlay_raw) == 0):
        overlay_filters = [_normalise_filter_rule({}, default_split="train")]
    recipe_id = str(raw.get("recipe_id") or raw.get("id") or _selection_rule_id())
    overlay_filter = overlay_filters[0] if overlay_filters else {}
    overlay_label = (
        str(raw.get("overlay"))
        if raw.get("overlay")
        else (_single_label_from_rule(overlay_filter) if overlay_filters else "")
    )
    # Recipes built from frozen-test buckets default to frozen-test output so
    # augmented test material lands in the test split without an extra click.
    frozen_raw = raw.get("frozen_test")
    if frozen_raw is None:
        frozen_test = bool(base_filters) and all(
            str(rule.get("dataset_role") or rule.get("split") or "").strip().lower() == "test"
            for rule in base_filters
        )
    else:
        frozen_test = _bool_value(frozen_raw)
    recipe = {
        "id": recipe_id,
        "recipe_id": recipe_id,
        "base": str(raw.get("base") or _single_label_from_rule(base_filters[0]) or "*"),
        "overlay": overlay_label,
        "base_filter": base_filters[0],
        "overlay_filter": overlay_filter,
        "base_filters": base_filters,
        "overlay_filters": overlay_filters,
        "description": str(raw.get("description") or "").strip(),
        "tags": _dedupe_strings(raw.get("tags")),
        "base_count": int(raw.get("base_count") or sum(int(r.get("count") or 0) for r in base_filters)),
        "overlay_count": int(raw.get("overlay_count") or sum(int(r.get("count") or 0) for r in overlay_filters)),
        # Per-recipe generation knobs that used to live at the page level.
        # `selected` defaults to True so existing recipes participate in
        # "Generate selected" until the user untoggles them.
        "prefix": str(raw.get("prefix") or "").strip(),
        "replace_existing": _bool_value(raw.get("replace_existing", False)),
        "frozen_test": frozen_test,
        "selected": _bool_value(raw.get("selected", True)),
        # Snapshot of the augmentation panel state at recipe-creation time.
        # When the recipe runs, these override the project defaults so two
        # recipes can use different pitch / SNR / mask thresholds. Empty dict
        # means "use whatever the project cfg has" (legacy recipes).
        "settings": _normalise_augmentation_settings(raw.get("settings") or {}, None)
                    if isinstance(raw.get("settings"), dict) else {},
        "updated_at": _now_iso(),
    }
    return recipe


def _labels_summary_from_filters(filters: list[dict]) -> str:
    labels: list[str] = []
    for rule in filters:
        raw = rule.get("labels", rule.get("label"))
        if isinstance(raw, str):
            labels.append(raw)
        elif isinstance(raw, list):
            labels.extend(str(item) for item in raw if str(item))
    labels = [label for label in labels if label and label != "*"]
    if not labels:
        return "*"
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return ",".join(out)


def _role_for_filter(rule: dict, default_role: str = "train") -> str:
    role = str(rule.get("dataset_role") or rule.get("split") or default_role).strip().lower()
    return role if role in {"train", "test", "ignore"} else default_role


def _count_library_filter(
    lib,
    rule: dict,
    *,
    project_id: str,
    default_role: str = "train",
    frozen_test: bool = False,
) -> int:
    from sed.data.augment import _library_paths_for_filters

    role = _role_for_filter(rule, default_role)
    return len(_library_paths_for_filters(
        lib,
        _labels_summary_from_filters([rule]),
        dataset_role=role,
        project_id=project_id,
        frozen_test=frozen_test or role == "test",
        filters=[rule],
    ))


def _refresh_filter_rule_counts(
    lib,
    filters: list[dict],
    *,
    project_id: str,
    default_role: str = "train",
    frozen_test: bool = False,
) -> bool:
    changed = False
    for rule in filters:
        if not isinstance(rule, dict):
            continue
        count = _count_library_filter(
            lib,
            rule,
            project_id=project_id,
            default_role=default_role,
            frozen_test=frozen_test,
        )
        try:
            old_count = int(rule.get("count"))
        except Exception:  # noqa: BLE001
            old_count = None
        if old_count != count:
            rule["count"] = count
            changed = True
    return changed


def _refresh_recipe_counts(
    recipe: dict,
    *,
    selection: dict,
    cfg: Config | None,
    project_id: str,
    lib=None,
) -> dict:
    settings = _normalise_augmentation_settings(
        recipe.get("settings") if isinstance(recipe.get("settings"), dict)
        else selection.get("augmentation_settings", {}),
        cfg,
    )
    base_filters = recipe.get("base_filters") if isinstance(recipe.get("base_filters"), list) else []
    if not base_filters and isinstance(recipe.get("base_filter"), dict):
        base_filters = [recipe["base_filter"]]
        recipe["base_filters"] = base_filters
    overlay_filters = recipe.get("overlay_filters") if isinstance(recipe.get("overlay_filters"), list) else []
    if not overlay_filters and isinstance(recipe.get("overlay_filter"), dict) and recipe.get("overlay_filter"):
        overlay_filters = [recipe["overlay_filter"]]
        recipe["overlay_filters"] = overlay_filters
    # The recipe's overlay bucket is authoritative: a selected bucket means the
    # overlay stage runs, regardless of the panel toggle captured in settings.
    # Only a recipe without overlay filters can be pitch-only.
    overlay_enabled = bool(overlay_filters) or bool(settings.get("overlay_enabled", True))
    if overlay_enabled and not overlay_filters:
        raise ValueError("Overlay is enabled, but no overlay bucket is selected.")
    settings["overlay_enabled"] = overlay_enabled

    from sed.data.augment import _library_paths_for_filters
    if lib is None:
        from sed.data.library import Library
        lib = Library()
    frozen = bool(recipe.get("frozen_test", False))
    role = "test" if frozen else "train"
    _refresh_filter_rule_counts(
        lib,
        base_filters,
        project_id=project_id,
        default_role=role,
        frozen_test=frozen,
    )
    if overlay_enabled:
        _refresh_filter_rule_counts(
            lib,
            overlay_filters,
            project_id=project_id,
            default_role=role,
            frozen_test=frozen,
        )
    recipe["base"] = _labels_summary_from_filters(base_filters)
    recipe["overlay"] = _labels_summary_from_filters(overlay_filters) if overlay_enabled else "pitch"
    recipe["base_count"] = len(_library_paths_for_filters(
        lib,
        recipe["base"],
        dataset_role=role,
        project_id=project_id,
        frozen_test=frozen,
        filters=base_filters,
    ))
    recipe["overlay_count"] = (
        len(_library_paths_for_filters(
            lib,
            recipe["overlay"],
            dataset_role=role,
            project_id=project_id,
            frozen_test=frozen,
            filters=overlay_filters,
        ))
        if overlay_enabled
        else 0
    )
    recipe["settings"] = settings
    return recipe


def _overlay_label_from_sidecar(wav_path: Path) -> str:
    """Read just the overlay class from a derived clip's sidecar (cheap)."""
    sidecar = wav_path.with_suffix(".json")
    if not sidecar.exists():
        return ""
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    params = meta.get("augmentation_params") or {}
    return str(
        meta.get("overlay_label")
        or params.get("actual_overlay_label")
        or params.get("overlay_label")
        or ""
    )


def _overlay_labels_for_rule(lib, rule: dict, *, project_id: str) -> list[str]:
    """Distinct overlay classes among the derived clips a source rule matches."""
    from sed.data.augment import _library_paths_for_filters
    role = _role_for_filter(rule)
    try:
        paths = _library_paths_for_filters(
            lib,
            _labels_summary_from_filters([rule]),
            dataset_role=role,
            project_id=project_id,
            frozen_test=role == "test",
            filters=[rule],
        )
    except Exception:  # noqa: BLE001
        return []
    seen: set[str] = set()
    out: list[str] = []
    for path, _label in paths:
        overlay = _overlay_label_from_sidecar(path)
        if overlay and overlay not in seen:
            seen.add(overlay)
            out.append(overlay)
    return sorted(out)


def _attach_overlay_labels(selection: dict, *, project_id: str) -> None:
    """Annotate derived (augmented) source rules with their distinct overlay
    classes so the Inspect UI can show + search them. Display-only; not saved."""
    assignments = selection.get("assignments")
    if not isinstance(assignments, dict):
        return
    from sed.data.library import Library
    lib = Library()
    for rows in assignments.values():
        rule_list = [rows] if isinstance(rows, dict) else rows
        if not isinstance(rule_list, list):
            continue
        for rule in rule_list:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("source_bucket") or "").lower() == "derived":
                rule["overlay_labels"] = _overlay_labels_for_rule(
                    lib, rule, project_id=project_id
                )
            else:
                rule.pop("overlay_labels", None)


def _refresh_selection_counts(selection: dict, *, cfg: Config | None, project_id: str) -> bool:
    """Update display counts on UI-managed source rules and recipes from disk."""
    from sed.data.library import Library

    changed = False
    lib = Library()
    assignments = selection.get("assignments", {})
    if isinstance(assignments, dict):
        for rows in assignments.values():
            if isinstance(rows, dict):
                rows = [rows]
            if isinstance(rows, list):
                changed |= _refresh_filter_rule_counts(
                    lib,
                    [row for row in rows if isinstance(row, dict)],
                    project_id=project_id,
                    default_role="train",
                )

    recipes = selection.get("augmentation_recipes", [])
    if isinstance(recipes, list):
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
            before = json.dumps(recipe, sort_keys=True, default=str)
            try:
                _refresh_recipe_counts(
                    recipe,
                    selection=selection,
                    cfg=cfg,
                    project_id=project_id,
                    lib=lib,
                )
            except ValueError:
                continue
            if json.dumps(recipe, sort_keys=True, default=str) != before:
                changed = True
    return changed


_AUG_BOOL_FIELDS = {
    "enabled",
    "pitch_shift_enabled",
    "overlay_enabled",
    "scale_noise_to_snr",
    "prevent_clipping",
    "mask_filter_enabled",
}
_AUG_INT_FIELDS = {
    "seed",
    "transient_window_ms",
    "mask_candidate_attempts",
    "mask_band_count",
    "mask_top_frames",
}
_AUG_FLOAT_FIELDS = {
    "samples_per_source",
}
_AUG_OPTIONAL_FLOAT_FIELDS = {
    "mask_min_mean_db",
    "mask_min_p25_db",
    "mask_min_band_db",
    "mask_min_effective_snr_db",
}
_AUG_RANGE_FIELDS = {"semitones_range", "snr_db_range"}


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _range_value(value: object) -> list[float] | None:
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) != 2:
        return None
    try:
        return [float(parts[0]), float(parts[1])]
    except Exception:  # noqa: BLE001
        return None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _augmentation_settings_from_cfg(cfg: Config | None) -> dict:
    if cfg is None:
        return {}
    aug = cfg.augmentation
    return {
        "enabled": bool(aug.enabled),
        "seed": int(aug.seed),
        "samples_per_source": float(aug.samples_per_source),
        "pitch_shift_enabled": bool(aug.pitch_shift_enabled),
        "semitones_range": [float(aug.semitones_range[0]), float(aug.semitones_range[1])],
        "overlay_enabled": bool(aug.overlay_enabled),
        "snr_db_range": [float(aug.snr_db_range[0]), float(aug.snr_db_range[1])],
        "scale_noise_to_snr": bool(aug.scale_noise_to_snr),
        "prevent_clipping": bool(aug.prevent_clipping),
        "transient_window_ms": int(aug.transient_window_ms),
        "mask_filter_enabled": bool(aug.mask_filter_enabled),
        "mask_candidate_attempts": int(aug.mask_candidate_attempts),
        "mask_min_mean_db": _optional_float(aug.mask_min_mean_db),
        "mask_min_p25_db": _optional_float(aug.mask_min_p25_db),
        "mask_min_band_db": _optional_float(aug.mask_min_band_db),
        "mask_min_effective_snr_db": _optional_float(aug.mask_min_effective_snr_db),
        "mask_band_count": int(aug.mask_band_count),
        "mask_top_frames": int(aug.mask_top_frames),
    }


def _normalise_augmentation_settings(raw: object, cfg: Config | None = None) -> dict:
    settings = _augmentation_settings_from_cfg(cfg)
    if not isinstance(raw, dict):
        return settings
    for key in _AUG_BOOL_FIELDS:
        if key in raw:
            settings[key] = _bool_value(raw[key])
    for key in _AUG_INT_FIELDS:
        if key in raw and raw[key] not in (None, ""):
            try:
                settings[key] = int(float(raw[key]))
            except Exception:  # noqa: BLE001
                pass
    for key in _AUG_FLOAT_FIELDS:
        if key in raw and raw[key] not in (None, ""):
            try:
                settings[key] = float(raw[key])
            except Exception:  # noqa: BLE001
                pass
    for key in _AUG_OPTIONAL_FLOAT_FIELDS:
        if key in raw:
            settings[key] = _optional_float(raw[key])
    for key in _AUG_RANGE_FIELDS:
        if key in raw:
            value = _range_value(raw[key])
            if value is not None:
                settings[key] = value
    return settings


def _apply_augmentation_settings(cfg: Config, settings: dict) -> None:
    clean = _normalise_augmentation_settings(settings, cfg)
    for key, value in clean.items():
        if key in _AUG_RANGE_FIELDS:
            setattr(cfg.augmentation, key, tuple(value))
        else:
            setattr(cfg.augmentation, key, value)


def _ensure_ui_selection(selection: dict) -> dict:
    selection["schema_version"] = 2
    selection["ui_managed"] = True
    selection.setdefault("assignments", {})
    selection.setdefault("clip_assignments", {})
    selection.setdefault("clip_overrides", {})
    selection.setdefault("augmentation_recipes", [])
    selection.setdefault("augmentation_settings", {})
    selection["updated_at"] = _now_iso()
    return selection


def make_handler(
    cfg: Config | None,
    project_id: str,
    render_html_callable,
    layout: ProjectLayout,
    shutdown_event: threading.Event | None = None,
    max_thumbs_per_class: int = 0,
):
    """Factory that closes over the project/config so we can build a stdlib
    BaseHTTPRequestHandler subclass without globals."""
    shutdown_event = shutdown_event or threading.Event()
    augment_lock = threading.Lock()
    augment_job: dict | None = None
    augment_cancel_event = threading.Event()
    delete_lock = threading.Lock()
    delete_job: dict | None = None
    refresh_lock = threading.Lock()
    refresh_job: dict | None = None
    compose_lock = threading.Lock()
    compose_job: dict | None = None

    class Handler(BaseHTTPRequestHandler):

        # Silence the default access logging to stderr per request — too
        # noisy when 100 audio clips load on one page.
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            _ = (format, args)
            if self.path.startswith("/api/"):
                log.info("%s %s", self.command, self.path)

        # ---- GET --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                qs = parse_qs(url.query)
                run_id = (qs.get("run", [""])[0] or "").strip() or None
                try:
                    html = render_html_callable(run_id=run_id)
                except TypeError:
                    html = render_html_callable()
                self._send_html(html)
            elif url.path == "/audio":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_audio_path(layout, paths[0])
                if resolved is None:
                    self._send(403, b"forbidden"); return
                self._send_file(resolved)
            elif url.path == "/thumbnail":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                t_type = qs.get("type", ["spec"])[0]
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_audio_path(layout, paths[0])
                if resolved is None:
                    self._send(403, b"forbidden"); return
                from sed.data.inspect import _get_or_generate_thumbnail_file
                png_path = _get_or_generate_thumbnail_file(resolved, t_type)
                if png_path is None or not png_path.exists():
                    self._send(500, b"failed to generate thumbnail"); return
                self._send_file(png_path)
            elif url.path == "/image":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_asset_path(layout, paths[0], _IMAGE_SUFFIXES)
                if resolved is None:
                    self._send(403, b"forbidden"); return
                self._send_file(resolved)
            elif url.path == "/api/frozen-run/image":
                qs = parse_qs(url.query)
                run_id = (qs.get("run", [""])[0] or "").strip() or None
                kind = (qs.get("kind", [""])[0] or "").strip()
                from sed.data.inspect import frozen_run_report_image_path
                resolved = frozen_run_report_image_path(layout, run_id, kind)
                if resolved is None:
                    self._send(404, b"not found"); return
                self._send_file(resolved)
            elif url.path == "/api/frozen-run":
                qs = parse_qs(url.query)
                run_id = (qs.get("run", [""])[0] or "").strip() or None
                self._handle_frozen_run(run_id)
            elif url.path == "/api/selection":
                self._handle_selection_get()
            elif url.path == "/api/augmentation-generated-counts":
                self._handle_augmentation_generated_counts()
            elif url.path == "/api/panel-totals":
                self._handle_panel_totals()
            elif url.path == "/api/clips":
                qs = parse_qs(url.query)
                self._handle_clips_page(
                    tab=(qs.get("tab", [""])[0] or "").strip(),
                    class_name=(qs.get("class", [""])[0] or ""),
                    offset=(qs.get("offset", ["0"])[0] or "0"),
                    limit=(qs.get("limit", ["100"])[0] or "100"),
                    run_id=(qs.get("run_id", qs.get("run", [""]))[0] or "").strip() or None,
                )
            elif url.path == "/api/review-data":
                qs = parse_qs(url.query)
                scopes = qs.get("scopes", [None])[0]
                run_id = (qs.get("run_id", qs.get("run", [""]))[0] or "").strip() or None
                self._handle_review_data(scopes=scopes, run_id=run_id)
            elif url.path == "/api/review-data/status":
                qs = parse_qs(url.query)
                self._handle_review_data_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/augment/status":
                qs = parse_qs(url.query)
                self._handle_augment_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/delete/status":
                qs = parse_qs(url.query)
                self._handle_delete_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/compose/status":
                qs = parse_qs(url.query)
                self._handle_compose_status((qs.get("job_id", [""])[0] or "").strip())
            else:
                self._send(404, b"not found")

        # ---- POST -------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/api/done":
                self._handle_done()
                threading.Thread(target=shutdown_event.set, daemon=True).start()
                return
            if url.path == "/api/delete":
                self._handle_delete(); return
            if url.path == "/api/delete/start":
                self._handle_delete_start(); return
            if url.path == "/api/review-data/start":
                self._handle_review_data_start(); return
            if url.path == "/api/promote-mined":
                self._handle_promote_mined()
                return
            if url.path == "/api/selection":
                self._handle_selection()
                return
            if url.path == "/api/selection-rule":
                self._handle_selection_rule()
                return
            if url.path == "/api/augmentation-rule":
                self._handle_augmentation_rule()
                return
            if url.path == "/api/augmentation-settings":
                self._handle_augmentation_settings()
                return
            if url.path == "/api/augment/start":
                self._handle_augment_start()
                return
            if url.path == "/api/augment/cancel":
                self._handle_augment_cancel()
                return
            if url.path == "/api/compose":
                self._handle_compose()
                return
            if url.path == "/api/compose/start":
                self._handle_compose_start(); return
            if url.path == "/api/clip-role":
                self._handle_clip_role()
                return
            if url.path == "/api/evaluate-frozen":
                self._handle_evaluate_frozen()
                return
            self._send(404, b"not found")

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        def _delete_paths(self, raw_paths: list, job: dict | None = None) -> dict:
            deleted: list[str] = []
            errors: list[dict] = []
            total = len(raw_paths)
            for i, rp in enumerate(raw_paths, 1):
                resolved = _safe_resolve_audio_path(layout, rp)
                if job is not None:
                    with delete_lock:
                        job["completed"] = i - 1
                        job["current_name"] = Path(str(rp)).name
                if resolved is None:
                    errors.append({"path": str(rp), "error": "forbidden or missing"})
                    if job is not None:
                        with delete_lock:
                            job["completed"] = i
                            job["failed_count"] = len(errors)
                            job["errors"] = errors[-10:]
                    continue
                sidecar = resolved.with_suffix(".json")
                thumb = resolved.with_suffix(".spec.png")
                mix_map = resolved.with_suffix(".mix.png")
                try:
                    resolved.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    thumb.unlink(missing_ok=True)
                    mix_map.unlink(missing_ok=True)
                    deleted.append(str(resolved))
                    log.info("deleted %s", resolved.name)
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": str(resolved), "error": str(e)})
                    log.warning("delete failed %s: %s", resolved, e)
                if job is not None:
                    with delete_lock:
                        job["completed"] = i
                        job["deleted"] = list(deleted)
                        job["deleted_count"] = len(deleted)
                        job["failed_count"] = len(errors)
                        job["errors"] = errors[-10:]
            if deleted:
                if job is not None:
                    with delete_lock:
                        job["phase"] = "cleanup"
                        job["current_name"] = ""
                _cleanup_after_file_changes(layout)
            return {
                "deleted": deleted,
                "count": len(deleted),
                "errors": errors,
                "failed_count": len(errors),
                "total": total,
            }

        def _handle_delete(self) -> None:
            """Permanently delete WAV + JSON sidecar for requested paths."""
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            raw_paths = body.get("paths", [])
            if not isinstance(raw_paths, list):
                self._send(400, b"paths must be an array"); return
            result = self._delete_paths(raw_paths)
            self._send_json({"ok": True, **result})

        def _delete_job_payload(self, job: dict | None) -> dict:
            if not job:
                return {"ok": False, "error": "no delete job"}
            payload = dict(job)
            started = float(payload.get("started_at") or time.time())
            ended = payload.get("ended_at")
            payload["elapsed_s"] = max(0.0, float(ended or time.time()) - started)
            payload.pop("thread", None)
            return {"ok": True, "job": payload}

        def _handle_delete_status(self, job_id: str = "") -> None:
            with delete_lock:
                job = delete_job
                if job_id and job and str(job.get("job_id")) != job_id:
                    job = None
                payload = self._delete_job_payload(job)
            self._send_json(payload, status=200 if payload.get("ok") else 404)

        def _handle_delete_start(self) -> None:
            nonlocal delete_job
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            raw_paths = body.get("paths", [])
            if not isinstance(raw_paths, list):
                self._send(400, b"paths must be an array"); return
            with delete_lock:
                if delete_job and str(delete_job.get("state")) in {"starting", "running", "cleanup"}:
                    self._send_json({"ok": False, "error": "delete job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                delete_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "phase": "delete",
                    "total": len(raw_paths),
                    "completed": 0,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "current_name": "",
                    "deleted": [],
                    "errors": [],
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def run_job() -> None:
                with delete_lock:
                    if delete_job and delete_job.get("job_id") == job_id:
                        delete_job["state"] = "running"
                try:
                    result = self._delete_paths(raw_paths, delete_job)
                    with delete_lock:
                        if delete_job and delete_job.get("job_id") == job_id:
                            delete_job.update(result)
                            delete_job["state"] = "finished"
                            delete_job["phase"] = "finished"
                            delete_job["current_name"] = ""
                            delete_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background delete failed")
                    with delete_lock:
                        if delete_job and delete_job.get("job_id") == job_id:
                            delete_job["state"] = "error"
                            delete_job["phase"] = "error"
                            delete_job["error"] = str(exc)
                            delete_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with delete_lock:
                delete_job["thread"] = thread
                payload = self._delete_job_payload(delete_job)
                thread.start()
            self._send_json(payload)

        def _handle_promote_mined(self) -> None:
            """Move mined WAV + JSON sidecar into data_library/sounds/<label>/."""
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            raw_paths = body.get("paths", [])
            label = body.get("label", "mined_false_positive")
            if not isinstance(raw_paths, list) or not isinstance(label, str):
                self._send(400, b"invalid payload")
                return
            moved = []
            for rp in raw_paths:
                resolved = _safe_resolve_audio_path(layout, rp)
                if resolved is None:
                    continue
                dest = _promote_mined_clip(layout, cfg, resolved, label)
                if dest is None:
                    continue
                moved.append({"src": str(resolved), "dest": str(dest)})
                log.info("promoted mined clip %s -> %s", resolved.name, dest)
            self._send_json({"ok": True, "moved": moved, "count": len(moved)})

        def _handle_frozen_run(self, run_id: str | None) -> None:
            from sed.data.inspect import frozen_run_prediction_payload
            log.info("switching frozen-test inspection to run %s", run_id or "selected/latest")
            payload = frozen_run_prediction_payload(layout, run_id=run_id)
            self._send_json(payload)

        def _handle_evaluate_frozen(self) -> None:
            """Re-score one model against the CURRENT frozen test set (test.csv).

            Rewrites the model's reports/ (confusion_matrix.{png,json},
            test_metrics.json, test_predictions.tsv) so the Frozen test tab can
            refresh after the frozen set changes — no CLI round-trip needed.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            run_id = str(body.get("run_id") or "").strip()
            from sed.train import registry
            from sed.train.eval import evaluate_run
            project_id = layout.id
            model_id = run_id or (registry.get_active_model(project_id) or "")
            if not model_id:
                self._send_json(
                    {"ok": False, "error": "no model selected and no active model"},
                    status=400,
                )
                return
            log.info("evaluating model %s on current frozen test", model_id)
            try:
                metrics = evaluate_run(cfg, project_id, model_id)
            except Exception as exc:  # noqa: BLE001 — surface to the UI status line.
                log.exception("frozen-test evaluation failed")
                self._send_json({"ok": False, "error": str(exc), "model_id": model_id})
                return
            self._send_json({
                "ok": True,
                "model_id": model_id,
                "accuracy": metrics.get("accuracy"),
                "f1_macro_targets": metrics.get("f1_macro_targets"),
                "n_samples": metrics.get("n_samples"),
            })

        def _handle_selection(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection
            if body.get("action") == "assign_clips":
                class_name = str(body.get("class_name") or "").strip()
                clips = body.get("clips", [])
                if not class_name or not isinstance(clips, list):
                    self._send(400, b"invalid assignment payload")
                    return
                selection = load_dataset_selection(layout)
                by_class = selection.setdefault("clip_assignments", {})
                rows = by_class.setdefault(class_name, [])
                existing_keys = {
                    (str(r.get("uuid") or ""), str(r.get("relpath") or ""), str(r.get("path") or ""))
                    for r in rows
                    if isinstance(r, dict)
                }
                added = 0
                for clip in clips:
                    if not isinstance(clip, dict):
                        continue
                    row = {
                        k: str(clip.get(k))
                        for k in ("uuid", "relpath", "path", "label", "project_id", "dataset_role")
                        if clip.get(k)
                    }
                    if not row:
                        continue
                    key = (row.get("uuid", ""), row.get("relpath", ""), row.get("path", ""))
                    if key in existing_keys:
                        continue
                    rows.append(row)
                    existing_keys.add(key)
                    added += 1
                selection["updated_at"] = datetime.now(UTC).isoformat()
                path = write_dataset_selection(layout, selection)
                self._send_json({"ok": True, "selection": str(path), "added": added})
                return

            path = write_dataset_selection(layout, body)
            self._send_json({"ok": True, "selection": str(path)})

        def _handle_selection_get(self) -> None:
            from sed.data.compose import load_dataset_selection, selection_manifest_path
            selection = load_dataset_selection(layout)
            settings = selection.get("augmentation_settings")
            if not isinstance(settings, dict) or not settings:
                settings = _augmentation_settings_from_cfg(cfg)
                selection = dict(selection)
                selection["augmentation_settings"] = settings
            # Recompute display counts from disk so deletes/imports show through.
            try:
                _refresh_selection_counts(selection, cfg=cfg, project_id=layout.id)
            except Exception:  # noqa: BLE001 — counts are display-only.
                log.debug("selection count refresh failed", exc_info=True)
            # Annotate derived source rules with their overlay class(es) for the UI.
            try:
                _attach_overlay_labels(selection, project_id=layout.id)
            except Exception:  # noqa: BLE001 — overlay labels are display-only.
                log.debug("overlay-label annotation failed", exc_info=True)
            self._send_json({
                "ok": True,
                "selection": selection,
                "path": str(selection_manifest_path(layout)),
                "augmentation_settings": settings,
            })

        def _handle_augmentation_generated_counts(self) -> None:
            from sed.data.compose import load_dataset_selection
            from sed.data.inspect import augmentation_recipe_counts
            selection = load_dataset_selection(layout)
            settings = selection.get("augmentation_settings")
            if not isinstance(settings, dict) or not settings:
                settings = _augmentation_settings_from_cfg(cfg)
            recipes = selection.get("augmentation_recipes")
            if not isinstance(recipes, list):
                recipes = []
            self._send_json({
                "ok": True,
                "augmentation_generated_counts": augmentation_recipe_counts(
                    layout,
                    recipes=recipes,
                    # Baseline mirrors the run-time cfg snapshot; each recipe's
                    # own settings dict is layered on top inside the counter.
                    settings=_normalise_augmentation_settings(settings, cfg),
                ),
            })

        def _handle_panel_totals(self) -> None:
            if cfg is None:
                self._send(400, b"missing config")
                return
            from sed.data.inspect import panel_totals
            self._send_json({"ok": True, "panel_totals": panel_totals(cfg, layout)})

        def _handle_clips_page(
            self, tab: str, class_name: str, offset: str = "0",
            limit: str = "100", run_id: str | None = None,
        ) -> None:
            if cfg is None:
                self._send(400, b"missing config")
                return
            if tab not in {"dataset", "augmented", "frozen", "mined"}:
                self._send_json({"ok": False, "error": f"unknown tab: {tab}"}, status=400)
                return
            try:
                off = max(0, int(offset))
                lim = max(1, min(1000, int(limit)))
            except (TypeError, ValueError):
                self._send_json({"ok": False, "error": "offset/limit must be integers"}, status=400)
                return
            try:
                from sed.data.inspect import clips_page
                cards = clips_page(cfg, layout, tab, class_name, off, lim, run_id=run_id)
            except Exception as exc:  # noqa: BLE001
                log.exception("clips page failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({
                "ok": True, "tab": tab, "class": class_name,
                "offset": off, "limit": lim, "clips": cards,
            })

        def _handle_selection_rule(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection

            action = str(body.get("action") or "add").strip().lower()
            class_name = str(body.get("class_name") or "").strip()
            if action == "add" and not class_name:
                self._send(400, b"class_name is required")
                return
            if cfg is not None and class_name and class_name not in set(cfg.model.classes):
                self._send_json({"ok": False, "error": "class_name is not in target_classes"}, status=400)
                return

            selection = _ensure_ui_selection(load_dataset_selection(layout))
            assignments = selection.setdefault("assignments", {})
            removed = 0
            added = 0

            if action == "add":
                rule = _normalise_filter_rule(
                    body.get("rule") or {},
                    default_split=str(body.get("split") or "train"),
                )
                if str(rule.get("source_bucket") or "").lower() == "mined":
                    self._send_json({
                        "ok": False,
                        "error": "Mined clips must be promoted before assignment.",
                    }, status=400)
                    return
                rows = assignments.setdefault(class_name, [])
                if not isinstance(rows, list):
                    rows = [rows] if isinstance(rows, dict) else []
                    assignments[class_name] = rows
                rows.append(rule)
                added = 1
            elif action == "remove":
                rule_id = str(body.get("rule_id") or "").strip()
                targets = [class_name] if class_name else list(assignments.keys())
                for target in targets:
                    rows = assignments.get(target, [])
                    if not isinstance(rows, list):
                        continue
                    kept = []
                    for row in rows:
                        if isinstance(row, dict) and rule_id and str(row.get("id") or "") == rule_id:
                            removed += 1
                            continue
                        kept.append(row)
                    if kept:
                        assignments[target] = kept
                    else:
                        assignments.pop(target, None)
            elif action == "clear":
                if class_name:
                    removed = len(assignments.get(class_name, []) or [])
                    assignments[class_name] = []
                else:
                    removed = sum(len(v) for v in assignments.values() if isinstance(v, list))
                    selection["assignments"] = {}
            else:
                self._send(400, b"unknown action")
                return

            selection["updated_at"] = _now_iso()
            path = write_dataset_selection(layout, selection)
            self._send_json({
                "ok": True,
                "selection": selection,
                "path": str(path),
                "added": added,
                "removed": removed,
            })

        def _handle_augmentation_settings(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection

            raw_settings = body.get("settings", body)
            settings = _normalise_augmentation_settings(raw_settings, cfg)
            selection = _ensure_ui_selection(load_dataset_selection(layout))
            selection["augmentation_settings"] = settings
            selection["updated_at"] = _now_iso()
            path = write_dataset_selection(layout, selection)
            self._send_json({
                "ok": True,
                "settings": settings,
                "selection": selection,
                "path": str(path),
            })

        def _handle_augmentation_rule(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection

            action = str(body.get("action") or "add").strip().lower()

            selection = _ensure_ui_selection(load_dataset_selection(layout))
            recipes = selection.setdefault("augmentation_recipes", [])
            if not isinstance(recipes, list):
                recipes = []
                selection["augmentation_recipes"] = recipes
            added = 0
            removed = 0
            deleted_generated = 0

            if action == "add":
                recipe = _normalise_augmentation_recipe(body.get("recipe") or {})
                try:
                    recipe = _refresh_recipe_counts(
                        recipe, selection=selection, cfg=cfg, project_id=project_id,
                    ) if cfg is not None else recipe
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                recipes.append(recipe)
                added = 1
            elif action == "update":
                raw_patch = body.get("recipe") or {}
                target_id = str(raw_patch.get("recipe_id") or raw_patch.get("id") or "").strip()
                replaced = False
                for i, row in enumerate(recipes):
                    if (
                        isinstance(row, dict)
                        and str(row.get("recipe_id") or row.get("id") or "") == target_id
                    ):
                        # Merge the raw patch onto the existing row BEFORE
                        # normalising, so unspecified fields (filters, mask
                        # thresholds, description, …) keep their saved values
                        # instead of being defaulted away by the normaliser.
                        merged_raw = {**row, **raw_patch}
                        recipe = _normalise_augmentation_recipe(merged_raw)
                        try:
                            recipe = _refresh_recipe_counts(
                                recipe, selection=selection, cfg=cfg, project_id=project_id,
                            ) if cfg is not None else recipe
                        except ValueError as exc:
                            self._send_json({"ok": False, "error": str(exc)}, status=400)
                            return
                        recipes[i] = recipe
                        replaced = True
                        break
                if not replaced:
                    self._send_json(
                        {"ok": False, "error": "recipe not found"}, status=404,
                    )
                    return
                added = 1
            elif action == "delete_generated":
                rule_id = str(body.get("rule_id") or body.get("recipe_id") or "").strip()
                if not rule_id:
                    self._send_json(
                        {"ok": False, "error": "rule_id required"}, status=400,
                    )
                    return
                from sed.data.augment import delete_generated_for_recipes
                deleted_generated = delete_generated_for_recipes(
                    project_id,
                    [rule_id],
                    frozen_test=None,
                )
                # No selection mutation — just respond with current state.
                selection["updated_at"] = _now_iso()
                self._send_json({
                    "ok": True,
                    "selection": selection,
                    "added": 0,
                    "removed": 0,
                    "deleted_generated": deleted_generated,
                })
                return
            elif action == "remove":
                rule_id = str(body.get("rule_id") or "").strip()
                kept = []
                removed_ids: list[str] = []
                for row in recipes:
                    if isinstance(row, dict) and rule_id and str(row.get("id") or row.get("recipe_id") or "") == rule_id:
                        removed += 1
                        removed_ids.append(str(row.get("recipe_id") or row.get("id") or rule_id))
                        continue
                    kept.append(row)
                selection["augmentation_recipes"] = kept
                if body.get("delete_generated") and removed_ids:
                    from sed.data.augment import delete_generated_for_recipes
                    deleted_generated = delete_generated_for_recipes(
                        project_id,
                        removed_ids,
                        frozen_test=None,
                    )
            elif action == "clear":
                removed_ids = [
                    str(row.get("recipe_id") or row.get("id") or "")
                    for row in recipes
                    if isinstance(row, dict)
                ]
                removed = len(recipes)
                selection["augmentation_recipes"] = []
                if body.get("delete_generated") and removed_ids:
                    from sed.data.augment import delete_generated_for_recipes
                    deleted_generated = delete_generated_for_recipes(
                        project_id,
                        removed_ids,
                        frozen_test=None,
                    )
            else:
                self._send(400, b"unknown action")
                return

            selection["updated_at"] = _now_iso()
            path = write_dataset_selection(layout, selection)
            self._send_json({
                "ok": True,
                "selection": selection,
                "path": str(path),
                "added": added,
                "removed": removed,
                "deleted_generated": deleted_generated,
            })

        def _augment_job_payload(self, job: dict | None) -> dict:
            if not job:
                return {"ok": False, "error": "no augmentation job"}
            payload = dict(job)
            started = float(payload.get("started_at") or time.time())
            ended = payload.get("ended_at")
            elapsed = float(ended or time.time()) - started
            payload["elapsed_s"] = max(0.0, elapsed)
            completed = int(payload.get("completed") or 0)
            payload["clips_per_sec"] = (completed / elapsed) if elapsed > 0 else 0.0
            payload.pop("thread", None)
            return {"ok": True, "job": payload}

        def _handle_augment_status(self, job_id: str = "") -> None:
            with augment_lock:
                job = augment_job
                if job_id and job and str(job.get("job_id")) != job_id:
                    job = None
                payload = self._augment_job_payload(job)
            self._send_json(payload, status=200 if payload.get("ok") else 404)

        def _handle_augment_cancel(self) -> None:
            with augment_lock:
                job = augment_job
                if not job or str(job.get("state")) not in {"running", "starting"}:
                    self._send_json({"ok": False, "error": "no running augmentation job"}, status=404)
                    return
                augment_cancel_event.set()
                job["state"] = "canceling"
                payload = self._augment_job_payload(job)
            self._send_json(payload)

        def _handle_augment_start(self) -> None:
            nonlocal augment_job
            if cfg is None:
                self._send(400, b"missing config")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection
            from sed.data.augment import (
                AugmentationCancelled,
                AugmentationDisabled,
                materialize_library_recipes,
            )

            selection = load_dataset_selection(layout)
            all_recipes = selection.get("augmentation_recipes", [])
            if not isinstance(all_recipes, list) or not all_recipes:
                self._send_json({"ok": False, "error": "No UI augmentation recipes saved."}, status=400)
                return

            # Select recipes by id. New shape: recipe_ids: [...]; legacy single
            # recipe_id is still accepted. If neither is provided, fall back to
            # the persisted per-recipe "selected" flag (default true) so old
            # clients keep working.
            raw_ids = body.get("recipe_ids")
            recipe_ids: list[str] = []
            if isinstance(raw_ids, list):
                recipe_ids = [str(rid).strip() for rid in raw_ids if str(rid).strip()]
            legacy_id = str(body.get("recipe_id") or "").strip()
            if not recipe_ids and legacy_id:
                recipe_ids = [legacy_id]

            def _row_id(row: dict) -> str:
                return str(row.get("recipe_id") or row.get("id") or "").strip()

            if recipe_ids:
                wanted = set(recipe_ids)
                recipes = [row for row in all_recipes if isinstance(row, dict) and _row_id(row) in wanted]
            else:
                # Honor the per-recipe `selected` flag when the client didn't
                # explicitly select. Default True keeps prior behavior of
                # "generate everything" when the field is absent.
                recipes = [
                    row for row in all_recipes
                    if isinstance(row, dict) and bool(row.get("selected", True))
                ]

            if not recipes:
                self._send_json(
                    {"ok": False, "error": "No recipes selected for generation."}, status=400,
                )
                return
            with augment_lock:
                if augment_job and str(augment_job.get("state")) in {"starting", "running", "canceling"}:
                    self._send_json({"ok": False, "error": "augmentation job already running"}, status=409)
                    return
                augment_cancel_event.clear()
                job_id = uuid.uuid4().hex[:12]
                generation_id = uuid.uuid4().hex[:12]
                augment_job = {
                    "job_id": job_id,
                    "generation_id": generation_id,
                    "state": "starting",
                    "total": 0,
                    "completed": 0,
                    "generated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "current_recipe": "",
                    "counts": {},
                    "warnings": [],
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def progress(update: dict) -> None:
                with augment_lock:
                    if not augment_job or augment_job.get("job_id") != job_id:
                        return
                    for key in (
                        "state", "total", "completed", "generated", "skipped",
                        "failed", "current_recipe", "generation_id", "warnings",
                    ):
                        if key in update:
                            augment_job[key] = update[key]

            def run_job() -> None:
                nonlocal augment_job
                run_cfg = cfg.model_copy(deep=True)
                _apply_augmentation_settings(run_cfg, selection.get("augmentation_settings", {}))
                try:
                    counts = materialize_library_recipes(
                        run_cfg,
                        project_id,
                        recipes,
                        seed=int(body["seed"]) if body.get("seed") not in (None, "") else None,
                        generation_id=generation_id,
                        progress=progress,
                        should_cancel=augment_cancel_event.is_set,
                    )
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["counts"] = counts
                            augment_job["state"] = "finished"
                            augment_job["ended_at"] = time.time()
                except AugmentationCancelled:
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["state"] = "canceled"
                            augment_job["ended_at"] = time.time()
                except AugmentationDisabled as exc:
                    log.warning("augmentation disabled: %s", exc)
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["state"] = "error"
                            augment_job["error"] = str(exc)
                            augment_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background UI augmentation failed")
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["state"] = "error"
                            augment_job["error"] = str(exc)
                            augment_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with augment_lock:
                # augment_job was set unconditionally above inside this same
                # critical section, so it's always truthy here. Keep the lock
                # held across thread.start() so the "running" state is visible
                # before the worker can change it.
                augment_job["thread"] = thread
                augment_job["state"] = "running"
                payload = self._augment_job_payload(augment_job)
                thread.start()
            self._send_json(payload)

        def _handle_review_data(self, scopes: object | None = None, run_id: str | None = None) -> None:
            if cfg is None:
                self._send(400, b"missing config")
                return
            try:
                from sed.data.inspect import review_data_payload
                payload = review_data_payload(
                    cfg,
                    layout,
                    max_per_class=max_thumbs_per_class,
                    run_id=run_id,
                    scopes=scopes,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("review refresh failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json(payload)

        def _refresh_job_payload(self, job: dict | None) -> dict:
            if not job:
                return {"ok": False, "error": "no refresh job"}
            payload = dict(job)
            started = float(payload.get("started_at") or time.time())
            ended = payload.get("ended_at")
            elapsed = max(0.0, float(ended or time.time()) - started)
            payload["elapsed_s"] = elapsed
            progress_value = float(payload.get("progress") or 0.0)
            if progress_value > 0 and progress_value < 1 and elapsed > 0:
                payload["eta_s"] = max(0.0, elapsed * (1.0 - progress_value) / progress_value)
            elif str(payload.get("state") or "") == "finished":
                payload["eta_s"] = 0.0
            else:
                payload["eta_s"] = None
            payload.pop("thread", None)
            return {"ok": True, "job": payload}

        def _handle_review_data_status(self, job_id: str = "") -> None:
            with refresh_lock:
                job = refresh_job
                if job_id and job and str(job.get("job_id")) != job_id:
                    job = None
                payload = self._refresh_job_payload(job)
            self._send_json(payload, status=200 if payload.get("ok") else 404)

        def _handle_review_data_start(self) -> None:
            nonlocal refresh_job
            if cfg is None:
                self._send(400, b"missing config")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            try:
                from sed.data.inspect import normalise_review_refresh_scopes
                scopes, full_refresh = normalise_review_refresh_scopes(body.get("scopes"))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            run_id = str(body.get("run_id") or body.get("run") or "").strip() or None
            with refresh_lock:
                if refresh_job and str(refresh_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "refresh job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                refresh_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "phase": "starting",
                    "phase_detail": "starting",
                    "phase_count": 0,
                    "progress": 0.0,
                    "eta_s": None,
                    "scopes": scopes,
                    "full_refresh": full_refresh,
                    "summary_counts": {},
                    "payload": None,
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def progress(update: dict) -> None:
                with refresh_lock:
                    if not refresh_job or refresh_job.get("job_id") != job_id:
                        return
                    refresh_job["state"] = "running"
                    if "phase" in update:
                        refresh_job["phase"] = update["phase"]
                    if "phase_detail" in update:
                        refresh_job["phase_detail"] = update["phase_detail"]
                    if "count" in update:
                        refresh_job["phase_count"] = update["count"]
                    if "progress" in update:
                        refresh_job["progress"] = update["progress"]

            def run_job() -> None:
                try:
                    from sed.data.inspect import review_data_payload
                    payload = review_data_payload(
                        cfg,
                        layout,
                        max_per_class=max_thumbs_per_class,
                        run_id=run_id,
                        scopes=None if full_refresh else scopes,
                        progress=progress,
                    )
                    with refresh_lock:
                        if refresh_job and refresh_job.get("job_id") == job_id:
                            refresh_job["payload"] = payload
                            refresh_job["summary_counts"] = payload.get("summary_counts", {})
                            refresh_job["state"] = "finished"
                            refresh_job["phase"] = "finished"
                            refresh_job["phase_detail"] = "finished"
                            refresh_job["progress"] = 1.0
                            refresh_job["eta_s"] = 0.0
                            refresh_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background review refresh failed")
                    with refresh_lock:
                        if refresh_job and refresh_job.get("job_id") == job_id:
                            refresh_job["state"] = "error"
                            refresh_job["phase"] = "error"
                            refresh_job["phase_detail"] = "error"
                            refresh_job["error"] = str(exc)
                            refresh_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with refresh_lock:
                refresh_job["thread"] = thread
                payload = self._refresh_job_payload(refresh_job)
                thread.start()
            self._send_json(payload)

        def _handle_clip_role(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            raw_paths = body.get("paths", [])
            dataset_role = str(body.get("dataset_role") or "").strip()
            if dataset_role not in {"train", "test", "ignore"}:
                self._send(400, b"dataset_role must be train, test, or ignore")
                return
            if not isinstance(raw_paths, list):
                self._send(400, b"paths must be an array")
                return
            updated = []
            for rp in raw_paths:
                resolved = _safe_resolve_audio_path(layout, rp)
                if resolved is None:
                    continue
                sidecar = resolved.with_suffix(".json")
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                    meta["dataset_role"] = dataset_role
                    meta["review_status"] = meta.get("review_status") or "accepted"
                    if dataset_role == "test":
                        meta["project_id"] = layout.id
                    meta["updated_at"] = datetime.now(UTC).isoformat()
                    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    updated.append(str(resolved))
                except Exception as exc:  # noqa: BLE001
                    log.warning("clip role update failed %s: %s", resolved, exc)
            try:
                from sed.data.library import Library
                Library().rebuild_manifest()
            except Exception as exc:  # noqa: BLE001
                log.warning("manifest rebuild after role update failed: %s", exc)
            self._send_json({"ok": True, "updated": updated, "count": len(updated)})

        def _handle_compose(self) -> None:
            if cfg is None:
                self._send(400, b"missing config")
                return
            try:
                from sed.data.compose import compose_dataset
                summary = compose_dataset(cfg, project_id)
            except Exception as exc:  # noqa: BLE001
                log.exception("compose from inspect server failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, "summary": summary})

        def _compose_job_payload(self, job: dict | None) -> dict:
            if not job:
                return {"ok": False, "error": "no compose job"}
            payload = dict(job)
            started = float(payload.get("started_at") or time.time())
            ended = payload.get("ended_at")
            payload["elapsed_s"] = max(0.0, float(ended or time.time()) - started)
            payload.pop("thread", None)
            return {"ok": True, "job": payload}

        def _handle_compose_status(self, job_id: str = "") -> None:
            with compose_lock:
                job = compose_job
                if job_id and job and str(job.get("job_id")) != job_id:
                    job = None
                payload = self._compose_job_payload(job)
            self._send_json(payload, status=200 if payload.get("ok") else 404)

        def _handle_compose_start(self) -> None:
            nonlocal compose_job
            if cfg is None:
                self._send(400, b"missing config")
                return
            with compose_lock:
                if compose_job and str(compose_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "compose job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                compose_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "phase": "starting",
                    "summary": {},
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def run_job() -> None:
                nonlocal compose_job
                try:
                    from sed.data.compose import compose_dataset
                    with compose_lock:
                        if compose_job and compose_job.get("job_id") == job_id:
                            compose_job["state"] = "running"
                            compose_job["phase"] = "composing"
                    summary = compose_dataset(cfg, project_id)
                    with compose_lock:
                        if compose_job and compose_job.get("job_id") == job_id:
                            compose_job["summary"] = summary
                            compose_job["state"] = "finished"
                            compose_job["phase"] = "finished"
                            compose_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background compose failed")
                    with compose_lock:
                        if compose_job and compose_job.get("job_id") == job_id:
                            compose_job["state"] = "error"
                            compose_job["phase"] = "error"
                            compose_job["error"] = str(exc)
                            compose_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with compose_lock:
                compose_job["thread"] = thread
                payload = self._compose_job_payload(compose_job)
                thread.start()
            self._send_json(payload)

        def _handle_done(self) -> None:
            """Persist the currently ticked clips as the next-compose exclusion set."""
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw_paths = []
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    raw_paths = body.get("exclude_paths", body.get("paths", []))
                except Exception:  # noqa: BLE001
                    self._send(400, b"invalid JSON")
                    return
            if not isinstance(raw_paths, list):
                self._send(400, b"exclude_paths must be an array")
                return
            excluded = []
            for rp in raw_paths:
                resolved = _safe_resolve_audio_path(layout, rp)
                if resolved is not None:
                    excluded.append(resolved)
            from sed.data.compose import write_excluded_clip_paths
            manifest = write_excluded_clip_paths(layout, excluded)
            _cleanup_after_file_changes(layout)
            self._send_json({
                "ok": True,
                "excluded": [str(p) for p in excluded],
                "count": len(excluded),
                "manifest": str(manifest),
            })

        # ---- helpers ----------------------------------------------------

        def _send(self, status: int, body: bytes, ctype: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _send_json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send_html(self, html: str) -> None:
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_file(self, path: Path) -> None:
            ctype, _ = mimetypes.guess_type(str(path))
            ctype = ctype or "application/octet-stream"
            try:
                size = path.stat().st_size
            except OSError:
                self._send(404, b"not found"); return
            # Honour Range requests so the browser can seek long audio
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if rng and rng.startswith("bytes="):
                try:
                    spec = rng[6:]
                    sp, ep = spec.split("-")
                    if sp: start = int(sp)
                    if ep: end = int(ep)
                    end = min(end, size - 1)
                    status = 206
                except Exception:  # noqa: BLE001
                    status = 200; start, end = 0, size - 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(
    cfg: Config,
    project_id: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    max_thumbs_per_class: int = 0,
    run_id: str | None = None,
) -> None:
    """Start the interactive inspect server. Blocks until Ctrl+C."""
    from sed.data.inspect import inspect_project
    _cleanup_after_file_changes(ProjectLayout(project_id))
    # Re-render the HTML on every GET / so newly recorded clips appear
    # after a browser refresh.
    layout = ProjectLayout(project_id)

    log.info("rendering initial report ...")
    out_path = inspect_project(
        cfg, project_id,
        max_thumbs_per_class=max_thumbs_per_class,
        interactive=True,
        run_id=run_id,
    )
    log.info("static report also written to %s", out_path)

    def render_html_now(requested_run_id: str | None = None) -> str:
        from sed.data.inspect import _quick_rerender
        return _quick_rerender(cfg, project_id,
                               max_thumbs_per_class=max_thumbs_per_class,
                               run_id=requested_run_id or run_id)

    shutdown_event = threading.Event()
    Handler = make_handler(
        cfg,
        project_id,
        render_html_now,
        layout,
        shutdown_event,
        max_thumbs_per_class=max_thumbs_per_class,
    )

    # Bind, possibly auto-incrementing the port if busy
    for try_port in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, try_port), Handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"could not bind any port in [{port},{port+19}]")

    url = f"http://{host}:{httpd.server_port}/"
    log.info("inspect server listening on %s", url)
    log.info("Ctrl+C, or press 'Done' in the browser, to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    # Serve on a worker thread so the main thread can also watch for the
    # in-browser "Done" button (which sets shutdown_event).
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not shutdown_event.is_set():
            if shutdown_event.wait(timeout=0.5):
                break
        log.info("Done received from browser — shutting down.")
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.shutdown()
        server_thread.join(timeout=2.0)
