"""Model bundle registry.

Each `sed train` writes one self-contained model bundle at
`projects/<id>/models/<model_id>/` holding the effective config snapshot, the
checkpoint, the exported OpenVINO model, the training curves, the test-set
metrics + confusion matrix, and the dataset CSVs it was trained on. There is no
separate runs/ directory and no deploy-copy step: a model is selected simply by
name via `[deploy] active_model` in config.toml (`sed models select <id>`).

This module owns ID generation, manifest read/write, listing, checkpoint
lookup, selection (active_model), and deletion.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sed.config import ProjectLayout, load_config

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SLUG_MAX = 24


def _slugify(text: str, max_len: int = _SLUG_MAX) -> str:
    slug = _SLUG_RE.sub("-", text.strip()).strip("-").lower()
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def new_model_id(
    project_id: str,
    description: str | None = None,
    now: datetime | None = None,
) -> str:
    """Build a unique, sortable model ID: ``YYYY-MM-DD_HHMM`` plus a short slug
    derived from the description. Collisions get a numeric suffix."""
    now = now or datetime.now(timezone.utc)
    base = now.strftime("%Y-%m-%d_%H%M")
    slug = _slugify(description or "")
    if slug:
        base = f"{base}-{slug}"
    layout = ProjectLayout(project_id)
    candidate = base
    n = 2
    while layout.model_dir(candidate).exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def load_manifests(project_id: str) -> list[dict]:
    """Return every model manifest for a project, newest first."""
    layout = ProjectLayout(project_id)
    if not layout.models_dir.exists():
        return []
    manifests: list[dict] = []
    for model_dir in sorted(layout.models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifests.append(json.load(f))
        except Exception:  # noqa: BLE001
            continue
    manifests.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return manifests


def load_manifest(project_id: str, model_id: str) -> dict:
    layout = ProjectLayout(project_id)
    path = layout.model_dir(model_id) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No model manifest at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(project_id: str, model_id: str, manifest: dict) -> Path:
    layout = ProjectLayout(project_id)
    path = layout.model_dir(model_id) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    tmp.replace(path)
    return path


def get_checkpoint(project_id: str, model_id: str) -> Path:
    """Locate the trained checkpoint inside a model bundle."""
    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        checkpoint = load_manifest(project_id, model_id).get("files", {}).get("checkpoint")
        if checkpoint:
            path = model_dir / checkpoint
            if path.exists():
                return path
    candidates = sorted(model_dir.glob("checkpoint_*.ckpt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Model has no checkpoint in {model_dir}")


def get_active_model(project_id: str) -> str | None:
    """Return the active model id, or None.

    Reads [deploy] active_model from config.toml, falling back to the only
    trained model when exactly one is present.
    """
    layout = ProjectLayout(project_id)
    try:
        active = load_config(project_id).deploy.active_model
    except Exception:  # noqa: BLE001
        active = ""
    if active and (layout.model_dir(active) / "training_config.json").exists():
        return active
    models = layout.list_models()
    if len(models) == 1:
        return models[0]
    return None


def set_active_model(project_id: str, model_id: str) -> None:
    """Point [deploy] active_model at model_id in the project config.toml,
    preserving comments and formatting. No-op if the config does not exist."""
    layout = ProjectLayout(project_id)
    path = layout.config_toml
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")

    # 1. An active_model line already exists → replace its value.
    new_text, n = re.subn(
        r"(?m)^(\s*active_model\s*=\s*).*$",
        lambda m: f'{m.group(1)}"{model_id}"',
        text,
        count=1,
    )
    if n:
        path.write_text(new_text, encoding="utf-8")
        return

    # 2. A [deploy] section exists but has no active_model → insert under it.
    new_text, n = re.subn(
        r"(?m)^(\[deploy\][^\n]*\n)",
        lambda m: f'{m.group(1)}active_model = "{model_id}"\n',
        text,
        count=1,
    )
    if n:
        path.write_text(new_text, encoding="utf-8")
        return

    # 3. No [deploy] section at all → append one.
    path.write_text(
        text.rstrip("\n") + f'\n\n[deploy]\nactive_model = "{model_id}"\n',
        encoding="utf-8",
    )


def backfill_model_stats(
    project_id: str, *, force: bool = False, dry_run: bool = False,
) -> list[dict]:
    """Fill ``manifest.model_stats`` for existing model bundles.

    Older runs never persisted parameter counts or the resolved input geometry.
    This walks every bundle with a manifest, sums the checkpoint state_dict
    numel on CPU, and derives the input shape from the bundle's config
    snapshot. Bundles without a manifest are skipped on purpose — creating one
    would resurrect aborted runs in the Runs list. ``trainable_params`` is not
    recoverable from a checkpoint and is intentionally absent.
    """
    from sed.config import load_config_snapshot
    from sed.train.backbones import model_input_geometry

    layout = ProjectLayout(project_id)
    results: list[dict] = []
    if not layout.models_dir.exists():
        return results
    for model_dir in sorted(layout.models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        record: dict = {"model_id": model_id, "status": ""}
        results.append(record)
        if not (model_dir / "manifest.json").exists():
            record["status"] = "skipped_no_manifest"
            continue
        manifest = load_manifest(project_id, model_id)
        if manifest.get("model_stats") and not force:
            record["status"] = "skipped_present"
            record["model_stats"] = manifest["model_stats"]
            continue
        snapshot_path = model_dir / "training_config.json"
        if not snapshot_path.exists():
            record["status"] = "skipped_no_config"
            continue
        try:
            cfg = load_config_snapshot(snapshot_path)
        except Exception as exc:  # noqa: BLE001
            record["status"] = f"error_config: {exc}"
            continue
        try:
            checkpoint = get_checkpoint(project_id, model_id)
        except FileNotFoundError:
            record["status"] = "skipped_no_checkpoint"
            continue
        try:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("state_dict") or {}
            total = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
        except Exception as exc:  # noqa: BLE001
            record["status"] = f"error_checkpoint: {exc}"
            continue
        stats = {
            "total_params": total,
            **model_input_geometry(cfg),
            "source": "backfill_checkpoint",
        }
        record["model_stats"] = stats
        record["status"] = "would_update" if dry_run else "updated"
        if not dry_run:
            manifest["model_stats"] = stats
            save_manifest(project_id, model_id, manifest)
    return results


def delete_model(project_id: str, model_id: str) -> Path:
    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    if not model_dir.exists():
        raise FileNotFoundError(f"No such model: {model_dir}")
    shutil.rmtree(model_dir)
    # Clear the active pointer if it named the deleted model.
    try:
        if load_config(project_id).deploy.active_model == model_id:
            set_active_model(project_id, "")
    except Exception:  # noqa: BLE001
        pass
    return model_dir
