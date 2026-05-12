"""Tests for `sed evaluate augmented` — the model audit of augmented clips."""

import json

import numpy as np
import pytest
import soundfile as sf


def _write_derived_clip(
    library_dir,
    project_id: str,
    split: str,
    label: str,
    name: str,
    *,
    recipe_id: str,
    masking_score_db=None,
    effective_snr_db=None,
):
    out_dir = library_dir / "derived" / project_id / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{name}.wav"
    sf.write(str(wav), np.zeros(32000, dtype=np.float32), 32000)
    sidecar = wav.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "label": label,
        "base_label": label,
        "recipe_id": recipe_id,
        "masking_score_db": masking_score_db,
        "effective_snr_db": effective_snr_db,
    }), encoding="utf-8")
    return wav


def test_audit_augmented_report_shape_and_seen_flag(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.train import eval as eval_mod
    from sed.train import registry

    layout = ProjectLayout("p")
    layout.create()
    lib = tmp_path / "data_library"

    seen_wav = _write_derived_clip(
        lib, "p", "trainval", "glass", "a",
        recipe_id="r1", masking_score_db=7.2, effective_snr_db=4.0,
    )
    miss_wav = _write_derived_clip(
        lib, "p", "trainval", "glass", "b",
        recipe_id="r1", masking_score_db=1.0, effective_snr_db=-9.0,
    )
    frozen_wav = _write_derived_clip(
        lib, "p", "test", "glass", "c",
        recipe_id="r2", masking_score_db=8.0, effective_snr_db=6.0,
    )
    # A clip whose label the model does not know is skipped, not scored.
    _write_derived_clip(lib, "p", "trainval", "mystery", "d", recipe_id="r3")

    # Minimal model bundle: classes + a train.csv naming the "seen" clip.
    model_dir = layout.model_dir("m1")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "classes.json").write_text(
        json.dumps({"classes": ["_negative", "glass"]}), encoding="utf-8",
    )
    dataset_dir = model_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.csv").write_text(
        "path,class_name\n" f"{seen_wav},glass\n", encoding="utf-8",
    )

    monkeypatch.setattr(registry, "get_active_model", lambda project_id: "m1")
    monkeypatch.setattr(registry, "get_checkpoint", lambda project_id, model_id: model_dir / "ckpt.ckpt")

    def fake_collect(run_cfg, project_id, ckpt_path, audit_dir, classes):
        import csv as csv_mod
        with open(audit_dir / "test.csv", encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))
        # glass confident for all but the "miss" clip, which the negative gate
        # catches (high _negative probability).
        probs = []
        for row in rows:
            if row["path"] == str(miss_wav):
                probs.append([0.9, 0.1])
            else:
                probs.append([0.05, 0.95])
        probs = np.asarray(probs, dtype=np.float32)
        return probs, np.zeros_like(probs), classes, rows

    monkeypatch.setattr(eval_mod, "_collect_predictions", fake_collect)

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})
    report = eval_mod.audit_augmented(cfg, "p")

    assert report["model_id"] == "m1"
    assert report["n_clips"] == 3
    assert report["skipped_labels"] == ["mystery"]

    per_clip = report["per_clip"]
    assert per_clip[str(seen_wav)]["correct"] is True
    assert per_clip[str(seen_wav)]["seen_in_training"] is True
    assert per_clip[str(miss_wav)]["correct"] is False
    assert per_clip[str(miss_wav)]["predicted_class"] == "_negative"
    assert per_clip[str(miss_wav)]["seen_in_training"] is False
    assert per_clip[str(frozen_wav)]["correct"] is True
    assert per_clip[str(frozen_wav)]["split"] == "test"

    summary = report["summary"]
    overall = summary["overall"]["all"]
    assert overall["total"] == 3 and overall["recognized"] == 2
    # Unseen rate excludes the memorized training clip.
    assert overall["unseen_total"] == 2 and overall["unseen_recognized"] == 1
    assert summary["by_recipe"]["r1"]["total"] == 2
    assert summary["by_mask_bin"]["5–10"]["recognized"] == 2
    assert summary["by_mask_bin"]["0–5"]["recognized"] == 0
    assert summary["by_effective_snr_bin"]["-10–0"]["total"] == 1

    out_path = layout.dataset_inspect_dir / "augment_audit.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["n_clips"] == 3


def test_audit_augmented_errors_without_model_or_clips(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.train import eval as eval_mod
    from sed.train import registry

    ProjectLayout("p").create()
    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})

    monkeypatch.setattr(registry, "get_active_model", lambda project_id: None)
    with pytest.raises(RuntimeError, match="no trained model"):
        eval_mod.audit_augmented(cfg, "p")

    monkeypatch.setattr(registry, "get_active_model", lambda project_id: "m1")
    model_dir = ProjectLayout("p").model_dir("m1")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "classes.json").write_text(
        json.dumps({"classes": ["_negative", "glass"]}), encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no augmented clips"):
        eval_mod.audit_augmented(cfg, "p")
