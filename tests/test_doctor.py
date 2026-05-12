"""Tests for the doctor pre-flight checks (the parts that don't need
real audio hardware or a trained model)."""

import json
from pathlib import Path

import numpy as np

from sed.config import ProjectLayout, load_config
from sed import doctor


def _write_csv(path: Path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "session_id", "sample_rate",
                    "duration_s", "is_target", "origin"])
        for r in rows:
            w.writerow(r)


def test_doctor_train_detects_missing_dataset(tmp_path, monkeypatch):
    # Point ProjectLayout at a temp dir
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("nope")
    layout.create()
    cfg = load_config()
    checks = doctor.doctor_train(cfg, "nope")
    # layout.create() makes dataset_dir exist but the split CSVs don't —
    # the doctor should FAIL on the missing train/val/test.csv files.
    fails = [c for c in checks if c.status == "FAIL"]
    assert fails, f"expected at least one FAIL, got: {[c.name for c in checks]}"
    assert any(".csv" in c.name or "dataset" in c.name for c in fails), \
        f"expected dataset/CSV-related FAIL, got: {[c.name for c in fails]}"


def _seed_negatives(tmp_path: Path, n_clips: int = 12, total_min: float = 25.0,
                    labels: tuple[str, ...] = ("bar_busy", "restaurant_dinner", "cafe")):
    """Create a tmp negative pool + library manifest entries for doctor checks."""
    import json as _json
    import soundfile as _sf
    neg_dir = tmp_path / "data_library" / "negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "data_library" / "manifest.jsonl"
    sr = 32000
    per_clip_s = (total_min * 60) / n_clips
    samples_per_clip = int(per_clip_s * sr)
    with open(manifest, "w", encoding="utf-8") as f:
        for i in range(n_clips):
            label = labels[i % len(labels)]
            (neg_dir / label).mkdir(parents=True, exist_ok=True)
            uuid = f"clip_{i}"
            wav = (neg_dir / label / f"{uuid}.wav")
            # Tiny silent WAV — enough for AudioPool to count it
            _sf.write(str(wav), np.zeros(samples_per_clip, dtype="float32"), sr)
            density = "busy" if i % 2 == 0 else "normal"
            entry = {
                "uuid": uuid,
                "relpath": str(wav.relative_to(tmp_path / "data_library")),
                "category": "negatives",
                "label": label,
                "duration_s": per_clip_s,
                "sample_rate": sr,
                "channels": 1,
                "recorded_at": "2026-01-01T00:00:00",
                "session_id": "",
                "device": "",
                "mic_position": "",
                "tags": [f"density:{density}"],
                "notes": "",
            }
            wav.with_suffix(".json").write_text(
                _json.dumps(entry, indent=2) + "\n",
                encoding="utf-8",
            )
            f.write(_json.dumps(entry) + "\n")
    return neg_dir


def _patch_library_paths(monkeypatch, tmp_path: Path):
    """Wire the augment + library + config modules to a tmp data_library dir."""
    from sed import config as cfg_mod
    from sed.data import augment as aug_mod
    from sed.data import library as lib_mod
    new_library = tmp_path / "data_library"
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", new_library)
    monkeypatch.setattr(aug_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", new_library)


def test_doctor_train_passes_minimal_dataset(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    _patch_library_paths(monkeypatch, tmp_path)
    _seed_negatives(tmp_path)

    layout = ProjectLayout("p")
    layout.create()
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)

    # Compose a valid mini-dataset with NO session leakage and 3 classes
    classes = ["_negative", "clink_wine_thin"]
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes,
                   "class_index": {c: i for i, c in enumerate(classes)}}, f)

    train = []
    for i in range(60):
        train.append([f"/dev/null/{i}.wav", "clink_wine_thin", f"sess_t_{i//10}",
                      32000, 1.0, 1, "project"])
    for i in range(120):
        train.append([f"/dev/null/n{i}.wav", "_negative", f"sess_n_{i//10}",
                      32000, 1.0, 0, "project"])
    val = [
        [f"/dev/null/v{i}.wav", "clink_wine_thin", f"sess_vt_{i}",
         32000, 1.0, 1, "project"] for i in range(10)
    ] + [
        [f"/dev/null/vn{i}.wav", "_negative", f"sess_vn_{i}",
         32000, 1.0, 0, "project"] for i in range(20)
    ]
    test = [
        [f"/dev/null/te{i}.wav", "clink_wine_thin", f"sess_tet_{i}",
         32000, 1.0, 1, "project"] for i in range(10)
    ] + [
        [f"/dev/null/ten{i}.wav", "_negative", f"sess_ten_{i}",
         32000, 1.0, 0, "project"] for i in range(20)
    ]
    _write_csv(layout.dataset_dir / "train.csv", train)
    _write_csv(layout.dataset_dir / "val.csv", val)
    _write_csv(layout.dataset_dir / "test.csv", test)

    cfg = load_config()
    cfg.model.classes = classes
    checks = doctor.doctor_train(cfg, "p")
    fails = [c for c in checks if c.status == "FAIL"]
    assert not fails, f"expected no FAILs, got: {[(c.name, c.detail) for c in fails]}"


def test_doctor_train_does_not_check_session_leakage_for_random_splits(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("leaky")
    layout.create()
    classes = ["_negative", "x"]
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_index": {c: i for i, c in enumerate(classes)}}, f)
    # Same session_id in train AND val is allowed now: splits are random
    # clip-level per class and session_id is retained for traceability only.
    _write_csv(layout.dataset_dir / "train.csv",
               [[f"/dev/null/{i}.wav", "x", "shared_sess",
                 32000, 1.0, 1, "p"] for i in range(60)] +
               [[f"/dev/null/n{i}.wav", "_negative", f"sn_{i}",
                 32000, 1.0, 0, "p"] for i in range(120)])
    _write_csv(layout.dataset_dir / "val.csv",
               [[f"/dev/null/v{i}.wav", "x", "shared_sess",
                 32000, 1.0, 1, "p"] for i in range(10)])
    _write_csv(layout.dataset_dir / "test.csv",
               [[f"/dev/null/te{i}.wav", "x", "fresh",
                 32000, 1.0, 1, "p"] for i in range(10)])
    cfg = load_config()
    cfg.model.classes = classes
    checks = doctor.doctor_train(cfg, "leaky")
    leakage = [c for c in checks if "leakage" in c.name]
    assert not leakage


def test_doctor_train_warns_on_empty_negative_pool(tmp_path, monkeypatch):
    """The negative/background pool is expected for FP-resistant training."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    _patch_library_paths(monkeypatch, tmp_path)
    # Deliberately do NOT seed negatives.
    layout = ProjectLayout("p")
    layout.create()
    classes = ["_negative", "clink"]
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes,
                   "class_index": {c: i for i, c in enumerate(classes)}}, f)
    for split in ("train", "val", "test"):
        _write_csv(layout.dataset_dir / f"{split}.csv",
                   [[f"/dev/null/{split}{i}.wav", "clink", f"s_{split}_{i}",
                     32000, 1.0, 1, "p"] for i in range(20)] +
                   [[f"/dev/null/{split}n{i}.wav", "_negative", f"sn_{split}_{i}",
                     32000, 1.0, 0, "p"] for i in range(60)])
    cfg = load_config()
    cfg.model.classes = classes
    checks = doctor.doctor_train(cfg, "p")
    warns = [c for c in checks if c.status == "WARN"]
    neg_warns = [c for c in warns if c.name == "negatives"]
    assert neg_warns, \
        f"expected negatives WARN, got: {[(c.name, c.status) for c in checks]}"
    fix_text = neg_warns[0].fix
    assert "assign labels to _negative" in fix_text


def test_doctor_train_warns_thin_negatives(tmp_path, monkeypatch):
    """Less than 20 minutes total negative/background material should warn."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    _patch_library_paths(monkeypatch, tmp_path)
    _seed_negatives(tmp_path, n_clips=8, total_min=8.0,
                    labels=("bar_one", "bar_two", "bar_three"))
    layout = ProjectLayout("thin")
    layout.create()
    classes = ["_negative", "x"]
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes,
                   "class_index": {c: i for i, c in enumerate(classes)}}, f)
    for split in ("train", "val", "test"):
        _write_csv(layout.dataset_dir / f"{split}.csv",
                   [[f"/dev/null/{split}{i}.wav", "x", f"s_{split}_{i}",
                     32000, 1.0, 1, "p"] for i in range(20)] +
                   [[f"/dev/null/{split}n{i}.wav", "_negative", f"sn_{split}_{i}",
                     32000, 1.0, 0, "p"] for i in range(60)])
    cfg = load_config()
    cfg.model.classes = classes
    checks = doctor.doctor_train(cfg, "thin")
    # A thin pool surfaces in the consolidated "negatives" row, with the
    # duration figure quoted in the detail string.
    neg_warns = [c for c in checks
                 if c.name == "negatives" and c.status == "WARN"]
    assert neg_warns, \
        f"expected negatives WARN, got: {[(c.name, c.status, c.detail) for c in checks]}"
    assert "min" in neg_warns[0].detail


def test_doctor_train_skips_negative_pool_when_negative_class_absent(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    _patch_library_paths(monkeypatch, tmp_path)
    _seed_negatives(tmp_path, n_clips=2, total_min=1.0, labels=("cutlery",))
    layout = ProjectLayout("target_only")
    layout.create()
    classes = ["cheers_loud", "cheers_quiet"]
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_index": {c: i for i, c in enumerate(classes)}}, f)
    for split in ("train", "val", "test"):
        _write_csv(layout.dataset_dir / f"{split}.csv",
                   [[f"/dev/null/{split}a{i}.wav", "cheers_loud", f"sa_{split}_{i}",
                     32000, 1.0, 1, "p"] for i in range(60)] +
                   [[f"/dev/null/{split}b{i}.wav", "cheers_quiet", f"sb_{split}_{i}",
                     32000, 1.0, 1, "p"] for i in range(60)])
    cfg = load_config()
    cfg.model.classes = classes

    checks = doctor.doctor_train(cfg, "target_only")

    assert not [c for c in checks if c.name == "negatives"]
