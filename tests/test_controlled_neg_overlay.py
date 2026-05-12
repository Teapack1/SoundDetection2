import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sed.data.augment import AudioPool, materialize_augmented
from sed.config import Config, ProjectLayout


@pytest.fixture(autouse=True)
def _restore_library_dir_module_attr():
    """_import_library_clip assigns sed.data.library.LIBRARY_DIR directly (not
    via monkeypatch); restore it so the leak never bleeds into other files."""
    from sed.data import library as lib_mod

    original = lib_mod.LIBRARY_DIR
    yield
    lib_mod.LIBRARY_DIR = original


def _noise(n: int, amp: float = 0.1) -> np.ndarray:
    return amp * np.random.RandomState(0).randn(n).astype(np.float32)


def _import_library_clip(path: Path, label: str, *, project_id: str):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    from sed.data.library import Library

    lib_mod.LIBRARY_DIR = cfg_mod.LIBRARY_DIR
    return Library().import_clip(path, label=label, project_id=project_id)


def _derived_dir(tmp_path: Path, project_id: str, class_name: str) -> Path:
    return tmp_path / "data_library" / "derived" / project_id / "trainval" / class_name

def test_audiopool_excluded():
    # Setup dummy paths representing different nested folder structures
    roots = ["/tmp/dummy_pool"]
    pool = AudioPool(roots, sample_rate=32000)
    
    # Manually populate paths for testing AudioPool.excluded
    pool.paths = [
        Path("/tmp/dummy_pool/ambience/room_a.wav"),
        Path("/tmp/dummy_pool/ambience/room_b.wav"),
        Path("/tmp/dummy_pool/cutlery/spoon.wav"),
        Path("/tmp/dummy_pool/speech/talk.wav"),
    ]
    
    # Exclude "ambience" label
    ex_pool = pool.excluded(["ambience"])
    assert len(ex_pool.paths) == 2
    assert Path("/tmp/dummy_pool/cutlery/spoon.wav") in ex_pool.paths
    assert Path("/tmp/dummy_pool/speech/talk.wav") in ex_pool.paths
    assert Path("/tmp/dummy_pool/ambience/room_a.wav") not in ex_pool.paths
    
    # Exclude both "ambience" and "cutlery"
    ex_pool_multiple = pool.excluded(["ambience", "cutlery"])
    assert len(ex_pool_multiple.paths) == 1
    assert ex_pool_multiple.paths == [Path("/tmp/dummy_pool/speech/talk.wav")]

def test_controlled_negative_overlay(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    project_id = "test_proj"
    layout = ProjectLayout(project_id)
    layout.create()
    sr = 32000

    # Create negatives matching the layout
    ambience_dir = layout.negatives_dir("ambience")
    cutlery_dir = layout.negatives_dir("cutlery")
    ambience_dir.mkdir(parents=True, exist_ok=True)
    cutlery_dir.mkdir(parents=True, exist_ok=True)

    ambient_a = ambience_dir / "ambient_a.wav"
    ambient_b = ambience_dir / "ambient_b.wav"
    fork = cutlery_dir / "fork_clink.wav"
    sf.write(str(ambient_a), _noise(sr, amp=0.01), sr)
    sf.write(str(ambient_b), _noise(sr, amp=0.01), sr)
    sf.write(str(fork), _noise(sr, amp=0.05), sr)
    _import_library_clip(ambient_a, "ambience", project_id=project_id)
    _import_library_clip(ambient_b, "ambience", project_id=project_id)
    _import_library_clip(fork, "cutlery", project_id=project_id)

    cfg = Config()
    cfg.model.classes = ["_negative"]
    cfg.augmentation.enabled = True
    cfg.augmentation.overlay_enabled = True
    cfg.augmentation.samples_per_source = 2
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.generate_positive_overlay = False
    cfg.augmentation.generate_negative_overlay = True
    cfg.augmentation.negative_overlay_base_labels = ["ambience"]
    cfg.dataset.project = {
        "_negative": [{"category": "negatives", "labels": ["ambience", "cutlery"]}]
    }

    counts = materialize_augmented(cfg, project_id, seed=42)

    # 2 ambient base clips * 2 samples per source = 4 negative overlays
    assert counts["_negative"]["overlay"] == 4
    
    out_dir = _derived_dir(tmp_path, project_id, "_negative")
    sidecars = sorted(out_dir.glob("*_aug_*.json"))
    assert len(sidecars) == 4

    for sidecar in sidecars:
        meta = json.loads(sidecar.read_text())
        assert meta["augmentation"] == "overlay"
        params = meta["augmentation_params"]
        assert params["base_role"] == "negative"
        
        # Base (source) must be one of the ambient base clips
        assert params["source"] in {"ambient_a.wav", "ambient_b.wav"}
        # Overlay (negative_source) must be the cutlery clip, NOT an ambient clip
        assert params["negative_source"] == "fork_clink.wav"

def test_controlled_negative_overlay_empty_fallback(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    project_id = "test_proj_fallback"
    layout = ProjectLayout(project_id)
    layout.create()
    sr = 32000

    ambience_dir = layout.negatives_dir("ambience")
    cutlery_dir = layout.negatives_dir("cutlery")
    ambience_dir.mkdir(parents=True, exist_ok=True)
    cutlery_dir.mkdir(parents=True, exist_ok=True)

    ambient_a = ambience_dir / "ambient_a.wav"
    fork = cutlery_dir / "fork_clink.wav"
    sf.write(str(ambient_a), _noise(sr, amp=0.01), sr)
    sf.write(str(fork), _noise(sr, amp=0.05), sr)
    _import_library_clip(ambient_a, "ambience", project_id=project_id)
    _import_library_clip(fork, "cutlery", project_id=project_id)

    cfg = Config()
    cfg.model.classes = ["_negative"]
    cfg.augmentation.enabled = True
    cfg.augmentation.overlay_enabled = True
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.generate_positive_overlay = False
    cfg.augmentation.generate_negative_overlay = True
    cfg.augmentation.negative_overlay_base_labels = [] # Empty = fallback to mixing all
    cfg.dataset.project = {
        "_negative": [{"category": "negatives", "labels": ["ambience", "cutlery"]}]
    }

    counts = materialize_augmented(cfg, project_id, seed=42)

    # 2 total files * 1 sample per source = 2 negative overlays
    assert counts["_negative"]["overlay"] == 2


def test_materialize_augmented_respects_negative_label_filters(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    project_id = "test_proj_filtered"
    layout = ProjectLayout(project_id)
    layout.create()
    sr = 32000

    ambience_dir = layout.negatives_dir("ambience")
    cutlery_dir = layout.negatives_dir("cutlery")
    ambience_dir.mkdir(parents=True, exist_ok=True)
    cutlery_dir.mkdir(parents=True, exist_ok=True)

    ambient_a = ambience_dir / "ambient_a.wav"
    fork = cutlery_dir / "fork_clink.wav"
    sf.write(str(ambient_a), _noise(sr, amp=0.01), sr)
    sf.write(str(fork), _noise(sr, amp=0.05), sr)
    _import_library_clip(ambient_a, "ambience", project_id=project_id)
    _import_library_clip(fork, "cutlery", project_id=project_id)

    cfg = Config()
    cfg.model.classes = ["_negative"]
    cfg.augmentation.enabled = True
    cfg.augmentation.overlay_enabled = True
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.generate_positive_overlay = False
    cfg.augmentation.generate_negative_overlay = True
    cfg.augmentation.negative_overlay_base_labels = [] 

    # Restrict negative dataset to only 'ambience'
    cfg.dataset.project = {
        "_negative": [{"category": "negatives", "labels": ["ambience"]}]
    }

    counts = materialize_augmented(cfg, project_id, seed=42)

    # 1 ambient file in pool * 1 sample per source = 1 overlay generated
    assert counts["_negative"]["overlay"] == 1

    out_dir = _derived_dir(tmp_path, project_id, "_negative")
    sidecars = sorted(out_dir.glob("*_aug_*.json"))
    assert len(sidecars) == 1

    meta = json.loads(sidecars[0].read_text())
    params = meta["augmentation_params"]
    assert params["source"] == "ambient_a.wav"
    # Since 'cutlery' (fork_clink.wav) was excluded from the pool, negative_source cannot be fork_clink.wav
    assert params["negative_source"] != "fork_clink.wav"
