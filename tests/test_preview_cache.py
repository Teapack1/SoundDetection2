"""Signature-aware, shared inspect preview cache tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from sed.data.preview_cache import (
    CACHE_LAYOUT_VERSION,
    PreviewCache,
    PreviewWorkerPool,
    _display_pixels,
    preview_signature,
)


def _features(*, n_mels: int = 64) -> dict:
    return {
        "sample_rate": 16000,
        "frontend": "efficientat",
        "feature_type": "mel",
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": n_mels,
        "fmin": 0,
        "fmax": 7000,
        "log_power": True,
        "pcen": False,
        "pcen_params": None,
        "layout": CACHE_LAYOUT_VERSION,
        "palette_colors": 128,
    }


def _wav(path: Path, samples: int = 16000) -> Path:
    wave = (0.1 * np.sin(np.linspace(0, 440, samples))).astype(np.float32)
    sf.write(path, wave, 16000)
    return path


def test_variants_share_paths_overwrite_changed_source_and_reclaim_only_last_project(tmp_path):
    wav = _wav(tmp_path / "clip.wav")
    cache = PreviewCache(root=tmp_path / "library")
    first_features = _features()
    second_features = _features(n_mels=80)

    first = cache.ensure(wav, first_features, project_id="one", label="first")
    same = cache.lookup(wav, first_features, project_id="two")
    assert same is not None and same.path == first.path and same.created is False

    second = cache.ensure(wav, second_features, project_id="one", label="second")
    assert first.path.exists() and second.path.exists()
    assert first.path != second.path

    # A source update atomically replaces the one requested signature.  The
    # preserved comparison variant remains on disk until explicitly refreshed.
    original_second = second.path.read_bytes()
    _wav(wav, samples=24000)
    refreshed = cache.ensure(wav, first_features, project_id="one", label="first")
    assert refreshed.created is True
    assert first.path.exists()
    assert second.path.read_bytes() == original_second

    # Project associations share identical source/signature assets.  Removing
    # one project cannot reclaim the entry while another still references it.
    shared_path = first.path
    one = cache.delete_project_variant("one", preview_signature(first_features))
    assert one["removed"] >= 1
    assert shared_path.exists()
    two = cache.delete_project_variant("two", preview_signature(first_features))
    assert two["files"] == 1
    assert not shared_path.exists()


def test_migrates_valid_legacy_thumb_and_worker_pool_deduplicates(tmp_path):
    wav = _wav(tmp_path / "clip.wav")
    legacy = wav.with_suffix(".spec.png")
    Image.new("RGB", (32, 16), "purple").save(legacy)
    cache = PreviewCache(root=tmp_path / "library")
    features = _features()

    migrated = cache.migrate_legacy(wav, features, project_id="p", label="legacy")
    assert migrated is not None and migrated.path.exists()
    assert not legacy.exists()
    with Image.open(migrated.path) as image:
        assert image.mode == "P"

    other = _wav(tmp_path / "other.wav")
    workers = PreviewWorkerPool(cache, workers=2)
    first = workers.submit(other, features, project_id="p", priority=10)
    promoted = workers.submit(other, features, project_id="p", priority=0)
    assert first is promoted
    assert first.result(timeout=30).path.exists()


def test_preview_cache_display_pixels_put_low_frequency_at_bottom():
    spec = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    pixels = _display_pixels(spec)

    assert np.array_equal(pixels[0], spec[-1])
    assert np.array_equal(pixels[-1], spec[0])


def test_preview_cache_layout_signature_bumps_for_orientation_fix():
    current = _features()
    old = dict(current)
    old["layout"] = "palette-png-v1"

    assert CACHE_LAYOUT_VERSION != "palette-png-v1"
    assert preview_signature(current) != preview_signature(old)
