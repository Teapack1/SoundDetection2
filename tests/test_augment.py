"""Tests for the augmentation pipeline — especially the masking-aware mix."""

import json
import random

import numpy as np
import pytest
import soundfile as sf

from sed.data.augment import (
    _ensure_length,
    _mix_with_scaled_noise,
    masking_score_db,
    masking_score_details,
    mix_at_snr,
)


def _impulse(n: int, position: int = 100, amp: float = 0.5) -> np.ndarray:
    """A short transient burst at `position`, decaying like a clink."""
    x = np.zeros(n, dtype=np.float32)
    decay_n = 800
    end = min(n, position + decay_n)
    decay = np.exp(-np.arange(end - position) / 200.0)
    x[position:end] = amp * decay * np.sin(
        2 * np.pi * 4000 * np.arange(end - position) / 32000
    )
    return x


def _noise(n: int, amp: float = 0.1) -> np.ndarray:
    return amp * np.random.RandomState(0).randn(n).astype(np.float32)


def _soft_cling(
    n: int = 32000,
    position: int = 12000,
    amp: float = 0.04,
    attack_ms: int = 50,
    decay_ms: int = 250,
    freq: float = 4200.0,
) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    dur = int((attack_ms + decay_ms) / 1000.0 * 32000)
    attack = max(1, int(attack_ms / 1000.0 * 32000))
    end = min(n, position + dur)
    env = np.ones(dur, dtype=np.float32)
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    env[attack:] = np.exp(
        -np.arange(dur - attack, dtype=np.float32) / (decay_ms / 1000.0 * 32000 / 3.0)
    )
    k = np.arange(end - position, dtype=np.float32)
    x[position:end] = amp * env[: end - position] * np.sin(2 * np.pi * freq * k / 32000)
    return x


def _tone_segment(
    n: int = 32000,
    start: int = 12000,
    duration: int = 1600,
    freq: float = 4200.0,
    amp: float = 0.05,
) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    end = min(n, start + duration)
    k = np.arange(end - start, dtype=np.float32)
    x[start:end] = amp * np.sin(2 * np.pi * freq * k / 32000)
    return x


def _import_library_clip(
    path,
    label: str,
    *,
    project_id: str = "p",
    dataset_role: str = "train",
):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    from sed.data.library import Library

    lib_mod.LIBRARY_DIR = cfg_mod.LIBRARY_DIR
    return Library().import_clip(
        path,
        label=label,
        project_id=project_id,
        dataset_role=dataset_role,
    )


def _derived_dir(tmp_path, project_id: str, role: str, class_name: str):
    return tmp_path / "data_library" / "derived" / project_id / role / class_name


def test_library_filter_paths_respect_triggered_microphone_set(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    from sed.data.augment import _library_paths_for_filters
    from sed.data.library import Library

    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    mic0_src = tmp_path / "glass_mic0.wav"
    mic2_src = tmp_path / "glass_mic2.wav"
    sf.write(str(mic0_src), np.zeros(32000, dtype=np.float32), 32000)
    sf.write(str(mic2_src), np.zeros(32000, dtype=np.float32), 32000)

    lib = Library()
    mic0 = lib.import_clip(mic0_src, label="glass", project_id="p")
    mic2 = lib.import_clip(mic2_src, label="glass", project_id="p")
    for clip, channels in ((mic0, [0]), (mic2, [2])):
        wav = clip.absolute_path(lib.root)
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        meta["recorded_channels"] = channels
        meta["triggered_channels"] = channels
        wav.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    lib.rebuild_manifest()

    paths = _library_paths_for_filters(
        lib,
        "glass",
        dataset_role="train",
        project_id="p",
        frozen_test=False,
        filters=[{"labels": ["glass"], "project_ids": ["p"], "triggered_channels": [2]}],
    )

    assert len(paths) == 1
    assert paths[0][0].name == mic2.absolute_path(lib.root).name


def test_mix_at_snr_keeps_target_audible_at_high_snr():
    sig = _impulse(32000)
    noise = _noise(32000, amp=0.1)
    mixed = mix_at_snr(sig, noise, snr_db=20.0)
    assert mixed.shape == sig.shape
    # at +20 dB SNR the mix should be very close to the signal in peak
    assert abs(mixed.max() - sig.max()) < 0.2


def test_masking_score_high_when_target_dominant():
    """When the target is much louder than the noise, its transient should
    dominate the mix → high transient-dominance dB."""
    sig = _impulse(32000, amp=0.5)
    noise = _noise(32000, amp=0.05)
    mixed = mix_at_snr(sig, noise, snr_db=20.0)
    score = masking_score_db(sig, mixed, sr=32000, background=mixed - sig)
    # the 4 kHz transient should poke well above noise floor at its peak frame
    assert score > 6.0, f"expected target to dominate, got {score:.2f} dB"


def test_masking_score_decreases_with_lower_snr():
    """The transient-dominance score should drop monotonically as the SNR
    drops — a clean target dominates much more than the same target buried
    under heavy in-band noise."""
    sig = _impulse(32000, amp=0.1, position=100)
    rng = np.random.RandomState(0)
    # Co-band tonal noise (simulates speech formants overlapping the clink band)
    t = np.arange(32000) / 32000.0
    co_band = (np.sin(2*np.pi*3800*t) + np.sin(2*np.pi*4200*t)
               + 0.3 * rng.randn(32000)).astype(np.float32)

    scores = []
    for snr_db in (30.0, 10.0, -5.0, -20.0):
        mixed = mix_at_snr(sig, co_band, snr_db=snr_db)
        scores.append(masking_score_db(sig, mixed, sr=32000, background=mixed - sig))

    # Each subsequent (lower) SNR should yield ≤ the previous score (within
    # small tolerance for noise variance)
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i-1] + 1.0, \
            f"non-monotonic: SNR step {i} score {scores[i]:.2f} > prev {scores[i-1]:.2f}"
    # And the dynamic range should be substantial — at least a few dB
    # between best and worst case
    assert scores[0] - scores[-1] > 3.0, \
        f"metric not responsive enough: range={scores[0]-scores[-1]:.2f} dB"


def test_transient_band_selection_ignores_steady_low_hum():
    sr = 32000
    t = np.arange(sr) / sr
    hum = 0.35 * np.sin(2 * np.pi * 90 * t).astype(np.float32)
    sig = hum + _soft_cling(position=12000, amp=0.04, attack_ms=50)

    details = masking_score_details(
        sig,
        sig,
        sr=sr,
        background=np.zeros_like(sig),
        transient_window_ms=100,
    )
    freqs = details["mask_selected_band_hz"]

    assert sum(3000.0 <= f <= 5500.0 for f in freqs) >= 6
    assert sum(f < 500.0 for f in freqs) <= 2


def test_masking_score_is_band_specific():
    sig = _impulse(32000, position=12000, amp=0.08)
    t = np.arange(32000) / 32000.0
    low_rumble = (0.8 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)
    in_band = (0.8 * np.sin(2 * np.pi * 4050 * t)).astype(np.float32)

    low_mix = sig + low_rumble
    band_mix = sig + in_band

    low_score = masking_score_db(sig, low_mix, sr=32000, background=low_rumble)
    band_score = masking_score_db(sig, band_mix, sr=32000, background=in_band)

    assert low_score > band_score + 6.0


def test_masking_top_frames_emphasize_50ms_attack():
    sr = 32000
    pos = 12000
    target = _tone_segment(sr, start=pos, duration=int(0.050 * sr), amp=0.06)
    background = _tone_segment(
        sr,
        start=pos + int(0.0625 * sr),
        duration=int(0.050 * sr),
        amp=0.06,
    )

    top5 = masking_score_details(
        target,
        target + background,
        sr=sr,
        background=background,
        transient_window_ms=100,
        mask_top_frames=5,
    )
    full10 = masking_score_details(
        target,
        target + background,
        sr=sr,
        background=background,
        transient_window_ms=100,
        mask_top_frames=10,
    )

    assert top5["mask_top_frames"] == 5
    assert top5["masking_score_p25_db"] > full10["masking_score_p25_db"] + 3.0


def test_masking_metadata_shows_same_background_bands_are_compared():
    sig = _soft_cling(position=12000, amp=0.05)
    background = _tone_segment(freq=4200.0, amp=0.04)

    details = masking_score_details(
        sig,
        sig + background,
        sr=32000,
        background=background,
        transient_window_ms=100,
    )

    assert details["mask_compared_background_bands"] == details["mask_selected_bands"]


def test_overlay_mix_records_transient_aligned_reference():
    sig = _impulse(32000, position=15000, amp=0.08)
    noise = _noise(32000, amp=0.1)

    mixed, _, _, params = _mix_with_scaled_noise(
        sig,
        noise,
        snr_db=10.0,
        transient_window_ms=100,
        sample_rate=32000,
        prevent_clipping=False,
    )

    assert mixed.shape == sig.shape
    assert params["snr_reference"] == "transient_window_aligned"
    assert 0 <= 15000 - params["transient_sample"] < 3200
    assert params["transient_window_frames"] == 10


def test_augmentation_defaults_use_100ms_transient_windows():
    from sed.config import Config

    cfg = Config()

    assert cfg.augmentation.transient_window_ms == 100
    assert cfg.augmentation.overlay_enabled is True
    assert cfg.augmentation.mask_filter_enabled is True
    assert cfg.augmentation.mask_candidate_attempts == 8
    assert cfg.augmentation.mask_min_mean_db == 6.0
    assert cfg.augmentation.mask_min_p25_db == 3.0
    assert cfg.augmentation.mask_min_band_db == -3.0
    assert cfg.augmentation.mask_band_count == 10
    assert cfg.augmentation.mask_top_frames == 5


def test_masking_details_include_window_and_band_diagnostics():
    sr = 32000
    sig = _soft_cling(position=12000, amp=0.05)
    details = masking_score_details(
        sig,
        sig,
        sr=sr,
        background=np.zeros_like(sig),
        transient_window_ms=100,
    )

    assert details["mask_window_frames"] == 10
    assert isinstance(details["mask_transient_sample"], int)
    assert len(details["mask_selected_bands"]) == 10
    assert len(details["mask_selected_band_hz"]) == 10
    assert len(details["mask_band_margins_db"]) == 10
    assert details["mask_band_count"] == 10
    assert details["mask_compared_background_bands"] == details["mask_selected_bands"]
    assert details["mask_top_frames"] == 5
    assert "masking_score_p25_db" in details
    assert "masking_score_min_db" in details


def test_mask_band_count_controls_number_of_compared_mel_bands():
    sr = 32000
    sig = _soft_cling(position=12000, amp=0.05)
    details = masking_score_details(
        sig,
        sig,
        sr=sr,
        background=np.zeros_like(sig),
        transient_window_ms=100,
        mask_band_count=6,
        mask_top_frames=5,
    )

    assert details["mask_band_count"] == 6
    assert len(details["mask_selected_bands"]) == 6
    assert len(details["mask_compared_background_bands"]) == 6
    assert len(details["mask_band_margins_db"]) == 6


def test_overlay_per_class_min_snr_and_negative_label_filter(
    tmp_path, monkeypatch
):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data import library as lib_mod
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    project_id = "p"
    layout = ProjectLayout(project_id)
    layout.create()

    sr = 32000
    positive_dir = layout.positives_dir("cheers_quiet")
    negative_dir = layout.negatives_dir("cutlery")
    other_negative_dir = layout.negatives_dir("speech")
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    other_negative_dir.mkdir(parents=True, exist_ok=True)
    quiet = positive_dir / "quiet.wav"
    spoon = negative_dir / "spoon.wav"
    talk = other_negative_dir / "talk.wav"
    sf.write(str(quiet), _impulse(sr, amp=0.08), sr)
    sf.write(str(spoon), _noise(sr, amp=0.1), sr)
    sf.write(str(talk), _noise(sr, amp=0.2), sr)
    _import_library_clip(quiet, "cheers_quiet", project_id=project_id)
    _import_library_clip(spoon, "cutlery", project_id=project_id)
    _import_library_clip(talk, "speech", project_id=project_id)

    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_quiet"]
    cfg.augmentation.enabled = True
    cfg.augmentation.overlay_enabled = True
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False
    cfg.augmentation.samples_per_source = 2
    cfg.augmentation.snr_db_range = (0.0, 5.0)
    cfg.augmentation.per_class_min_snr_db = {"cheers_quiet": 10.0}
    cfg.augmentation.per_class_negative_label = {
        "cheers_quiet": ["cutlery"]
    }
    cfg.augmentation.generate_positive_overlay = True
    cfg.augmentation.generate_negative_overlay = False
    cfg.dataset.project = {
        "_negative": [{"category": "negatives", "labels": ["cutlery", "speech"]}]
    }

    counts = materialize_augmented(cfg, project_id, seed=123)

    assert counts["cheers_quiet"]["overlay"] == 2

    out_dir = _derived_dir(tmp_path, project_id, "trainval", "cheers_quiet")
    sidecars = sorted(out_dir.glob("*_aug_*.json"))
    assert len(sidecars) == 2
    for sidecar_path in sidecars:
        meta = json.loads(sidecar_path.read_text())
        assert meta["augmentation"] == "overlay"
        assert meta["augmentation_params"]["snr_db"] >= 10.0
        assert meta["augmentation_params"]["negative_source"] == "spoon.wav"
        assert meta["augmentation_params"]["base_role"] == "positive"
        assert meta["augmentation_params"]["pitch_shift_enabled"] is False
        assert meta["mask_window_frames"] == 10
        assert len(meta["mask_selected_bands"]) == 10
        assert meta["mask_band_count"] == 10
        assert len(meta["mask_selected_band_hz"]) == 10
        assert len(meta["mask_band_margins_db"]) == 10


def test_target_based_augmentation_labels_output_and_metadata(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "quiet.wav"
    overlay = tmp_path / "spoon.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.02), sr)
    _import_library_clip(base, "quiet_take_a", project_id="p")
    _import_library_clip(overlay, "cutlery", project_id="p")

    cfg = Config.model_validate({
        "dataset": {"target_classes": ["_negative", "cheers_quiet"]},
        "augmentation": {
            "combinations": {
                "cheers_quiet": [
                    {
                        "base": "quiet_take_a",
                        "overlay": "cutlery",
                        "recipe_id": "recipe-quiet-cutlery",
                        "description": "quiet take with cutlery foreground risk",
                        "tags": ["cutlery", "quiet-review"],
                        "mask_min_mean_db": 10.0,
                    }
                ]
            }
        },
    })
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False

    counts = materialize_augmented(cfg, "p", seed=123, run_prefix="targeted")

    assert counts["cheers_quiet"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "cheers_quiet").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    params = meta["augmentation_params"]
    assert meta["class_name"] == "cheers_quiet"
    assert meta["target_class"] == "cheers_quiet"
    assert meta["recipe_id"] == "recipe-quiet-cutlery"
    assert meta["augmentation_settings"]["samples_per_source"] == 1.0
    assert meta["description"] == "quiet take with cutlery foreground risk"
    assert meta["tags"] == ["cutlery", "quiet-review"]
    assert params["base_label"] == "quiet_take_a"
    assert params["overlay_label"] == "cutlery"
    assert params["actual_base_label"] == "quiet_take_a"
    assert params["actual_overlay_label"] == "cutlery"
    assert params["combination_description"] == "quiet take with cutlery foreground risk"
    assert params["combination_tags"] == ["cutlery", "quiet-review"]
    assert params["recipe_id"] == "recipe-quiet-cutlery"
    assert params["mask_min_mean_db"] == 10.0
    assert params["target_class"] == "cheers_quiet"


def test_target_based_augmentation_uses_ui_bucket_filters(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base_keep = tmp_path / "base_keep.wav"
    base_drop = tmp_path / "base_drop.wav"
    overlay_keep = tmp_path / "overlay_keep.wav"
    overlay_drop = tmp_path / "overlay_drop.wav"
    sf.write(str(base_keep), _impulse(sr, amp=0.08), sr)
    sf.write(str(base_drop), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay_keep), _noise(sr, amp=0.02), sr)
    sf.write(str(overlay_drop), _noise(sr, amp=0.02), sr)
    _import_library_clip(base_keep, "base_keep", project_id="p")
    _import_library_clip(base_drop, "base_keep", project_id="_library")
    _import_library_clip(overlay_keep, "overlay_keep", project_id="p")
    _import_library_clip(overlay_drop, "overlay_keep", project_id="_library")

    cfg = Config.model_validate({
        "dataset": {"target_classes": ["cheers_quiet"]},
        "augmentation": {
            "combinations": {
                "cheers_quiet": [
                    {
                        "base": "*",
                        "overlay": "*",
                        "base_filters": [
                            {"labels": ["base_keep"], "project_ids": ["p"]},
                            {"labels": ["base_keep"], "project_ids": ["_library"]},
                        ],
                        "overlay_filters": [
                            {"labels": ["overlay_keep"], "project_ids": ["p"]},
                        ],
                        "mask_min_p25_db": 2.5,
                        "mask_min_band_db": -2.0,
                    }
                ]
            }
        },
    })
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False

    counts = materialize_augmented(cfg, "p", seed=123, run_prefix="filtered")

    assert counts["cheers_quiet"]["overlay"] == 2
    sidecars = sorted(_derived_dir(tmp_path, "p", "trainval", "cheers_quiet").glob("*.json"))
    assert len(sidecars) == 2
    sidecar = sidecars[0]
    params = json.loads(sidecar.read_text())["augmentation_params"]
    assert params["actual_base_label"] == "base_keep"
    assert params["actual_overlay_label"] == "overlay_keep"
    assert params["mask_min_p25_db"] == 2.5
    assert params["mask_min_band_db"] == -2.0


def test_library_recipe_augmentation_saves_under_base_label_and_replace_is_recipe_scoped(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_library_recipes

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "ikea.wav"
    overlay = tmp_path / "cutlery.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.02), sr)
    _import_library_clip(base, "ikea_wine", project_id="p")
    _import_library_clip(overlay, "cutlery", project_id="p")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False
    recipe = {
        "id": "recipe-a",
        "recipe_id": "recipe-a",
        "base_filters": [{"labels": ["ikea_wine"], "project_ids": ["p"]}],
        "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
        "description": "ikea with cutlery",
        "tags": ["ui-recipe"],
    }

    counts = materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="ui")

    assert counts["ikea_wine"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "ikea_wine").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["label"] == "ikea_wine"
    assert meta["class_name"] == "ikea_wine"
    assert "target_class" not in meta
    assert meta["base_label"] == "ikea_wine"
    assert meta["overlay_label"] == "cutlery"
    assert meta["recipe_id"] == "recipe-a"
    assert meta["recipe_signature"]
    assert meta["generation_id"]
    assert meta["augmentation_settings"]["samples_per_source"] == 1.0
    assert meta["augmentation_params"]["base_source"]
    assert meta["augmentation_params"]["overlay_source"]

    materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="ui", replace=True)
    assert len(list(_derived_dir(tmp_path, "p", "trainval", "ikea_wine").glob("*.wav"))) == 1


def test_library_recipe_inherits_base_density_and_tags(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data import library as lib_mod
    from sed.data.augment import materialize_library_recipes
    from sed.data.library import Library

    lib_mod.LIBRARY_DIR = cfg_mod.LIBRARY_DIR
    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "ikea.wav"
    overlay = tmp_path / "cutlery.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.02), sr)
    Library().import_clip(
        base, label="ikea_wine", project_id="p", density="empty", tags=["wine", "loud"],
    )
    Library().import_clip(overlay, label="cutlery", project_id="p")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False
    recipe = {
        "id": "recipe-meta",
        "recipe_id": "recipe-meta",
        "base_filters": [{"labels": ["ikea_wine"], "project_ids": ["p"]}],
        "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
        "tags": ["ui-recipe", "wine"],
    }

    counts = materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="ui")

    assert counts["ikea_wine"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "ikea_wine").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    # Density inherited from the base clip; tags = recipe tags + base tags, deduped.
    assert meta["density"] == "empty"
    assert meta["tags"] == ["ui-recipe", "wine", "loud"]


def test_effective_snr_gate_rejects_overlay_dominated_candidates(tmp_path, monkeypatch):
    """mask_min_effective_snr_db is a broadband floor: with an impossible floor
    every candidate is rejected (no fallback — the gate is explicit); with a
    permissive floor generation proceeds."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_library_recipes

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "glass.wav"
    overlay = tmp_path / "cutlery.wav"
    sf.write(str(base), _impulse(sr, amp=0.05), sr)
    sf.write(str(overlay), _noise(sr, amp=0.2), sr)  # overlay much louder broadband
    _import_library_clip(base, "glass", project_id="p")
    _import_library_clip(overlay, "cutlery", project_id="p")

    def make_cfg(floor):
        cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})
        cfg.augmentation.samples_per_source = 1
        cfg.augmentation.pitch_shift_enabled = False
        cfg.augmentation.scale_noise_to_snr = False
        cfg.augmentation.mask_filter_enabled = True
        cfg.augmentation.mask_candidate_attempts = 2
        cfg.augmentation.mask_min_mean_db = None
        cfg.augmentation.mask_min_p25_db = None
        cfg.augmentation.mask_min_band_db = None
        cfg.augmentation.mask_min_effective_snr_db = floor
        return cfg

    recipe = {
        "id": "recipe-snr",
        "recipe_id": "recipe-snr",
        "base_filters": [{"labels": ["glass"], "project_ids": ["p"]}],
        "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
    }

    counts = materialize_library_recipes(make_cfg(40.0), "p", [recipe], seed=123, run_prefix="snr")
    assert counts == {}
    assert not list(_derived_dir(tmp_path, "p", "trainval", "glass").glob("*.wav"))

    counts = materialize_library_recipes(make_cfg(-60.0), "p", [recipe], seed=123, run_prefix="snr")
    assert counts["glass"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "glass").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["augmentation_params"]["mask_min_effective_snr_db"] == -60.0
    assert meta["effective_snr_db"] < 0.0


def test_frozen_recipe_from_test_bucket_writes_mix_map_and_effective_snr(tmp_path, monkeypatch):
    """Test-bucket base + train overlay generates into the test split; the
    sidecar carries effective_snr_db (even with scale_noise_to_snr off) and a
    rendered two-color mix map that recipe deletion removes again."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import delete_generated_for_recipes, materialize_library_recipes

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "glass.wav"
    overlay = tmp_path / "amb.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.02), sr)
    _import_library_clip(base, "glass", project_id="p", dataset_role="test")
    _import_library_clip(overlay, "amb", project_id="p", dataset_role="train")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False
    cfg.augmentation.scale_noise_to_snr = False
    recipe = {
        "id": "recipe-frozen",
        "recipe_id": "recipe-frozen",
        "frozen_test": True,
        "base_filters": [{
            "labels": ["glass"], "project_ids": ["p"],
            "split": "test", "dataset_role": "test",
        }],
        "overlay_filters": [{
            "labels": ["amb"], "project_ids": ["p"],
            "split": "train", "dataset_role": "train",
        }],
    }

    counts = materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="frozen")

    assert counts["glass"]["overlay"] == 1
    out_dir = _derived_dir(tmp_path, "p", "test", "glass")
    sidecar = next(out_dir.glob("*.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["dataset_role"] == "test"
    assert isinstance(meta["effective_snr_db"], float)
    mix_png = sidecar.with_suffix(".mix.png")
    assert meta["mix_map"] == mix_png.name
    assert mix_png.exists() and mix_png.stat().st_size > 0

    deleted = delete_generated_for_recipes("p", ["recipe-frozen"], frozen_test=True)
    assert deleted == 1
    assert not mix_png.exists()
    assert not sidecar.exists()


def test_library_recipe_can_generate_pitch_only_when_overlay_disabled(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data import augment as augment_mod

    monkeypatch.setattr(
        augment_mod,
        "apply_pitch_shift",
        lambda wave, sr, semitones: (wave.copy(), {"semitones": float(semitones)}),
    )

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "ikea.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    _import_library_clip(base, "ikea_wine", project_id="p")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = True
    cfg.augmentation.overlay_enabled = False
    cfg.augmentation.mask_filter_enabled = True
    recipe = {
        "id": "pitch-only",
        "recipe_id": "pitch-only",
        "base_filters": [{"labels": ["ikea_wine"], "project_ids": ["p"]}],
        "overlay_filters": [],
        "settings": {
            "enabled": True,
            "samples_per_source": 1,
            "pitch_shift_enabled": True,
            "overlay_enabled": False,
            "mask_filter_enabled": True,
        },
    }

    counts = augment_mod.materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="ui")

    assert counts["ikea_wine"]["pitch"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "ikea_wine").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["augmentation"] == "pitch"
    assert meta["label"] == "ikea_wine"
    assert meta["overlay_label"] == ""
    assert meta["overlay_source"] == ""
    assert meta["augmentation_params"]["overlay_enabled"] is False
    assert meta["augmentation_params"]["mask_filter_enabled"] is False


def test_library_recipe_rejects_noop_augmentation_settings(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data import augment as augment_mod

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "ikea.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    _import_library_clip(base, "ikea_wine", project_id="p")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.overlay_enabled = False
    recipe = {
        "id": "noop",
        "recipe_id": "noop",
        "base_filters": [{"labels": ["ikea_wine"], "project_ids": ["p"]}],
        "overlay_filters": [],
        "settings": {
            "enabled": True,
            "samples_per_source": 1,
            "pitch_shift_enabled": False,
            "overlay_enabled": False,
        },
    }

    with pytest.raises(augment_mod.AugmentationDisabled, match="enable pitch shift"):
        augment_mod.materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="ui")


@pytest.mark.parametrize(
    ("threshold_field", "metric_field"),
    [
        ("mask_min_mean_db", "masking_score_db"),
        ("mask_min_p25_db", "masking_score_p25_db"),
        ("mask_min_band_db", "masking_score_min_db"),
    ],
)
def test_library_recipe_mask_filter_allows_single_threshold(
    tmp_path, monkeypatch, threshold_field, metric_field,
):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data import augment as augment_mod

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "base.wav"
    overlay = tmp_path / "overlay.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.01), sr)
    _import_library_clip(base, "ikea_wine", project_id="p")
    _import_library_clip(overlay, "cutlery", project_id="p")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = True
    cfg.augmentation.mask_candidate_attempts = 1
    cfg.augmentation.mask_min_mean_db = None
    cfg.augmentation.mask_min_p25_db = None
    cfg.augmentation.mask_min_band_db = None
    setattr(cfg.augmentation, threshold_field, -999.0)
    recipe = {
        "id": f"single-{threshold_field}",
        "recipe_id": f"single-{threshold_field}",
        "base_filters": [{"labels": ["ikea_wine"], "project_ids": ["p"]}],
        "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
        "settings": {
            "enabled": True,
            "samples_per_source": 1,
            "pitch_shift_enabled": False,
            "overlay_enabled": True,
            "mask_filter_enabled": True,
            "mask_min_mean_db": None,
            "mask_min_p25_db": None,
            "mask_min_band_db": None,
            threshold_field: -999.0,
        },
    }

    counts = augment_mod.materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix=threshold_field)

    assert counts["ikea_wine"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "ikea_wine").glob(f"{threshold_field}_aug_*.json"))
    meta = json.loads(sidecar.read_text())
    assert metric_field in meta
    assert meta["augmentation_params"][threshold_field] == -999.0
    assert meta["augmentation_params"]["mask_filter_passed"] is True


def test_library_recipe_blank_mask_thresholds_accept_and_store_mask_metadata(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data import augment as augment_mod

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "base.wav"
    overlay = tmp_path / "overlay.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.01), sr)
    _import_library_clip(base, "ikea_wine", project_id="p")
    _import_library_clip(overlay, "cutlery", project_id="p")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = True
    cfg.augmentation.mask_candidate_attempts = 1
    cfg.augmentation.mask_min_mean_db = None
    cfg.augmentation.mask_min_p25_db = None
    cfg.augmentation.mask_min_band_db = None
    recipe = {
        "id": "blank-mask",
        "recipe_id": "blank-mask",
        "base_filters": [{"labels": ["ikea_wine"], "project_ids": ["p"]}],
        "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
        "settings": {
            "enabled": True,
            "samples_per_source": 1,
            "pitch_shift_enabled": False,
            "overlay_enabled": True,
            "mask_filter_enabled": True,
            "mask_min_mean_db": None,
            "mask_min_p25_db": None,
            "mask_min_band_db": None,
        },
    }

    counts = augment_mod.materialize_library_recipes(cfg, "p", [recipe], seed=123, run_prefix="blank-mask")

    assert counts["ikea_wine"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "ikea_wine").glob("blank_mask_aug_*.json"))
    meta = json.loads(sidecar.read_text())
    assert "masking_score_db" in meta
    assert "masking_score_p25_db" in meta
    assert "masking_score_min_db" in meta
    assert meta["augmentation_params"]["mask_min_mean_db"] is None
    assert meta["augmentation_params"]["mask_min_p25_db"] is None
    assert meta["augmentation_params"]["mask_min_band_db"] is None
    assert meta["augmentation_params"]["mask_filter_passed"] is True


def test_target_based_star_overlay_and_negative_skip_mask(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    room = tmp_path / "room.wav"
    cutlery = tmp_path / "cutlery.wav"
    speech = tmp_path / "speech.wav"
    sf.write(str(room), _noise(sr, amp=0.01), sr)
    sf.write(str(cutlery), _noise(sr, amp=0.02), sr)
    sf.write(str(speech), _noise(sr, amp=0.03), sr)
    _import_library_clip(room, "room_tone", project_id="p")
    _import_library_clip(cutlery, "cutlery", project_id="p")
    _import_library_clip(speech, "speech", project_id="p")

    cfg = Config.model_validate({
        "dataset": {"target_classes": ["_negative"]},
        "augmentation": {
            "combinations": {
                "_negative": [
                    {
                        "base": "room_tone",
                        "overlay": "*",
                        "description": "negative-only ambience pile",
                        "tags": "negative, ambience",
                    }
                ]
            }
        },
    })
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False

    counts = materialize_augmented(cfg, "p", seed=123, run_prefix="neg")

    assert counts["_negative"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "trainval", "_negative").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    params = meta["augmentation_params"]
    assert meta["class_name"] == "_negative"
    assert meta["description"] == "negative-only ambience pile"
    assert meta["tags"] == ["negative", "ambience"]
    assert "masking_score_db" not in meta
    assert params["base_role"] == "negative"
    assert params["overlay_label"] == "*"
    assert params["actual_overlay_label"] in {"cutlery", "speech", "room_tone"}
    assert params["mask_filter_passed"] is True


def test_target_based_frozen_test_augmentation_writes_test_role(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    base = tmp_path / "cling_test.wav"
    overlay = tmp_path / "room_test.wav"
    sf.write(str(base), _impulse(sr, amp=0.08), sr)
    sf.write(str(overlay), _noise(sr, amp=0.02), sr)
    _import_library_clip(base, "cling", project_id="p", dataset_role="test")
    _import_library_clip(overlay, "room", project_id="p", dataset_role="test")

    cfg = Config.model_validate({
        "dataset": {"target_classes": ["cling"]},
        "augmentation": {
            "combinations": {
                "cling": [{"base": "cling", "overlay": "room"}]
            }
        },
    })
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False

    counts = materialize_augmented(
        cfg,
        "p",
        seed=123,
        run_prefix="target-test",
        frozen_test=True,
    )

    assert counts["cling"]["overlay"] == 1
    sidecar = next(_derived_dir(tmp_path, "p", "test", "cling").glob("*.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["dataset_role"] == "test"
    assert meta["target_class"] == "cling"


def test_augmented_filenames_include_run_prefix_and_append(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    positive_dir = layout.positives_dir("cling")
    negative_dir = layout.negatives_dir("room")
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    clear = positive_dir / "clear_glass_hit.wav"
    busy = negative_dir / "busy_bar.wav"
    sf.write(str(clear), _impulse(sr, amp=0.08), sr)
    sf.write(str(busy), _noise(sr, amp=0.03), sr)
    _import_library_clip(clear, "cling")
    _import_library_clip(busy, "room")

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False
    cfg.augmentation.generate_negative_overlay = False
    cfg.dataset.project = {"_negative": [{"label": "room"}]}

    materialize_augmented(cfg, "p", seed=123, run_prefix="Bar Night 01")
    materialize_augmented(cfg, "p", seed=123, run_prefix="Bar Night 01")

    out_dir = _derived_dir(tmp_path, "p", "trainval", "cling")
    sidecars = sorted(out_dir.glob("*_aug_*.json"))
    assert len(sidecars) == 2
    names = [p.name for p in sidecars]
    assert names == [
        "bar_night_01_aug_cling_0001.json",
        "bar_night_01_aug_cling_0002.json",
    ]
    meta = json.loads(sidecars[0].read_text())
    assert meta["augmentation_run_prefix"] == "Bar Night 01"


def test_frozen_test_augmentation_writes_to_test_augmented(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    positive_dir = layout.test_positives_dir("cling")
    negative_dir = layout.test_negatives_dir("room")
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    clear = positive_dir / "clear_glass_hit.wav"
    room = negative_dir / "quiet_room.wav"
    sf.write(str(clear), _impulse(sr, amp=0.08), sr)
    sf.write(str(room), _noise(sr, amp=0.03), sr)
    _import_library_clip(clear, "cling", dataset_role="test")
    _import_library_clip(room, "room", dataset_role="test")

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.mask_filter_enabled = False
    cfg.augmentation.generate_negative_overlay = False
    cfg.dataset.project = {"_negative": [{"label": "room"}]}

    counts = materialize_augmented(
        cfg,
        "p",
        seed=123,
        run_prefix="Frozen Batch",
        frozen_test=True,
    )
    second_counts = materialize_augmented(
        cfg,
        "p",
        seed=123,
        run_prefix="Frozen Batch",
        frozen_test=True,
    )

    assert counts["cling"]["overlay"] == 1
    assert second_counts["cling"]["overlay"] == 1
    out_dir = _derived_dir(tmp_path, "p", "test", "cling")
    sidecars = sorted(out_dir.glob("*_aug_*.json"))
    assert [p.name for p in sidecars] == [
        "frozen_batch_aug_cling_0001.json",
        "frozen_batch_aug_cling_0002.json",
    ]
    meta = json.loads(sidecars[0].read_text())
    assert meta["class_name"] == "cling"
    assert meta["source"] == "clear_glass_hit.wav"
    assert meta["augmentation_run_prefix"] == "Frozen Batch"
    assert not _derived_dir(tmp_path, "p", "trainval", "cling").exists()


def test_overlay_pitch_shift_metadata_is_inside_overlay(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    positive_dir = layout.positives_dir("cling")
    negative_dir = layout.negatives_dir("room")
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    cling = positive_dir / "cling.wav"
    room = negative_dir / "room.wav"
    sf.write(str(cling), _impulse(sr, amp=0.08), sr)
    sf.write(str(room), _noise(sr, amp=0.03), sr)
    _import_library_clip(cling, "cling")
    _import_library_clip(room, "room")

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = True
    cfg.augmentation.semitones_range = (1.0, 1.0)
    cfg.augmentation.mask_filter_enabled = False
    cfg.augmentation.generate_negative_overlay = False
    cfg.dataset.project = {"_negative": [{"label": "room"}]}

    counts = materialize_augmented(cfg, "p", seed=123)

    assert counts["cling"]["overlay"] == 1
    meta = json.loads(
        next(_derived_dir(tmp_path, "p", "trainval", "cling").glob("*_aug_*.json"))
        .read_text()
    )
    assert meta["augmentation"] == "overlay"
    assert meta["augmentation_params"]["pitch_shift_enabled"] is True
    assert meta["augmentation_params"]["semitones"] == 1.0
    assert meta["augmentation_params"]["negative_source"] == "room.wav"


def test_mask_aware_overlay_retries_until_candidate_is_not_buried(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data import augment as aug_mod
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    positive_dir = layout.positives_dir("cling")
    negative_dir = layout.negatives_dir("room")
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    positive = _soft_cling(position=12000, amp=0.08, attack_ms=50)
    cling = positive_dir / "cling.wav"
    sf.write(str(cling), positive, sr)
    bad_path = negative_dir / "bad_in_band.wav"
    good_path = negative_dir / "good_high_band.wav"
    sf.write(str(bad_path), _tone_segment(sr, start=0, duration=sr, freq=4200.0, amp=0.2), sr)
    sf.write(str(good_path), _tone_segment(sr, start=0, duration=sr, freq=10000.0, amp=0.2), sr)
    _import_library_clip(cling, "cling")
    _import_library_clip(bad_path, "room")
    _import_library_clip(good_path, "room")

    bad_wave = _tone_segment(sr, start=0, duration=sr, freq=4200.0, amp=0.2)
    good_wave = _tone_segment(sr, start=0, duration=sr, freq=10000.0, amp=0.2)
    sequence = [(bad_wave, bad_path), (good_wave, good_path)]

    def fake_random(self, n_samples, rng=None):  # noqa: ANN001
        if sequence:
            return sequence.pop(0)
        return good_wave, good_path

    monkeypatch.setattr(aug_mod.AudioPool, "random", fake_random)

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    cfg.augmentation.samples_per_source = 1
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.snr_db_range = (0.0, 0.0)
    cfg.augmentation.generate_negative_overlay = False
    cfg.augmentation.mask_filter_enabled = True
    cfg.augmentation.mask_candidate_attempts = 2
    cfg.dataset.project = {"_negative": [{"label": "room"}]}

    counts = materialize_augmented(cfg, "p", seed=123)

    assert counts["cling"]["overlay"] == 1
    meta = json.loads(
        next(_derived_dir(tmp_path, "p", "trainval", "cling").glob("*_aug_*.json"))
        .read_text()
    )
    params = meta["augmentation_params"]
    assert params["negative_source"] == "good_high_band.wav"
    assert params["mask_attempts_used"] == 2
    assert params["mask_filter_passed"] is True
    assert params["mask_fallback_used"] is False
    assert meta["masking_score_p25_db"] >= cfg.augmentation.mask_min_p25_db


def test_negative_overlay_uses_samples_per_source_per_base(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    layout = ProjectLayout("p")
    layout.create()
    sr = 32000
    negative_dir = layout.negatives_dir("room")
    negative_dir.mkdir(parents=True, exist_ok=True)
    room_a = negative_dir / "room_a.wav"
    room_b = negative_dir / "room_b.wav"
    sf.write(str(room_a), _noise(sr, amp=0.03), sr)
    sf.write(str(room_b), _noise(sr, amp=0.04), sr)
    _import_library_clip(room_a, "room")
    _import_library_clip(room_b, "room")

    cfg = Config()
    cfg.model.classes = ["_negative"]
    cfg.augmentation.samples_per_source = 2
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.generate_positive_overlay = False
    cfg.augmentation.generate_negative_overlay = True
    cfg.dataset.project = {"_negative": [{"label": "room"}]}

    counts = materialize_augmented(cfg, "p", seed=123)

    assert counts["_negative"]["overlay"] == 4
    out_dir = _derived_dir(tmp_path, "p", "trainval", "_negative")
    sidecars = sorted(out_dir.glob("*_aug_*.json"))
    assert len(sidecars) == 4
    for sidecar in sidecars:
        meta = json.loads(sidecar.read_text())
        assert meta["augmentation"] == "overlay"
        assert meta["augmentation_params"]["base_role"] == "negative"
        assert meta["augmentation_params"]["negative_source"] in {"room_a.wav", "room_b.wav"}


def test_augment_does_not_generate_negative_bucket_when_class_absent(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data import library as lib_mod
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config, ProjectLayout
    from sed.data.augment import materialize_augmented

    project_id = "p"
    layout = ProjectLayout(project_id)
    layout.create()
    sr = 32000

    positive_dir = layout.positives_dir("cling_a")
    negative_dir = layout.negatives_dir("cutlery")
    for folder in (positive_dir, negative_dir):
        folder.mkdir(parents=True, exist_ok=True)
    cling = positive_dir / "cling.wav"
    cutlery = negative_dir / "cutlery.wav"
    sf.write(str(cling), _impulse(sr), sr)
    sf.write(str(cutlery), _noise(sr, amp=0.1), sr)
    _import_library_clip(cling, "cling_a", project_id=project_id)
    _import_library_clip(cutlery, "cutlery", project_id=project_id)

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["cling_a"],
            "project": {
                "_negative": [{"category": "negatives", "labels": ["cutlery"]}],
            },
        },
    })
    cfg.augmentation.pitch_shift_enabled = False
    cfg.augmentation.generate_positive_overlay = False
    cfg.augmentation.generate_negative_overlay = True
    cfg.augmentation.samples_per_source = 2

    counts = materialize_augmented(cfg, project_id, seed=123)

    assert "_negative" not in counts
    assert not _derived_dir(tmp_path, project_id, "trainval", "_negative").exists()


# ---------------------------------------------------------------------------
# _ensure_length placement (short imported clips, e.g. legacy 0.4 s WAVs)
# ---------------------------------------------------------------------------


def test_ensure_length_default_pads_short_clip_at_start():
    x = np.ones(100, dtype=np.float32)
    out = _ensure_length(x, 300)
    assert len(out) == 300
    assert np.array_equal(out[:100], x)
    assert not out[100:].any()


def test_ensure_length_rng_places_short_clip_at_random_offset():
    x = np.ones(100, dtype=np.float32)
    rng = random.Random(7)
    offsets = set()
    for _ in range(20):
        out = _ensure_length(x, 1000, rng=rng)
        nonzero = np.nonzero(out)[0]
        assert len(nonzero) == 100
        offset = int(nonzero[0])
        assert 0 <= offset <= 900
        assert np.array_equal(out[offset:offset + 100], x)
        offsets.add(offset)
    assert len(offsets) > 1


def test_ensure_length_rng_is_seed_reproducible():
    x = np.ones(50, dtype=np.float32)
    a = _ensure_length(x, 500, rng=random.Random(3))
    b = _ensure_length(x, 500, rng=random.Random(3))
    assert np.array_equal(a, b)


def test_ensure_length_rng_long_clip_still_crops_to_target():
    x = np.arange(1000, dtype=np.float32)
    out = _ensure_length(x, 400, rng=random.Random(0))
    assert len(out) == 400
    # crop is a contiguous slice of the source
    start = int(out[0])
    assert np.array_equal(out, x[start:start + 400])


def test_ensure_length_exact_length_clip_passes_through():
    x = np.arange(200, dtype=np.float32)
    out = _ensure_length(x, 200, rng=random.Random(1))
    assert np.array_equal(out, x)


def test_ensure_length_long_clip_without_rng_is_deterministic_center_crop():
    x = np.arange(1000, dtype=np.float32)
    a = _ensure_length(x, 400)
    b = _ensure_length(x, 400)
    assert np.array_equal(a, b)
    assert a[0] == (1000 - 400) // 2


def test_ensure_length_long_clip_crop_is_seed_reproducible():
    x = np.arange(1000, dtype=np.float32)
    a = _ensure_length(x, 400, rng=random.Random(5))
    b = _ensure_length(x, 400, rng=random.Random(5))
    assert np.array_equal(a, b)
