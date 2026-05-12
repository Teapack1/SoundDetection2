from __future__ import annotations

import json
import sys
import types

import numpy as np
import soundfile as sf
from typer.testing import CliRunner


def _patch_paths(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


def _wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(32000, dtype="float32"), 32000)
    return path


def _multichannel_wav(path, channels=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((32000, channels), dtype="float32")
    sf.write(str(path), data, 32000)
    return path


def test_record_help_exposes_only_flat_record_command():
    from sed.cli import app

    result = CliRunner().invoke(app, ["record", "--help"])

    assert result.exit_code == 0
    assert "--class" in result.output
    assert "-c" in result.output
    assert "-n" in result.output
    assert "-du" in result.output
    assert "-de" in result.output
    assert "-mp" in result.output
    assert "-dev" in result.output
    assert "-ch" in result.output
    assert "--onset-threshold" in result.output
    assert "-ot" in result.output
    assert "-t" in result.output
    assert "--tags" in result.output
    assert "-tg" in result.output
    assert "--density" in result.output
    assert "--mode" not in result.output
    assert "--chunk" not in result.output
    assert "--randomize-window" not in result.output
    assert "--random-preroll" not in result.output
    assert "--preroll-ms" not in result.output


def test_record_cli_parses_tags_and_aliases(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.cli import app
    from sed.data import recorder as recorder_mod

    captured = {}

    def fake_record_auto(*args, **kwargs):
        captured["cfg"] = args[0]
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(recorder_mod, "record_auto", fake_record_auto)

    result = CliRunner().invoke(app, [
        "record",
        "-c", "x",
        "-n", "0",
        "-du", "1s",
        "-de", "quiet",
        "-mp", "table left",
        "-ot", "-77",
        "-tg", "wine,#full-50,wine",
    ])

    assert result.exit_code == 0
    assert captured["cfg"].onset.min_flux_db == -77.0
    assert abs(captured["cfg"].onset.min_flux - (10.0 ** (-77.0 / 20.0))) < 1e-12
    assert captured["tags"] == ["wine", "full-50"]
    assert captured["mic_position"] == "table left"
    assert captured["density"] == "quiet"


def test_record_bare_device_lists_devices(monkeypatch):
    from sed.audio import capture as capture_mod
    from sed.cli import app

    fake_sd = types.SimpleNamespace(
        default=types.SimpleNamespace(device=[7, None]),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(
        capture_mod,
        "list_input_devices",
        lambda: [
            capture_mod.AudioDevice(
                index=7,
                name="Fake USB Interface",
                max_input_channels=2,
                default_samplerate=48000,
                hostapi_name="ALSA",
            )
        ],
    )

    for flag in ("--device", "-dev"):
        result = CliRunner().invoke(app, ["record", flag])

        assert result.exit_code == 0
        assert "Audio input devices" in result.output
        assert "Fake USB Interface" in result.output
        assert "yes" in result.output


def test_recording_session_writes_record_sample_rate_and_tags_to_sidecar(tmp_path):
    from sed.config import Config
    from sed.data.recorder import RecordingSession

    cfg = Config()
    cfg.audio.record_sample_rate = 48000
    cfg.model.sample_rate = 32000
    session = RecordingSession(
        cfg,
        tmp_path,
        "cheers_quiet",
        "p",
        description="quiet cheers",
        tags=["wine-glass", "full-50"],
    )

    wav, _meta = session._save_one(np.zeros((48000, 1), dtype="float32"), None)

    assert sf.info(str(wav)).samplerate == 48000
    sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["sample_rate"] == 48000
    assert sidecar["duration_s"] == 1.0
    assert sidecar["description"] == "quiet cheers"
    assert sidecar["tags"] == ["wine-glass", "full-50"]


def test_record_validates_project_and_test_gating(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.cli import app

    runner = CliRunner()
    no_project = runner.invoke(app, ["record", "--class", "x", "--test"])
    assert no_project.exit_code != 0
    assert "--test requires --project" in no_project.output

    missing = runner.invoke(app, ["record", "--class", "x", "--project", "missing", "--num", "0"])
    assert missing.exit_code != 0
    assert "Project 'missing' does not exist" in missing.output


def test_library_import_uses_unified_sounds_sidecar(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.data.library import Library

    src = _wav(tmp_path / "src.wav")
    clip = Library().import_clip(
        src,
        label="glass",
        project_id="p",
        dataset_role="test",
        density="empty",
    )

    wav = tmp_path / "data_library" / clip.relpath
    assert wav.parent == tmp_path / "data_library" / "sounds" / "glass"
    meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["category"] == "sounds"
    assert meta["label"] == "glass"
    assert meta["project_id"] == "p"
    assert meta["dataset_role"] == "test"
    assert meta["density"] == "empty"


def test_library_repeated_queries_reuse_manifest_cache(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.data.library import Library

    writer = Library()
    writer.import_clip(_wav(tmp_path / "a.wav"), label="glass", project_id="p")
    writer.import_clip(_wav(tmp_path / "b.wav"), label="metal", project_id="p")

    lib = Library()
    real_open = open
    manifest_reads = 0

    def counting_open(file, *args, **kwargs):
        nonlocal manifest_reads
        mode = args[0] if args else kwargs.get("mode", "r")
        if str(file) == str(lib.manifest) and "r" in str(mode):
            manifest_reads += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)

    assert [c.label for c in lib.query(label="glass")] == ["glass"]
    assert [c.label for c in lib.query(label="metal")] == ["metal"]
    assert len(lib.all_clips()) == 2
    assert manifest_reads == 1


def test_rebuild_manifest_indexes_legacy_folders_as_sounds(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.data.library import Library

    legacy = _wav(tmp_path / "data_library" / "positives" / "glass" / "legacy.wav")
    legacy.with_suffix(".json").write_text(json.dumps({
        "uuid": "legacy-glass",
        "class_name": "glass",
        "category": "positives",
        "project_id": "p",
    }), encoding="utf-8")

    clips = Library().rebuild_manifest()

    assert len(clips) == 1
    assert clips[0].category == "sounds"
    assert clips[0].label == "glass"
    assert clips[0].relpath == "positives/glass/legacy.wav"


def test_library_manifest_prunes_manual_wav_deletion(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.data.library import Library

    src = _wav(tmp_path / "src.wav")
    lib = Library()
    clip = lib.import_clip(src, label="glass", project_id="p")
    wav = tmp_path / "data_library" / clip.relpath
    sidecar = wav.with_suffix(".json")
    thumb = wav.with_suffix(".spec.png")
    thumb.write_text("thumbnail", encoding="utf-8")

    wav.unlink()

    assert lib.all_clips() == []
    assert not sidecar.exists()
    assert not thumb.exists()
    assert (tmp_path / "data_library" / "manifest.jsonl").read_text(encoding="utf-8") == ""


def test_library_manifest_prunes_manual_sidecar_deletion(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.data.library import Library

    src = _wav(tmp_path / "src.wav")
    lib = Library()
    clip = lib.import_clip(src, label="glass", project_id="p")
    wav = tmp_path / "data_library" / clip.relpath

    wav.with_suffix(".json").unlink()

    assert lib.rebuild_manifest() == []
    assert not wav.exists()


def test_migrate_legacy_dry_run_and_copy(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.data.library import Library

    legacy = _wav(tmp_path / "data_library" / "negatives" / "room" / "room.wav")
    legacy.with_suffix(".json").write_text(json.dumps({
        "uuid": "legacy-room",
        "class_name": "room",
        "project_id": "p",
        "dataset_role": "train",
    }), encoding="utf-8")

    lib = Library()
    dry = lib.migrate_legacy(dry_run=True)

    assert dry["count"] == 1
    assert not (tmp_path / "data_library" / "sounds" / "room" / "room.wav").exists()

    copied = lib.migrate_legacy(mode="copy")
    migrated = tmp_path / "data_library" / "sounds" / "room" / "room.wav"
    meta = json.loads(migrated.with_suffix(".json").read_text(encoding="utf-8"))

    assert copied["count"] == 1
    assert legacy.exists()
    assert migrated.exists()
    assert meta["category"] == "sounds"
    assert meta["label"] == "room"
    assert meta["source_path"] == str(legacy.resolve())
    clips = lib.all_clips()
    assert [c.relpath for c in clips] == ["sounds/room/room.wav"]


def test_library_split_channels_cli_reports_progress(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.cli import app
    from sed.data.library import Library

    lib = Library()
    lib.import_clip(_multichannel_wav(tmp_path / "glass.wav", channels=2), label="glass")

    result = CliRunner().invoke(app, ["library", "split-channels", "--all"], color=False)

    assert result.exit_code == 0
    assert "Splitting 1 multichannel recording(s)" in result.output
    assert "Rebuilding library manifest" in result.output
    assert "split 1 recording(s) into 2 mono clip(s)" in result.output


def test_compose_ignores_project_wavs_and_uses_library_mappings(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    _wav(layout.recordings_dir / "positives" / "cling" / "ignored.wav")

    lib = Library()
    for i, project_id in enumerate(["_library", "p", "other"]):
        lib.import_clip(
            _wav(tmp_path / f"cling_{i}.wav"),
            label="cling",
            project_id=project_id,
        )
    lib.import_clip(_wav(tmp_path / "room.wav"), label="room", project_id="p")
    lib.import_clip(
        _wav(tmp_path / "test.wav"),
        label="cling",
        project_id="p",
        dataset_role="test",
    )

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    cfg.dataset.project = {"_negative": [{"label": "room"}]}

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["cling"] == 2
    assert summary["by_class"]["_negative"] == 1
    assert summary["frozen_test"]["by_class"]["cling"] == 1


def test_compose_prunes_deleted_clip_selection_and_exclusions(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import (
        compose_dataset,
        exclusion_manifest_path,
        load_dataset_selection,
        selection_manifest_path,
        write_dataset_selection,
        write_excluded_clip_paths,
    )
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    lib = Library()
    clip = lib.import_clip(_wav(tmp_path / "glass.wav"), label="glass", project_id="p")
    wav = tmp_path / "data_library" / clip.relpath
    write_dataset_selection(layout, {
        "clip_assignments": {
            "glass": [{"path": str(wav), "uuid": clip.uuid, "relpath": clip.relpath}]
        }
    })
    write_excluded_clip_paths(layout, [wav])

    wav.unlink()

    cfg = Config()
    cfg.model.classes = ["glass"]
    cfg.dataset.sources = {"glass": []}
    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["glass"] == 0
    assert load_dataset_selection(layout)["clip_assignments"] == {}
    exclusions = json.loads(exclusion_manifest_path(layout).read_text(encoding="utf-8"))
    assert exclusions["paths"] == []
    assert selection_manifest_path(layout).exists()


# ---------------------------------------------------------------------------
# import tags + slice_file (legacy clip / ambience bed import support)
# ---------------------------------------------------------------------------


def test_import_clip_stores_tags(tmp_path):
    from sed.data.library import Library

    src = _wav(tmp_path / "src" / "clip.wav")
    lib = Library(tmp_path / "data_library")
    clip = lib.import_clip(src, label="legacy_clink", tags=["legacy", "old-rig"])

    assert clip.tags == ["legacy", "old-rig"]
    meta = json.loads(
        clip.absolute_path(lib.root).with_suffix(".json").read_text(encoding="utf-8")
    )
    assert meta["tags"] == ["legacy", "old-rig"]
    found = lib.query(tags_all=["legacy"])
    assert [c.uuid for c in found] == [clip.uuid]


def test_library_import_cli_passes_tags(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.cli import app
    from sed.data.library import Library

    _wav(tmp_path / "legacy" / "a.wav")
    result = CliRunner().invoke(
        app,
        ["library", "import", str(tmp_path / "legacy"),
         "--class", "legacy_clink", "--tags", "legacy,old-rig"],
    )

    assert result.exit_code == 0, result.output
    clips = Library().query(label="legacy_clink")
    assert len(clips) == 1
    assert clips[0].tags == ["legacy", "old-rig"]


def test_slice_file_writes_window_clips_and_drops_partial_tail(tmp_path):
    from sed.data.library import Library

    sr = 16000
    src = tmp_path / "bed.wav"
    src.parent.mkdir(parents=True, exist_ok=True)
    wave = 0.1 * np.random.RandomState(0).randn(int(3.5 * sr)).astype("float32")
    sf.write(str(src), wave, sr)

    lib = Library(tmp_path / "data_library")
    clips = lib.slice_file(src, label="ambience_bed", seconds=1.0)

    assert len(clips) == 3  # 3.5 s → 3 full windows, 0.5 s tail dropped
    for i, clip in enumerate(clips, start=1):
        wav_path = clip.absolute_path(lib.root)
        info = sf.info(str(wav_path))
        assert info.frames == sr
        assert wav_path.with_suffix(".json").exists()
        assert clip.origin == "sliced"
        assert clip.label == "ambience_bed"
        assert clip.recording_counter == i
        assert clip.source_path == str(src.resolve())
        assert "slice" in clip.notes
    assert len({c.session_id for c in clips}) == 1
    assert len(lib.query(label="ambience_bed")) == 3


def test_slice_file_hop_controls_overlap(tmp_path):
    from sed.data.library import Library

    sr = 8000
    src = tmp_path / "bed.wav"
    sf.write(str(src), np.zeros(2 * sr, dtype="float32"), sr)

    lib = Library(tmp_path / "data_library")
    clips = lib.slice_file(src, label="amb", seconds=1.0, hop_seconds=0.5)

    # starts at 0.0, 0.5, 1.0 s — the 1.5 s start has no full window left
    assert len(clips) == 3


def test_library_slice_cli(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    from sed.cli import app
    from sed.data.library import Library

    src = tmp_path / "amb.wav"
    sf.write(str(src), np.zeros(2 * 16000, dtype="float32"), 16000)
    result = CliRunner().invoke(
        app,
        ["library", "slice", str(src), "--class", "ambience_bar", "--seconds", "1.0"],
    )

    assert result.exit_code == 0, result.output
    clips = Library().query(label="ambience_bar")
    assert len(clips) == 2
    assert all(c.origin == "sliced" for c in clips)
