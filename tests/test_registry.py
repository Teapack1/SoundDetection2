"""Tests for the model bundle registry (projects/<id>/models/<model_id>/)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sed.train import registry


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """Spin up a fake projects/<id>/ tree under tmp_path."""
    import sed.config as config_mod
    monkeypatch.setattr(config_mod, "PROJECTS_DIR", tmp_path / "projects")
    project_id = "demo"
    layout = config_mod.ProjectLayout(project_id)
    layout.create()
    config_mod.write_project_config_template(layout, name="Demo")
    return project_id, tmp_path


def _seed_model(project_id: str, model_id: str, manifest: dict | None = None) -> None:
    """Create a minimal trained-model bundle (config snapshot + manifest)."""
    from sed.config import ProjectLayout
    model_dir = ProjectLayout(project_id).model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")
    registry.save_manifest(project_id, model_id, manifest or {"model_id": model_id})


def test_new_model_id_unique_collision(project_root):
    project_id, _ = project_root
    fixed = datetime(2026, 5, 15, 14, 27, tzinfo=timezone.utc)
    a = registry.new_model_id(project_id, description="baseline", now=fixed)
    # Materialise the dir so the next call sees a collision and disambiguates.
    from sed.config import ProjectLayout
    ProjectLayout(project_id).model_dir(a).mkdir(parents=True)
    b = registry.new_model_id(project_id, description="baseline", now=fixed)
    assert a != b
    assert b.endswith("-2")


def test_new_model_id_slugifies_description(project_root):
    project_id, _ = project_root
    fixed = datetime(2026, 5, 15, 14, 27, tzinfo=timezone.utc)
    mid = registry.new_model_id(project_id, description="Wider SNR!", now=fixed)
    assert mid == "2026-05-15_1427-wider-snr"


def test_new_model_id_truncates_long_description(project_root):
    project_id, _ = project_root
    fixed = datetime(2026, 5, 15, 14, 27, tzinfo=timezone.utc)
    mid = registry.new_model_id(
        project_id,
        description="round two with mined hard negatives from the bar",
        now=fixed,
    )
    slug = mid.split("_1427-", 1)[1]
    assert len(slug) <= 24


def test_load_manifests_sorted_newest_first(project_root):
    project_id, _ = project_root
    for stamp, desc in [
        ("2026-05-14_1000", "old"),
        ("2026-05-15_1427", "new"),
        ("2026-05-15_0900", "mid"),
    ]:
        _seed_model(project_id, f"{stamp}-{desc}", {
            "model_id": f"{stamp}-{desc}",
            "description": desc,
            "created_at": stamp.replace("_", "T") + ":00Z",
        })
    descs = [m["description"] for m in registry.load_manifests(project_id)]
    assert descs == ["new", "mid", "old"]


def test_set_active_model_no_duplicate_deploy_section(project_root):
    """A [deploy] section without active_model must get the key inserted, not a
    second [deploy] appended (which would be invalid TOML)."""
    import tomllib
    from sed.config import ProjectLayout
    layout = ProjectLayout(project_root[0])
    path = layout.config_toml

    # 1. [deploy] exists but lacks active_model (e.g. only record_device set).
    path.write_text(
        '[deploy]\nrecord_device = "default"\n\n[mining]\nenabled = false\n',
        encoding="utf-8",
    )
    registry.set_active_model(layout.id, "model_a")
    text = path.read_text()
    assert text.count("[deploy]") == 1
    data = tomllib.loads(text)
    assert data["deploy"]["active_model"] == "model_a"
    assert data["deploy"]["record_device"] == "default"

    # 2. Re-setting replaces the value in place (still one section).
    registry.set_active_model(layout.id, "model_b")
    text = path.read_text()
    assert text.count("[deploy]") == 1
    assert tomllib.loads(text)["deploy"]["active_model"] == "model_b"

    # 3. No [deploy] at all → one is appended.
    path.write_text("[mining]\nenabled = false\n", encoding="utf-8")
    registry.set_active_model(layout.id, "model_c")
    text = path.read_text()
    assert text.count("[deploy]") == 1
    assert tomllib.loads(text)["deploy"]["active_model"] == "model_c"


def test_get_active_model_falls_back_to_single_model(project_root):
    project_id, _ = project_root
    # No active_model set, exactly one model → it is the de-facto active one.
    _seed_model(project_id, "2026-05-15_1000-only")
    assert registry.get_active_model(project_id) == "2026-05-15_1000-only"

    # Two models, none active → ambiguous → None.
    _seed_model(project_id, "2026-05-15_1100-other")
    assert registry.get_active_model(project_id) is None

    # Explicit selection resolves it.
    registry.set_active_model(project_id, "2026-05-15_1100-other")
    assert registry.get_active_model(project_id) == "2026-05-15_1100-other"


def test_models_compare_and_select_cli(project_root):
    """`sed models compare` runs and `sed models select` switches active_model."""
    import typer
    from sed import cli
    from sed.config import load_config
    project_id, _ = project_root
    for mid in ("2026-05-15_1000-a", "2026-05-15_1100-b"):
        _seed_model(project_id, mid, {"model_id": mid, "created_at": mid})

    cli.models_compare_cmd(project=project_id)  # must not raise

    cli.models_select_cmd(project=project_id, model_id="2026-05-15_1000-a", model_id_opt=None)
    assert load_config(project_id).deploy.active_model == "2026-05-15_1000-a"

    # #N handles resolve too (newest first → #1 is the 1100 model).
    cli.models_select_cmd(project=project_id, model_id="#1", model_id_opt=None)
    assert load_config(project_id).deploy.active_model == "2026-05-15_1100-b"

    with pytest.raises(typer.BadParameter):
        cli.models_select_cmd(project=project_id, model_id="not-a-model", model_id_opt=None)


def test_models_compare_cli_shows_training_info_and_copyable_ids(project_root):
    from typer.testing import CliRunner

    from sed import cli

    project_id, _ = project_root
    mid = "2026-06-12_1015-very-readable-model-id"
    _seed_model(project_id, mid, {
        "model_id": mid,
        "description": "long description stays below the id instead of squeezing it",
        "created_at": "2026-06-12T10:15:00Z",
        "training": {
            "epochs_ran": 12,
            "epochs_max": 20,
            "selected_val_loss": 0.12345,
            "selected_val_accuracy": 0.9876,
        },
        "test_metrics": {
            "accuracy": 0.976,
            "f1_macro_targets": 0.965,
            "ap_macro_targets": 0.991,
        },
    })

    result = CliRunner().invoke(cli.app, ["models", "compare", "--project", project_id])

    assert result.exit_code == 0
    assert "Model IDs" in result.output
    assert mid in result.output
    assert "2026-06-12 10:15" in result.output  # trained date/time in the header
    # training info folded in from the old `list` view:
    for row in ("created", "epochs", "IR", "calibrated"):
        assert row in result.output
    assert "12/20" in result.output


def test_models_compare_cli_uses_id_key_and_compact_scoreboard(project_root):
    from typer.testing import CliRunner

    from sed import cli

    project_id, _ = project_root
    first = "2026-06-12_1015-long-baseline-model-id"
    second = "2026-06-12_1045-long-augmented-model-id"
    for mid, val_loss, f1 in [
        (first, 0.25, 0.90),
        (second, 0.10, 0.95),
    ]:
        _seed_model(project_id, mid, {
            "model_id": mid,
            "description": f"description for {mid}",
            "created_at": mid[:16].replace("_", "T") + ":00Z",
            "training": {
                "epochs_ran": 8,
                "epochs_max": 10,
                "selected_val_loss": val_loss,
                "selected_val_accuracy": 0.88,
            },
            "test_metrics": {
                "accuracy": f1,
                "f1_macro_targets": f1,
                "ap_macro_targets": 0.99,
                "per_class": {
                    "alpha": {"f1": f1},
                    "beta": {"f1": f1 - 0.01},
                },
            },
        })

    result = CliRunner().invoke(cli.app, ["models", "compare", "--project", project_id])

    assert result.exit_code == 0
    assert "Model IDs" in result.output
    assert first in result.output
    assert second in result.output
    assert "Scoreboard #1-#2" in result.output
    assert "test F1" in result.output
    assert "F1 alpha" in result.output
    assert "F1·alpha" not in result.output


def test_evaluate_run_cli_all_models_replaces_existing_reports(project_root, monkeypatch):
    import importlib

    from typer.testing import CliRunner

    from sed import cli

    project_id, _ = project_root
    older = "2026-06-12_1015-long-baseline-model-id"
    newer = "2026-06-12_1045-long-augmented-model-id"
    for mid in (older, newer):
        _seed_model(project_id, mid, {
            "model_id": mid,
            "created_at": mid[:16].replace("_", "T") + ":00Z",
        })

    calls: list[tuple[str, str, float | None]] = []
    eval_mod = importlib.import_module("sed.train.eval")

    def fake_evaluate_run(cfg, project, model_id, threshold=None):
        calls.append((project, model_id, threshold))
        return {
            "n_samples": 17,
            "accuracy": 0.91,
            "probability_mode": "softmax",
            "f1_macro_targets": 0.92,
            "ap_macro_targets": 0.93,
        }

    monkeypatch.setattr(eval_mod, "evaluate_run", fake_evaluate_run)

    result = CliRunner().invoke(
        cli.app,
        ["evaluate", "--project", project_id, "--threshold", "0.42", "--all"],
    )

    assert result.exit_code == 0
    assert calls == [
        (project_id, newer, 0.42),
        (project_id, older, 0.42),
    ]
    assert "Existing frozen-test reports" in result.output
    assert newer in result.output
    assert older in result.output
    assert "evaluated 2/2 model(s)" in result.output


def test_evaluate_run_cli_all_models_rejects_model_argument(project_root):
    from typer.testing import CliRunner

    from sed import cli

    project_id, _ = project_root
    mid = "2026-06-12_1015-long-baseline-model-id"
    _seed_model(project_id, mid, {"model_id": mid, "created_at": "2026-06-12T10:15:00Z"})

    result = CliRunner().invoke(
        cli.app,
        ["evaluate", "--project", project_id, "--model-id", mid, "--all"],
    )

    assert result.exit_code != 0
    assert "pass either --model-id or --all" in result.output


def test_get_checkpoint_uses_manifest_path(project_root):
    project_id, _ = project_root
    from sed.config import ProjectLayout
    layout = ProjectLayout(project_id)
    mid = "2026-05-15_1427-baseline"
    model_dir = layout.model_dir(mid)
    model_dir.mkdir(parents=True)
    ckpt = model_dir / "checkpoint_best_epoch-022.ckpt"
    ckpt.write_text("ckpt")
    registry.save_manifest(project_id, mid, {
        "model_id": mid,
        "files": {"checkpoint": "checkpoint_best_epoch-022.ckpt"},
    })

    assert registry.get_checkpoint(project_id, mid) == ckpt


def test_delete_model_removes_directory_and_clears_selection(project_root):
    project_id, _ = project_root
    from sed.config import ProjectLayout, load_config
    layout = ProjectLayout(project_id)
    mid = "2026-05-15_1427-doomed"
    _seed_model(project_id, mid)
    registry.set_active_model(project_id, mid)
    assert registry.get_active_model(project_id) == mid

    registry.delete_model(project_id, mid)
    assert not layout.model_dir(mid).exists()
    assert load_config(project_id).deploy.active_model == ""
