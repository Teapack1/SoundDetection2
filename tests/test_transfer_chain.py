"""Recursive parent-init: any completed run can seed the next run's encoder.

Covers the unified transfer semantics: no project-kind gate, self-chaining
within one project, task mismatch as a warning (encoder-only transfer),
feature mismatch as adopt-able, and grandparent provenance in the chain.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from sed.config import Config, ProjectLayout, dump_config_snapshot  # noqa: E402


def _patch_dirs(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


def _make_cfg(project_id: str, *, backbone: str = "mn01", task: str = "clip",
              sample_rate: int = 32000, n_mels: int = 128,
              initialization: str = "random", parent_project: str = "",
              parent_run: str = "", feature_updates: dict | None = None) -> Config:
    features = {"n_mels": n_mels}
    features.update(feature_updates or {})
    return Config.model_validate({
        "project": {"id": project_id},
        "model": {
            "backbone": backbone, "pretrained": False, "task": task,
            "sample_rate": sample_rate, "initialization": initialization,
            "parent_project": parent_project, "parent_run": parent_run,
        },
        "features": features,
        "dataset": {"target_classes": ["a", "_negative"]},
    })


def _write_parent_bundle(project_id: str, run_id: str, *, backbone: str = "mn01",
                         task: str = "clip", sample_rate: int = 32000,
                         n_mels: int = 128, num_classes: int = 2,
                         init_block: dict | None = None,
                         feature_updates: dict | None = None) -> None:
    """A minimal completed run: snapshot + Lightning-shaped checkpoint + manifest."""
    from sed.audio.features import feature_band_count, feature_frame_count_for_cfg
    from sed.train.model import SedModel

    run_dir = ProjectLayout(project_id).model_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(project_id, backbone=backbone, task=task,
                    sample_rate=sample_rate, n_mels=n_mels,
                    feature_updates=feature_updates)
    dump_config_snapshot(cfg, run_dir / "training_config.json")
    model = SedModel(backbone, num_classes=num_classes, pretrained=False,
                     n_bands=feature_band_count(cfg.features, cfg.model.sample_rate),
                     n_frames=feature_frame_count_for_cfg(cfg))
    state = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state}, run_dir / "checkpoint_best_epoch-000.ckpt")
    manifest = {
        "model_id": run_id,
        "created_at": "2026-07-10T00:00:00+00:00",
        "model_recipe": {"task": task, "backbone": backbone},
        "training": {"objective": "supervised", "selected_score": 0.91,
                     "selection_metric": "val/ap_macro_targets"},
        "initialization": init_block or {"mode": "random", "weight_id": ""},
        "files": {"checkpoint": "checkpoint_best_epoch-000.ckpt"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_same_project_parent_is_compatible(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import compatible_parent_runs, parent_compatibility

    _write_parent_bundle("pa", "run-a")
    cfg = _make_cfg("pa")
    info = parent_compatibility(cfg, "pa", "run-a")
    assert info["ok"] is True
    assert info["warnings"] == []
    assert info["feature_matches"] is True
    assert info["checkpoint_sha256"]

    rows = compatible_parent_runs(cfg)
    keys = {(r["parent_project"], r["parent_run"]) for r in rows}
    assert ("pa", "run-a") in keys, "self-chaining within one project must be offered"
    row = next(r for r in rows if r["parent_run"] == "run-a")
    assert row["objective"] == "supervised"
    assert row["selection_metric"] == "val/ap_macro_targets"
    assert row["feature"]["n_mels"] == 128


def test_fitted_stat_provenance_is_reported_but_not_a_numeric_mismatch(
    tmp_path, monkeypatch,
):
    """Lineage describes fitted stats; only the resolved values feed the net."""
    _patch_dirs(tmp_path, monkeypatch)
    from sed.config import load_config_snapshot
    from sed.train.transfer import parent_compatibility, parent_feature_payload

    parent_feature = {
        "standardize": True,
        "standardize_mean": -4.25,
        "standardize_std": 2.75,
        "standardize_stats_resolved": True,
        "standardize_stats_dataset_sha256": "parent-dataset-sha",
        "standardize_stats_split_seed": 17,
        "standardize_stats_clips": 512,
    }
    _write_parent_bundle(
        "pa", "run-a", feature_updates=parent_feature,
    )
    # A separately estimated contract can have different provenance (including
    # whether the UI regards the estimate as resolved) while producing exactly
    # the same normalized tensor values.
    child = _make_cfg("pb", feature_updates={
        **parent_feature,
        "standardize_stats_resolved": False,
        "standardize_stats_dataset_sha256": "different-dataset-sha",
        "standardize_stats_split_seed": 99,
        "standardize_stats_clips": 2_048,
    })
    info = parent_compatibility(child, "pa", "run-a")
    assert info["ok"] is True, info["reason"]
    assert info["feature_matches"] is True

    parent = load_config_snapshot(
        ProjectLayout("pa").model_dir("run-a") / "training_config.json"
    )
    payload = parent_feature_payload(parent)
    assert payload["standardize_mean"] == -4.25
    assert payload["standardize_std"] == 2.75
    assert payload["standardize_stats_resolved"] is True
    assert payload["standardize_stats_dataset_sha256"] == "parent-dataset-sha"
    assert payload["standardize_stats_split_seed"] == 17
    assert payload["standardize_stats_clips"] == 512
    # The label names both value axes, so two encoders that see different
    # numbers can never read as the same frontend contract.
    assert payload["compression"] == "log"
    assert payload["normalization"] == "log+zscore-global-dataset"
    assert payload["resolved_feature_bands"] == 128
    assert payload["resolved_feature_frames"] == 40
    # 0.4 s of 32 kHz audio — the project's default model window.
    assert payload["window_samples"] == 12_800


@pytest.mark.parametrize(
    ("field", "value"),
    [("standardize_mean", -4.2), ("standardize_std", 2.7)],
)
def test_parent_rejects_changed_resolved_fitted_statistics(
    tmp_path, monkeypatch, field, value,
):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import parent_compatibility

    fitted = {
        "standardize": True,
        "standardize_mean": -4.25,
        "standardize_std": 2.75,
        "standardize_stats_resolved": True,
    }
    _write_parent_bundle("pa", "run-a", feature_updates=fitted)
    child_feature = dict(fitted)
    child_feature[field] = value
    info = parent_compatibility(
        _make_cfg("pb", feature_updates=child_feature), "pa", "run-a",
    )
    assert info["ok"] is False
    assert info["feature_matches"] is False
    assert info["hard_blocked"] is False


def test_cross_project_parent_needs_no_kind(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import parent_compatibility

    _write_parent_bundle("regular_project", "run-r")
    info = parent_compatibility(_make_cfg("other"), "regular_project", "run-r")
    assert info["ok"] is True, info["reason"]


def test_task_mismatch_is_warning_not_error(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import parent_compatibility

    _write_parent_bundle("pa", "run-a", task="clip")
    info = parent_compatibility(_make_cfg("pb", task="frame"), "pa", "run-a")
    assert info["ok"] is True, info["reason"]
    assert any("task differs" in w for w in info["warnings"])


def test_feature_mismatch_is_adoptable_not_hard_blocked(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import compatible_parent_runs, parent_compatibility

    _write_parent_bundle("pa", "run-a", n_mels=64)
    cfg = _make_cfg("pb", n_mels=128)
    info = parent_compatibility(cfg, "pa", "run-a")
    assert info["ok"] is False
    assert info["feature_matches"] is False
    assert info["hard_blocked"] is False

    row = next(r for r in compatible_parent_runs(cfg) if r["parent_run"] == "run-a")
    assert row["selectable_with_feature_adoption"] is True


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "sample_rate", 44100),
        ("model", "channels", 2),
        ("features", "window_seconds", 1.0),
        ("features", "frontend", "ced"),
        ("features", "feature_type", "linear"),
        ("features", "n_fft", 2048),
        ("features", "win_length", 640),
        ("features", "hop_length", 160),
        ("features", "n_mels", 64),
        ("features", "fmin", 80),
        ("features", "fmax", 14000),
        ("features", "stft_engine", "scipy"),
        ("features", "log_power", False),
        ("features", "db_norm", True),
        ("features", "db_min", -30.0),
        ("features", "db_max", 70.0),
        ("features", "pcen", True),
        ("features", "pcen_time_constant", 0.8),
        ("features", "pcen_gain", 0.9),
        ("features", "pcen_bias", 3.0),
        ("features", "pcen_power", 0.25),
        ("features", "pcen_eps", 1e-5),
        ("features", "standardize", True),
        ("features", "standardize_mean", 1.0),
        ("features", "standardize_std", 2.0),
        ("features", "colormap", "jet"),
        ("features", "image_size", 128),
    ],
)
def test_parent_rejects_every_frontend_semantic_mismatch(
    tmp_path, monkeypatch, section, field, value,
):
    """Encoder transfer is strict on geometry and value distribution.

    Direct assignment intentionally lets this test isolate each signature key;
    some individual values would require companion changes to form a valid
    trainable Config, but must still never compare equal to the parent.
    """
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import parent_compatibility

    _write_parent_bundle("pa", "run-a")
    cfg = _make_cfg("pb")
    setattr(getattr(cfg, section), field, value)
    info = parent_compatibility(cfg, "pa", "run-a")
    assert info["ok"] is False
    assert info["feature_matches"] is False
    assert info["hard_blocked"] is False
    assert "feature geometry/contract differs" in info["reason"]


def test_backbone_mismatch_hard_blocks(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import parent_compatibility

    _write_parent_bundle("pa", "run-a", backbone="mn01")
    info = parent_compatibility(_make_cfg("pb", backbone="mn04"), "pa", "run-a")
    assert info["ok"] is False
    assert info["hard_blocked"] is True
    assert "backbone differs" in info["reason"]


def test_encoder_loads_and_manifest_carries_grandparent(tmp_path, monkeypatch):
    """2-hop chain: run-a (random) → run-b (parent=run-a) → child init from run-b
    records run-a as the grandparent."""
    _patch_dirs(tmp_path, monkeypatch)
    from types import SimpleNamespace

    from sed.train.model import SedModel
    from sed.train.transfer import initialize_encoder_from_parent

    _write_parent_bundle("pa", "run-a")
    _write_parent_bundle("pa", "run-b", init_block={
        "mode": "parent", "parent_project": "pa", "parent_run": "run-a",
        "parent_recipe_signature": "sig-a",
    })

    cfg = _make_cfg("pb", initialization="parent",
                    parent_project="pa", parent_run="run-b")
    child = SedModel("mn01", num_classes=3, pretrained=False, n_bands=128, n_frames=100)
    before = {k: v.clone() for k, v in child.backbone.state_dict().items()}
    head_before = {k: v.clone() for k, v in child.head.state_dict().items()}

    info = initialize_encoder_from_parent(SimpleNamespace(model=child), cfg)
    assert info["mode"] == "parent"
    assert info["parent_project"] == "pa" and info["parent_run"] == "run-b"
    assert info["parent_initialization_mode"] == "parent"
    assert info["grandparent_project"] == "pa"
    assert info["grandparent_run"] == "run-a"
    assert info["grandparent_signature"] == "sig-a"

    parent_state = torch.load(
        ProjectLayout("pa").model_dir("run-b") / "checkpoint_best_epoch-000.ckpt",
        map_location="cpu", weights_only=False,
    )["state_dict"]
    changed = False
    for key, value in child.backbone.state_dict().items():
        expected = parent_state["model.backbone." + key]
        assert torch.equal(value, expected), f"encoder weight {key} not transferred"
        if not torch.equal(value, before[key]):
            changed = True
    assert changed, "at least one encoder tensor must differ from the fresh init"
    for key, value in child.head.state_dict().items():
        assert torch.equal(value, head_before[key]), "the classifier head must stay fresh"


def test_head_warm_starts_per_class_name_and_leaves_new_classes_random(
    tmp_path, monkeypatch,
):
    """Shared classes reuse the parent's fitted classifier row.

    The child's encoder IS the parent's encoder, so a class the parent already
    learned has its solution sitting in the same embedding space. Re-deriving
    that row from random is what makes a chained fine-tune spend its whole epoch
    budget getting back to where it started.
    """
    _patch_dirs(tmp_path, monkeypatch)
    from types import SimpleNamespace

    from sed.train.model import SedModel
    from sed.train.transfer import initialize_encoder_from_parent

    _write_parent_bundle("pa", "run-a", num_classes=2)
    run_dir = ProjectLayout("pa").model_dir("run-a")
    (run_dir / "classes.json").write_text(
        json.dumps(["_negative", "clink_wine"]), encoding="utf-8",
    )

    cfg = _make_cfg("pb", initialization="parent",
                    parent_project="pa", parent_run="run-a")
    child_classes = ["_negative", "clink_wine", "clink_tumbler"]
    child = SedModel("mn01", num_classes=len(child_classes), pretrained=False,
                     n_bands=128, n_frames=100)
    head_before = {k: v.clone() for k, v in child.head.state_dict().items()}

    info = initialize_encoder_from_parent(
        SimpleNamespace(model=child, classes=child_classes), cfg,
    )
    transfer = info["head_transfer"]
    assert transfer["transferred_classes"] == ["_negative", "clink_wine"]
    assert transfer["new_classes"] == ["clink_tumbler"]

    parent_state = torch.load(
        run_dir / "checkpoint_best_epoch-000.ckpt", map_location="cpu", weights_only=False,
    )["state_dict"]
    linear = child.head[-1]
    for child_row, parent_row in ((0, 0), (1, 1)):
        assert torch.equal(linear.weight[child_row], parent_state["model.head.1.weight"][parent_row])
        assert torch.equal(linear.bias[child_row], parent_state["model.head.1.bias"][parent_row])
    # The class the parent never saw keeps the fresh initialization.
    assert torch.equal(linear.weight[2], head_before["1.weight"][2])


def test_head_map_seeds_renamed_classes_from_one_parent_row(tmp_path, monkeypatch):
    """The cross-stage case: a pretraining run labels the umbrella event and the
    venue run labels variants of it, so exact matching warm-starts nothing.

    Without a map every target row starts random at a fine-tune learning rate,
    which is the observed cause of collapsed transfer runs.
    """
    _patch_dirs(tmp_path, monkeypatch)
    from types import SimpleNamespace

    from sed.train.model import SedModel
    from sed.train.transfer import initialize_encoder_from_parent

    _write_parent_bundle("pa", "run-a", num_classes=2)
    run_dir = ProjectLayout("pa").model_dir("run-a")
    (run_dir / "classes.json").write_text(
        json.dumps(["_negative", "cheers"]), encoding="utf-8",
    )

    cfg = _make_cfg("pb", initialization="parent",
                    parent_project="pa", parent_run="run-a")
    cfg.training.transfer_head_map = {"riedel_wine": "cheers", "ikea_wine": "cheers"}
    child_classes = ["_negative", "riedel_wine", "ikea_wine"]
    child = SedModel("mn01", num_classes=len(child_classes), pretrained=False,
                     n_bands=128, n_frames=100)

    info = initialize_encoder_from_parent(
        SimpleNamespace(model=child, classes=child_classes), cfg,
    )
    transfer = info["head_transfer"]
    assert transfer["sources"] == {
        "_negative": "exact",
        "riedel_wine": "mapped:cheers",
        "ikea_wine": "mapped:cheers",
    }
    assert transfer["new_classes"] == []

    parent_state = torch.load(
        run_dir / "checkpoint_best_epoch-000.ckpt", map_location="cpu", weights_only=False,
    )["state_dict"]
    source_row = parent_state["model.head.1.weight"][1]
    linear = child.head[-1]
    # Both mapped rows start from the parent's row but are not bit-identical:
    # duplicated rows get a small deterministic jitter.
    for row in (1, 2):
        assert torch.allclose(linear.weight[row], source_row, atol=source_row.std() * 0.2)
        assert torch.equal(linear.bias[row], parent_state["model.head.1.bias"][1])
    assert not torch.equal(linear.weight[1], linear.weight[2])
    # An exact match is still copied verbatim — jitter applies only to reuse.
    assert torch.equal(linear.weight[0], parent_state["model.head.1.weight"][0])


def test_head_map_jitter_is_seeded_by_the_training_seed(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    import torch as _torch

    from sed.train.transfer import _warm_start_head

    parent_names = ["_negative", "cheers"]
    child_names = ["_negative", "a", "b"]
    state = {
        "model.head.1.weight": _torch.randn(2, 8),
        "model.head.1.bias": _torch.randn(2),
    }
    head_map = {"a": "cheers", "b": "cheers"}

    def run(seed):
        head = _torch.nn.Sequential(_torch.nn.Dropout(0.0), _torch.nn.Linear(8, 3))
        _warm_start_head(head, state, parent_names, child_names,
                         head_map=head_map, seed=seed)
        return head[-1].weight.detach().clone()

    assert _torch.equal(run(42), run(42))
    assert not _torch.equal(run(42), run(7))


def test_head_map_ignores_a_source_the_parent_does_not_have(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from sed.train.transfer import resolve_head_sources

    sources = resolve_head_sources(
        ["_negative", "cheers"], ["_negative", "riedel_wine"], {"riedel_wine": "nope"},
    )
    assert sources == {"_negative": "exact", "riedel_wine": "random"}


def test_suggest_head_map_only_proposes_defensible_rows():
    from sed.train.transfer import suggest_head_map

    parent = ["_negative", "cheers", "clap"]
    # Unrelated names get no suggestion — the operator maps them by hand rather
    # than being handed a wrong parent row silently.
    assert suggest_head_map(parent, ["_negative", "riedel_wine"]) == {}
    # An umbrella name contained in the child's is the shape this pipeline makes.
    assert suggest_head_map(parent, ["cheers_riedel_wine"]) == {"cheers_riedel_wine": "cheers"}
    # Separator/case differences are the same class.
    assert suggest_head_map(["Cheers-Loud"], ["cheers_loud"]) == {"cheers_loud": "Cheers-Loud"}


def test_head_warm_start_can_be_switched_off(tmp_path, monkeypatch):
    _patch_dirs(tmp_path, monkeypatch)
    from types import SimpleNamespace

    from sed.train.model import SedModel
    from sed.train.transfer import initialize_encoder_from_parent

    _write_parent_bundle("pa", "run-a", num_classes=2)
    (ProjectLayout("pa").model_dir("run-a") / "classes.json").write_text(
        json.dumps(["_negative", "clink_wine"]), encoding="utf-8",
    )
    cfg = _make_cfg("pb", initialization="parent",
                    parent_project="pa", parent_run="run-a")
    cfg.training.transfer_head_init = "random"

    child = SedModel("mn01", num_classes=2, pretrained=False, n_bands=128, n_frames=100)
    head_before = {k: v.clone() for k, v in child.head.state_dict().items()}
    info = initialize_encoder_from_parent(
        SimpleNamespace(model=child, classes=["_negative", "clink_wine"]), cfg,
    )
    assert info["head_transfer"]["transferred_classes"] == []
    for key, value in child.head.state_dict().items():
        assert torch.equal(value, head_before[key])


def test_recipe_select_endpoint_persists_parent_and_adopts_feature(tmp_path, monkeypatch):
    """The Model tab flow end-to-end: select 'from previous run', adopt the
    parent's feature contract, and freeze both into config.toml."""
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    _patch_dirs(tmp_path, monkeypatch)
    from sed.config import load_config, write_project_config_template
    from sed.data.inspect_server import make_handler

    _write_parent_bundle("pa", "run-a", sample_rate=16000, n_mels=64)

    layout = ProjectLayout("pc")
    layout.create()
    write_project_config_template(layout, name="pc")
    cfg = load_config("pc")

    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    handler = make_handler(cfg, "pc", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        candidates = json.loads(urllib.request.urlopen(
            f"{base}/api/model-parent-candidates?backbone=mn01&task=clip", timeout=30
        ).read())
        assert candidates["ok"] is True
        keys = {(c["parent_project"], c["parent_run"]) for c in candidates["candidates"]}
        assert ("pa", "run-a") in keys

        body = json.dumps({
            "architecture_id": "efficientat-mn", "task": "clip", "weight_id": "",
            "backbone": "mn01", "initialization": "parent", "params": {},
            "parent_project": "pa", "parent_run": "run-a",
            "adopt_parent_feature": True, "feature": None,
        }).encode()
        req = urllib.request.Request(
            f"{base}/api/model-recipe/select", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert resp["ok"] is True, resp
        assert resp["recipe"]["initialization"] == "parent"
        assert resp["recipe"]["parent_project"] == "pa"
        assert resp["compatibility"]["ok"] is True
        assert resp["adopted_feature"] is not None

        saved = load_config("pc")
        assert saved.model.initialization == "parent"
        assert saved.model.parent_project == "pa"
        assert saved.model.parent_run == "run-a"
        assert saved.model.backbone == "mn01"
        assert saved.model.pretrained is False
        assert saved.model.weight_id == ""
        # Adopted feature contract: the parent's frozen geometry is now this
        # project's [features] + model.sample_rate.
        assert saved.model.sample_rate == 16000
        assert saved.features.n_mels == 64

        # The picker also carries the parent's class list, so the head-map
        # editor can render before anything is trained.
        parent = next(
            c for c in candidates["candidates"]
            if (c["parent_project"], c["parent_run"]) == ("pa", "run-a")
        )
        assert "classes" in parent and "head_sources" in parent

        # Head-map rows persist with the recipe, and a source the parent does
        # not have is refused rather than silently dropped at train time.
        target_classes = list(load_config("pc").dataset.target_classes)
        parent_classes = parent["classes"]
        child = next((c for c in target_classes if c not in parent_classes), None)
        source = next((c for c in parent_classes if c not in target_classes), None)
        if child and source:
            def select(head_map):
                payload = json.dumps({
                    "architecture_id": "efficientat-mn", "task": "clip", "weight_id": "",
                    "backbone": "mn01", "initialization": "parent", "params": {},
                    "parent_project": "pa", "parent_run": "run-a",
                    "adopt_parent_feature": True, "feature": None,
                    "transfer_head_map": head_map,
                }).encode()
                request = urllib.request.Request(
                    f"{base}/api/model-recipe/select", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                try:
                    return json.loads(urllib.request.urlopen(request, timeout=60).read())
                except urllib.error.HTTPError as exc:
                    return json.loads(exc.read())

            good = select({child: source})
            assert good["ok"] is True, good
            assert good["recipe"]["transfer_head_map"] == {child: source}
            assert good["head_sources"][child] == f"mapped:{source}"
            assert load_config("pc").training.transfer_head_map == {child: source}

            bad = select({child: "__not_a_parent_class__"})
            assert bad["ok"] is False
            assert "does not have" in bad["error"]
            # The rejected write must not have replaced the good rows.
            assert load_config("pc").training.transfer_head_map == {child: source}
    finally:
        httpd.shutdown()
