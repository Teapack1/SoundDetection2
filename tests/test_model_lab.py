"""Model Lab recipe, transfer, and temporal-encoder contracts."""

from __future__ import annotations


import pytest


def test_curated_catalog_exposes_sources_and_kws_templates():
    from sed.train.model_lab import catalog_payload

    catalog = catalog_payload()
    entries = {entry["id"]: entry for entry in catalog["architectures"]}
    assert "efficientat-mn" in entries
    assert "kws-bcresnet" in entries
    for model_id in (
        "transient-cnn-gap", "transient-resnet", "clink-cnn-v2",
        "ced-tiny", "ced-mini", "ced-small", "ced-base",
        "efficient-sed-fmn10", "efficient-sed-fmn20", "legacy-smallervggnet",
    ):
        assert model_id in entries
    # Robust ImageNet timm CNN backbones are offered for spectrograms.
    for timm_id in (
        "timm-resnet18", "timm-resnet34", "timm-resnet50",
        "timm-efficientnet-b0", "timm-efficientnet-b1",
        "timm-convnext-nano", "timm-convnext-tiny",
    ):
        assert timm_id in entries
    # The retained secondary baselines remain archived and hidden by default.
    for archived_id in (
        "timm-mobilenetv3-small", "kws-tcresnet", "kws-dscnn",
    ):
        assert archived_id not in entries
    for removed_id in (
        "hf-ast-base", "tiny-audio-transformer", "passt-lite",
        "cnn-conformer-tiny", "transient-gridnet",
    ):
        assert removed_id not in entries
    # The raw-waveform externals were removed — every listed model must build.
    assert "convnext-audio-tiny" not in entries
    assert "passt-external" not in entries
    weights = {weight["id"] for weight in entries["efficientat-mn"]["weights"]}
    assert "efficientat:mn10_as_mels_256" in weights
    assert entries["ced-tiny"]["export_supported"] is False
    assert entries["ced-tiny"]["trusted"] is True
    assert entries["ced-tiny"]["default_feature"]["frontend"] == "ced"
    assert entries["ced-tiny"]["default_feature"]["n_mels"] == 64
    assert "clink-cnn" not in entries, "ClinkCNN-S was removed from the catalog"
    v2 = entries["clink-cnn-v2"]
    assert not v2["weights"]
    assert "no_upstream_weights" in v2["tags"]
    assert v2["archived"] is False
    assert v2["default_feature"] == {
        "feature_type": "mel", "frontend": "efficientat", "sample_rate": 44100,
        "n_fft": 1024, "win_length": 882, "hop_length": 220, "n_mels": 128,
        "fmin": 80, "fmax": 20000, "window_seconds": 0.4,
        "compression": "log",
        "pcen": False, "log_power": True, "stft_engine": "librosa",
        "db_norm": False, "colormap": "none", "image_size": 96,
        "standardize": False, "standardize_mean": 0.0, "standardize_std": 1.0,
        # The EfficientAT affine named for what it is — a fixed global z-score —
        # rather than appearing as "no normalization".
        "scaling_method": "zscore", "scaling_scope": "global",
        "scaling_source": "fixed", "scaling_stats": {"mean": [-4.5], "std": [5.0]},
        "source": "ClinkCNN-v2 0.4 s feature contract",
    }
    assert "task_specific_candidate" in v2["tags"]
    assert "recommended" not in v2["tags"]
    assert catalog["tag_definitions"]["task_specific_candidate"]["label"] == "Task-specific candidate"
    assert entries["efficient-sed-fmn10"]["default_feature"]["n_mels"] == 128
    assert entries["efficient-sed-fmn10"]["export_supported"] is False
    # EfficientSED remains runtime-compatible with both heads for old/direct
    # recipes, while new Model-tab selection presents it only for frame SED.
    for model_id in ("efficient-sed-fmn10", "efficient-sed-fmn20"):
        assert set(entries[model_id]["capabilities"]) == {"clip", "frame"}
        assert entries[model_id]["selection_tasks"] == ["frame"]
    def offered_for(task):
        return {
            model_id for model_id, entry in entries.items()
            if task in entry["selection_tasks"]
        }
    assert "efficient-sed-fmn10" not in offered_for("clip")
    assert "efficient-sed-fmn20" not in offered_for("clip")
    assert {"efficient-sed-fmn10", "efficient-sed-fmn20"} <= offered_for("frame")
    assert any(b["name"] == "mn01" for b in catalog["scratch_backbones"])


def test_catalog_tags_are_defined_capability_facts_not_static_rankings():
    from sed.train.model_lab import CATALOG_TAG_DEFINITIONS, catalog_payload

    catalog = catalog_payload(keep_ids={"timm-mobilenetv3-small", "kws-tcresnet", "kws-dscnn"})
    tags = catalog["tag_definitions"]
    assert tags == CATALOG_TAG_DEFINITIONS
    for tag_id, meta in tags.items():
        assert tag_id
        assert set(meta) >= {"label", "description", "tone"}
        assert meta["description"]
        assert "recommended" not in meta["label"].lower()
    for architecture in catalog["architectures"]:
        assert set(architecture["tags"]) <= set(tags)
        assert not any("recommended" in tag.lower() for tag in architecture["tags"])
    entries = {entry["id"]: entry for entry in catalog["architectures"]}
    assert entries["efficientat-mn"]["tags"] == [
        "established_baseline", "audio_set_weights", "short_clip_native", "openvino_deployable",
    ]
    assert entries["efficientat-dymn"]["tags"] == [
        "research_candidate", "audio_set_weights", "short_clip_native", "pytorch_runtime_only",
    ]
    assert "archived" in entries["kws-tcresnet"]["tags"]


def test_catalog_exposes_dymn_family():
    from sed.train.model_lab import architecture_by_id, catalog_payload, weight_by_id
    from sed.train.backbones import get_spec

    catalog = catalog_payload()
    entries = {entry["id"]: entry for entry in catalog["architectures"]}
    assert "efficientat-dymn" in entries
    dymn = entries["efficientat-dymn"]
    assert dymn["family"] == "efficientat"
    assert set(dymn["capabilities"]) == {"clip", "frame"}
    assert dymn["trusted"] is True
    assert dymn["export_supported"] is False
    weights = {weight["id"] for weight in dymn["weights"]}
    assert {"efficientat:dymn04_as", "efficientat:dymn10_as", "efficientat:dymn20_as"} <= weights
    assert get_spec("dymn10_as").export_supported is False
    # The weight resolves and carries the AudioSet log-mel contract.
    spec = architecture_by_id("efficientat-dymn")
    weight = weight_by_id(spec.id, "efficientat:dymn10_as")
    assert weight is not None
    assert weight.expected_feature_contract["features.n_mels"] == 128


def test_selection_tasks_default_to_runtime_capabilities():
    from sed.train.model_lab import ArchitectureSpec

    spec = ArchitectureSpec("dual", "Dual", "local", "custom", "x:y", "test", ("clip", "frame"))
    assert spec.selection_tasks == ("clip", "frame")
    assert spec.to_payload()["selection_tasks"] == ["clip", "frame"]


def test_recipe_param_validation_and_signature_stability():
    from sed.train.model_lab import (
        ModelRecipe,
        architecture_by_id,
        recipe_signature,
        validate_recipe_params,
    )

    spec = architecture_by_id("kws-bcresnet")
    params = validate_recipe_params(spec, {"width_mult": "1.5", "time_stride": "1", "depth": 5})
    assert params == {"width_mult": 1.5, "time_stride": 1, "depth": 5}
    with pytest.raises(ValueError, match="time_stride"):
        validate_recipe_params(spec, {"time_stride": 3})
    with pytest.raises(ValueError, match="locked"):
        validate_recipe_params(architecture_by_id("timm-convnext-tiny"), {"pretrained": False})
    first = ModelRecipe(1, "kws-bcresnet", "kws-bcresnet", "clip", "", "random", params)
    second = ModelRecipe(1, "kws-bcresnet", "kws-bcresnet", "clip", "other", "upstream", params)
    assert recipe_signature(first) == recipe_signature(second)


def test_recipe_param_defaults_are_materialized():
    from sed.train.model_lab import architecture_by_id, validate_recipe_params

    assert validate_recipe_params(architecture_by_id("kws-bcresnet"), {}) == {
        "width_mult": 1.0,
        "time_stride": 1,
        "depth": 4,
    }
    clink_v2 = architecture_by_id("clink-cnn-v2")
    assert validate_recipe_params(clink_v2, {}) == {}
    with pytest.raises(ValueError, match="fixed architecture"):
        validate_recipe_params(clink_v2, {"depth": 5})
    assert clink_v2.to_payload()["params"] == ()
    assert validate_recipe_params(architecture_by_id("legacy-smallervggnet"), {}) == {
        "dense_units": 1024,
        "conv_dropout": 0.25,
        "dense_dropout": 0.5,
    }
    assert validate_recipe_params(architecture_by_id("timm-convnext-tiny"), {}) == {
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
    }


def test_catalog_constructor_contract_is_declared_typed_and_documented():
    from sed.train.model_lab import catalog_payload, validate_recipe_params, architecture_by_id

    supported_types = {"bool", "int", "float", "choice", "string", "int_list"}
    entries = {entry["id"]: entry for entry in catalog_payload(keep_ids={
        "timm-mobilenetv3-small", "kws-tcresnet", "kws-dscnn",
    })["architectures"]}
    for model_id, entry in entries.items():
        assert "advanced" not in entry
        assert entry["constructor"]["params"] == list(entry["params"])
        names = []
        for param in entry["params"]:
            names.append(param["name"])
            assert param["type"] in supported_types
            assert param.get("help", "").strip(), (model_id, param["name"])
        assert len(names) == len(set(names)), model_id
        validate_recipe_params(architecture_by_id(model_id), {})

    mn_names = {p["name"] for p in entries["efficientat-mn"]["params"]}
    assert "head_type" not in mn_names
    assert "multihead_attention_heads" not in mn_names
    strides = next(p for p in entries["efficientat-mn"]["params"] if p["name"] == "strides")
    assert strides["type"] == "int_list"
    assert strides["length"] == 4
    assert strides["item_choices"] == [1, 2]
    for fixed_id in (
        "clink-cnn-v2", "ced-tiny", "ced-mini", "ced-small", "ced-base",
        "efficient-sed-fmn10", "efficient-sed-fmn20",
    ):
        assert entries[fixed_id]["constructor"]["fixed"] is True
        assert not entries[fixed_id]["params"]


def test_constructor_validation_rejects_unknown_and_bad_list_shapes():
    from sed.train.model_lab import architecture_by_id, validate_recipe_params

    mn = architecture_by_id("efficientat-mn")
    with pytest.raises(ValueError, match="unsupported constructor parameter"):
        validate_recipe_params(mn, {"head_type": "attention"})
    with pytest.raises(ValueError, match="exactly 4"):
        validate_recipe_params(mn, {"strides": [2, 2, 2]})
    with pytest.raises(ValueError, match="values must be one of"):
        validate_recipe_params(mn, {"strides": [2, 2, 3, 2]})
    assert validate_recipe_params(mn, {"strides": "1,2,2,1"}) == {
        "strides": [1, 2, 2, 1],
    }


def test_param_guidance_scopes_recommendations_by_init_and_dataset_structure():
    """A recommendation must react to how the encoder starts and how redundant
    the corpus is, not only to its row count."""
    from sed.train.model_lab import architecture_by_id, param_guidance

    spec = architecture_by_id("timm-resnet34")

    def rec(dataset, init):
        rows = {r["name"]: r for r in param_guidance(spec, {}, dataset=dataset, init=init)}
        return rows["drop_path_rate"]

    small = {"rows": 1200, "min_class_groups": 40}
    # Same dataset, different starting weights → different schedules.
    assert rec(small, "upstream")["recommended"] == pytest.approx(0.0)
    assert rec(small, "random")["recommended"] == pytest.approx(0.0)
    mid = {"rows": 5000, "min_class_groups": 40}
    assert rec(mid, "upstream")["recommended"] == pytest.approx(0.05)
    assert rec(mid, "random")["recommended"] == pytest.approx(0.1)

    # Same row count, different diversity: few independent recording groups make
    # a big corpus behave like a redundant one, so it is not advised as "large".
    diverse = rec({"rows": 50000, "min_class_groups": 300}, "upstream")
    redundant = rec({"rows": 50000, "min_class_groups": 6}, "upstream")
    assert diverse["recommended"] == pytest.approx(0.1)
    assert redundant["recommended"] == pytest.approx(0.05)
    assert "recording groups" in redundant["why"]

    # The scope names the facts the band was chosen from, so the page can show
    # the recommendation is not a constant.
    assert "pretrained" in diverse["scope"]
    assert "scratch" in rec(mid, "random")["scope"]

    # An unmeasured dataset still resolves — constraints needing missing facts
    # are skipped rather than failing.
    bare = rec({}, "upstream")
    assert bare["recommended"] is not None


def test_catalog_exposes_fixed_regularization_and_per_init_guidance():
    """ClinkCNN-v2 has no constructor params, only built-in regularization. The
    Model tab must still be able to show it, or it looks like it has none."""
    from sed.train.model_lab import INIT_MODES, catalog_payload, constructor_contract, architecture_by_id

    contract = constructor_contract(architecture_by_id("clink-cnn-v2"))
    assert contract["params"] == []
    fixed = {row["name"]: row for row in contract["regularization"]}
    assert set(fixed) == {"pool_dropout", "drop_path"}
    assert fixed["pool_dropout"]["location"] == "pooled_embedding"
    assert fixed["drop_path"]["mechanism"] == "stochastic_depth"

    payload = catalog_payload(train_rows=1200, dataset={"rows": 1200, "min_class_groups": 8})
    entries = {entry["id"]: entry for entry in payload["architectures"]}
    resnet = entries["timm-resnet34"]
    assert set(resnet["guidance_by_init"]) == set(INIT_MODES)
    # `guidance` stays as the pretrained alias for callers that do not care.
    assert resnet["guidance"] == resnet["guidance_by_init"]["upstream"]
    assert payload["dataset_shape"]["min_class_groups"] == 8


def test_dropout_auto_distinguishes_frozen_and_trainable_final_embedding():
    from sed.config import Config
    from sed.train.model_lab import (
        architecture_dropout_info,
        effective_head_dropout,
        materialize_training_config,
    )

    frozen = architecture_dropout_info("clink-cnn-v2", train_depth="head")
    assert frozen["recommended_head_dropout"] == pytest.approx(0.2)
    assert not any(row["active"] for row in frozen["rows"])

    last_pool = architecture_dropout_info("clink-cnn-v2", train_depth="last_block")
    assert last_pool["recommended_head_dropout"] == pytest.approx(0.0)
    assert next(row for row in last_pool["rows"] if row["name"] == "pool_dropout")["active"]
    assert not next(row for row in last_pool["rows"] if row["name"] == "drop_path")["active"]

    full = architecture_dropout_info("clink-cnn-v2", train_depth="full")
    assert full["recommended_head_dropout"] == pytest.approx(0.0)
    assert all(row["active"] for row in full["rows"])

    cfg = Config()
    cfg.project.id = "p"
    cfg.model.backbone = "clink-cnn-v2"
    cfg.model.recipe_id = "clink-cnn-v2"
    cfg.model.pretrained = False
    cfg.model.initialization = "parent"
    cfg.model.parent_project = "p"
    cfg.model.parent_run = "seed"
    cfg.model.params = {}
    cfg.model.params_contract_version = 2
    cfg.training.head_dropout_auto = True
    cfg.training.fine_tune_policy = "head"
    assert effective_head_dropout(cfg) == pytest.approx(0.2)
    frozen_cfg = materialize_training_config(cfg)
    assert frozen_cfg.training.head_dropout == pytest.approx(0.2)
    assert frozen_cfg.training.head_dropout_auto is False

    unfrozen_cfg = cfg.model_copy(update={
        "training": cfg.training.model_copy(update={"fine_tune_policy": "full"}),
    })
    assert effective_head_dropout(unfrozen_cfg) == pytest.approx(0.0)
    assert materialize_training_config(unfrozen_cfg).training.head_dropout == pytest.approx(0.0)
    manual_cfg = unfrozen_cfg.model_copy(update={
        "training": unfrozen_cfg.training.model_copy(update={
            "head_dropout_auto": False, "head_dropout": 0.45,
        }),
    })
    assert effective_head_dropout(manual_cfg) == pytest.approx(0.45)
    byol_cfg = cfg.model_copy(update={
        "training": cfg.training.model_copy(update={"pretraining_objective": "byol"}),
    })
    assert effective_head_dropout(byol_cfg) == pytest.approx(0.0)
    from sed.data.inspect_server import _train_architecture_dropout

    byol_stack = _train_architecture_dropout(byol_cfg)
    assert byol_stack["classifier"]["active"] is False
    assert next(row for row in byol_stack["rows"] if row["name"] == "pool_dropout")["active"]


@pytest.mark.parametrize(
    "architecture_id,backbone,params,expected",
    [
        ("legacy-smallervggnet", "legacy-smallervggnet", {}, 0.0),
        ("timm-resnet18", "resnet18", {"drop_rate": 0.3}, 0.0),
        ("timm-resnet18", "resnet18", {"drop_rate": 0.0}, 0.2),
        ("transient-resnet", "transient-resnet", {"dropout": 0.3}, 0.2),
    ],
)
def test_dropout_auto_is_model_aware(architecture_id, backbone, params, expected):
    from sed.config import Config
    from sed.train.model_lab import effective_head_dropout

    cfg = Config.model_validate({
        "model": {
            "backbone": backbone, "recipe_id": architecture_id,
            "pretrained": False, "initialization": "parent",
            "parent_project": "p", "parent_run": "seed", "params": params,
            "params_contract_version": 2,
        },
        "training": {"head_dropout_auto": True, "fine_tune_policy": "full"},
    })
    assert effective_head_dropout(cfg) == pytest.approx(expected)


def test_explicit_legacy_head_dropout_remains_manual():
    from sed.config import TrainingCfg

    assert TrainingCfg.model_validate({}).head_dropout_auto is True
    explicit = TrainingCfg.model_validate({"head_dropout": 0.35})
    assert explicit.head_dropout == pytest.approx(0.35)
    assert explicit.head_dropout_auto is False


def test_efficientat_base_backbone_strips_checkpoint_variant_suffixes():
    """Checkpoint variants (hop/mels/fc) are the same network as their base
    width — a random-init recipe must reduce to the width-only name so it never
    implies a feature contract of weights that are not loaded."""
    from sed.train.model_lab import efficientat_base_backbone

    assert efficientat_base_backbone("mn10_as_mels_256") == "mn10_as"
    assert efficientat_base_backbone("mn10_as_hop_5") == "mn10_as"
    assert efficientat_base_backbone("mn10_as_fc") == "mn10_as"
    assert efficientat_base_backbone("mn10_as") == "mn10_as"
    assert efficientat_base_backbone("mn40_as") == "mn40_as"
    assert efficientat_base_backbone("dymn20_as_mAP_493") == "dymn20_as"
    assert efficientat_base_backbone("dymn10_as") == "dymn10_as"
    # Non-EfficientAT names pass through untouched.
    assert efficientat_base_backbone("convnext_tiny") == "convnext_tiny"
    assert efficientat_base_backbone("legacy-smallervggnet") == "legacy-smallervggnet"
    assert efficientat_base_backbone("") == ""


def test_fine_tune_policies_are_safe_and_scratch_trains_full():
    from sed.config import Config
    from sed.train.fine_tuning import available_fine_tune_policies, resolve_fine_tune_plan

    cfg = Config()
    cfg.model.pretrained = True
    cfg.training.fine_tune_policy = "last_two"
    cfg.training.epochs = 30
    plan = resolve_fine_tune_plan(cfg)
    assert plan.initial_depth == "head" and plan.target_depth == "last_n:2"
    assert plan.warmup_epochs == 3

    cfg.model.pretrained = False
    plan = resolve_fine_tune_plan(cfg)
    assert plan.policy == "full"
    assert plan.initial_depth == plan.target_depth == "full"
    assert plan.warmup_epochs == 0
    assert [p.id for p in available_fine_tune_policies(cfg, n_stages=4)] == ["full"]


def test_frame_capable_kws_model_returns_temporal_logits():
    torch = pytest.importorskip("torch")
    from sed.train.model import SedModel

    model = SedModel("kws-tcresnet", num_classes=3, pretrained=False,
                     n_bands=128, n_frames=80, params={"time_stride": 1}, task="frame")
    frames = model.forward_frames(torch.zeros(2, 1, 128, 80))
    assert frames.shape[0] == 2 and frames.shape[-1] == 3 and frames.shape[1] > 1
    assert model(torch.zeros(2, 1, 128, 80)).shape == (2, 3)


def test_frame_lit_module_projects_dense_targets_to_encoder_stride():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from sed.config import Config
    from sed.train.lit_module import SedLitModule

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})
    cfg.model.backbone = "kws-tcresnet"
    cfg.model.pretrained = False
    cfg.model.params = {"time_stride": 1}
    cfg.model.task = "frame"
    cfg.features.window_seconds = 0.4
    module = SedLitModule(cfg, ["_negative", "glass"])
    loss = module.training_step((torch.randn(2, 1, 128, 41), torch.zeros(2, 2, 41)), 0)
    assert torch.isfinite(loss)


def test_every_catalog_architecture_declares_a_default_feature():
    """Scratch runs show a recommended starting geometry (never a contract):
    every curated architecture must carry a default_feature in the payload."""
    from sed.train.model_lab import catalog_payload

    catalog = catalog_payload("")
    by_id = {a["id"]: a for a in catalog["architectures"]}
    for arch_id, arch in by_id.items():
        feature = arch.get("default_feature")
        assert isinstance(feature, dict) and feature, f"{arch_id} lacks default_feature"
        assert feature.get("feature_type"), arch_id
        assert feature.get("sample_rate"), arch_id

    mn = by_id["efficientat-mn"]["default_feature"]
    assert mn["feature_type"] == "mel"
    assert mn["sample_rate"] == 32000
    assert mn["n_mels"] in (64, 128)
    assert mn["window_seconds"] == 0.4
    assert mn["upstream_window_seconds"] == 1.0
    assert by_id["kws-bcresnet"]["default_feature"]["window_seconds"] == 0.4

    for arch_id, arch in by_id.items():
        if "clip" in (arch.get("selection_tasks") or []):
            assert arch["default_feature"]["window_seconds"] == 0.4, arch_id

    legacy = by_id["legacy-smallervggnet"]["default_feature"]
    assert legacy["feature_type"] == "linear"
    assert legacy["sample_rate"] == 44100
    assert legacy["colormap"] == "jet"
    assert legacy["image_size"] == 96


def test_export_supported_catalog_defaults_probe_successfully():
    pytest.importorskip("torch")
    from sed.audio.features import feature_band_count, feature_channels, feature_frame_count
    from sed.config import FeaturesCfg
    from sed.train.backbones import get_spec, probe_spec
    from sed.train.model_lab import (
        curated_architectures,
        recommended_feature_for,
        validate_recipe_params,
    )

    for architecture in curated_architectures():
        if not architecture.export_supported:
            continue
        if architecture.provider == "timm":
            pytest.importorskip("timm")
        feature = recommended_feature_for(architecture)
        assert feature is not None, architecture.id
        features = FeaturesCfg.model_validate(
            {k: v for k, v in feature.items() if k in FeaturesCfg.model_fields}
        )
        sample_rate = int(feature.get("sample_rate") or 32000)
        spec_name = architecture.source if architecture.provider in {"efficientat", "timm"} else architecture.id
        probe = probe_spec(
            get_spec(spec_name),
            n_bands=feature_band_count(features, sample_rate),
            n_frames=feature_frame_count(
                sample_rate=sample_rate,
                window_seconds=features.window_seconds,
                hop_length=features.hop_length,
                n_fft=features.n_fft,
                frontend=features.frontend,
                win_length=int(getattr(features, "win_length", 0) or 0) or None,
                stft_engine=str(getattr(features, "stft_engine", "librosa")),
            ),
            params=validate_recipe_params(architecture, {}),
            input_channels=feature_channels(features),
        )
        assert probe["feat_dim"] > 0, architecture.id
        assert probe["n_stages"] >= 1, architecture.id


def test_expected_contract_for_weight_resolves_hop_and_mel_variants():
    from sed.train.model_lab import expected_contract_for_backbone, expected_contract_for_weight

    hop5 = expected_contract_for_weight("efficientat:mn10_as_hop_5")
    assert hop5 is not None and hop5["features.hop_length"] == 160
    mels256 = expected_contract_for_weight("efficientat:mn10_as_mels_256")
    assert mels256 is not None and mels256["features.n_mels"] == 256
    assert expected_contract_for_weight("") is None
    assert expected_contract_for_weight("efficientat:nope") is None
    # Backbone-name resolution stays as the fallback path.
    assert expected_contract_for_backbone("mn10_as_hop_5")["features.hop_length"] == 160


def test_model_fit_report_prefers_weight_contract_over_backbone_name():
    from types import SimpleNamespace

    from sed.audio.features import model_fit_report

    # A 5 ms-hop feature scored against the hop-5 checkpoint is on-contract...
    feats = SimpleNamespace(
        feature_type="mel", frontend="efficientat", n_fft=1024, win_length=800,
        hop_length=160, n_mels=128, fmin=0, fmax=15000, window_seconds=1.0,
        pcen=False, stft_engine="librosa", colormap="none", image_size=96,
    )
    fit = model_fit_report(
        feats, "mn10_as", True, 32000, weight_id="efficientat:mn10_as_hop_5",
    )
    assert fit["on_contract"] is True
    # ...but off-contract for the base mn10_as checkpoint (hop 320).
    fit_base = model_fit_report(feats, "mn10_as", True, 32000)
    assert fit_base["on_contract"] is False
