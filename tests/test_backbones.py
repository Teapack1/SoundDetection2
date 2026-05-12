"""Tests for the pluggable backbone adapters, freezing, and the model registry."""

from __future__ import annotations

import sys
import textwrap
import importlib.util

import pytest

torch = pytest.importorskip("torch")

from sed.train import backbones as B
from sed.train.backbones import (
    apply_train_depth,
    describe_train_depth,
    select_trainable_stages,
)
from sed.train.model import SedModel


def _trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_efficientat_builds_and_forwards():
    m = SedModel("mn10_as", num_classes=4, pretrained=False, n_bands=128, n_frames=100)
    out = m(torch.zeros(2, 1, 128, 100))
    assert out.shape == (2, 4)
    assert len(m.backbone_stages()) > 1


@pytest.mark.parametrize("backbone", ["mn01", "mn04", "mn05", "mn30"])
def test_efficientat_scratch_widths_are_declared_frame_capable(backbone):
    assert set(B.get_spec(backbone).task_capabilities) == {"clip", "frame"}


def test_probe_reports_batchnorm_layer_availability_without_a_second_build():
    probe = B.probe_spec(B.get_spec("mn01"), n_bands=128, n_frames=81)
    assert probe["batchnorm_layers"] > 0


def test_efficientat_scratch_frame_path_keeps_temporal_logits():
    model = SedModel(
        "mn01", num_classes=3, pretrained=False,
        n_bands=128, n_frames=81, task="frame",
    )
    frames = model.forward_frames(torch.zeros(2, 1, 128, 81))
    assert frames.shape[0] == 2
    assert frames.shape[-1] == 3
    assert frames.shape[1] > 1


def test_efficientat_scratch_can_follow_three_channel_feature_input():
    m = SedModel(
        "mn01",
        num_classes=2,
        pretrained=False,
        n_bands=96,
        n_frames=96,
        input_channels=3,
    )
    out = m(torch.zeros(2, 3, 96, 96))
    assert out.shape == (2, 2)
    assert m.input_channels == 3


def test_efficientat_without_available_weights_treats_pretrained_as_scratch_for_channels():
    m = SedModel(
        "mn01",
        num_classes=2,
        pretrained=True,
        n_bands=96,
        n_frames=96,
        input_channels=3,
    )
    assert m(torch.zeros(2, 3, 96, 96)).shape == (2, 2)


def test_efficientat_pretrained_rejects_three_channel_feature_input():
    with pytest.raises(ValueError, match="pretrained weights require a 1-channel"):
        SedModel(
            "mn10_as",
            num_classes=2,
            pretrained=True,
            n_bands=96,
            n_frames=96,
            input_channels=3,
        )


def test_legacy_vgg_can_follow_one_channel_spectrogram_feature_input():
    m = SedModel(
        "legacy-smallervggnet",
        num_classes=2,
        pretrained=False,
        n_bands=96,
        n_frames=96,
        input_channels=1,
    )
    out = m(torch.zeros(2, 1, 96, 96))
    assert out.shape == (2, 2)
    assert m.input_channels == 1


@pytest.mark.parametrize("backbone,feat_dim", [("dymn04_as", 384), ("dymn10_as", 960)])
def test_dymn_builds_and_forwards(backbone, feat_dim):
    # DyMN routes to the vendored Dynamic MobileNet get_model; scratch build avoids
    # any checkpoint download.
    adapter = B.resolve_adapter(backbone)
    assert adapter.spec.family == "efficientat"
    m = SedModel(backbone, num_classes=5, pretrained=False, n_bands=128, n_frames=100)
    out = m(torch.zeros(2, 1, 128, 100))
    assert out.shape == (2, 5)
    # The head infers feat_dim from a real forward; confirm it matches the variant.
    assert m.head[-1].in_features == feat_dim


def test_dymn_frame_features_have_multiple_frames():
    # Guards the adapter fix: DyMN has no `.features` Sequential, so frame features
    # must come from `_feature_forward` and keep a real time axis (not collapse to 1).
    fit = B.measure_model_fit("dymn10_as", n_bands=128, n_frames=100)
    assert fit["feature_map"] is not None
    assert fit["feature_map"][1] > 1
    m = SedModel("dymn10_as", num_classes=3, pretrained=False, n_bands=128, n_frames=100, task="frame")
    frames = m.forward_frames(torch.zeros(2, 1, 128, 100))
    assert frames.shape[0] == 2 and frames.shape[2] == 3
    assert frames.shape[1] > 1   # more than one temporal frame


@pytest.mark.parametrize("input_channels,n_bands,n_frames", [(1, 128, 100), (3, 96, 96)])
def test_timm_convnext_builds_and_forwards(input_channels, n_bands, n_frames):
    pytest.importorskip("timm")
    m = SedModel(
        "convnext_tiny",
        num_classes=3,
        pretrained=False,
        n_bands=n_bands,
        n_frames=n_frames,
        input_channels=input_channels,
    )
    out = m(torch.zeros(2, input_channels, n_bands, n_frames))
    assert out.shape == (2, 3)
    assert m.input_channels == input_channels
    names = [n for n, _ in m.backbone_stage_modules()]
    assert "stem" in names and any(n.startswith("stage") for n in names)


@pytest.mark.parametrize(
    "backbone,params",
    [
        ("transient-cnn-gap", {"width_mult": 0.5, "depth": 2}),
        ("transient-resnet", {"width_mult": 0.5, "depth": 2}),
    ],
)
@pytest.mark.parametrize("input_channels,n_bands,n_frames", [(1, 64, 41), (3, 96, 96)])
def test_short_audio_models_follow_feature_channels(backbone, params, input_channels, n_bands, n_frames):
    m = SedModel(
        backbone,
        num_classes=3,
        pretrained=False,
        n_bands=n_bands,
        n_frames=n_frames,
        input_channels=input_channels,
        params=params,
    )
    out = m(torch.zeros(2, input_channels, n_bands, n_frames))
    assert out.shape == (2, 3)
    assert m.input_channels == input_channels


def test_short_audio_frame_capable_model_returns_temporal_logits():
    m = SedModel(
        "transient-resnet",
        num_classes=3,
        pretrained=False,
        n_bands=64,
        n_frames=41,
        params={"width_mult": 0.5, "depth": 1},
        task="frame",
    )
    frames = m.forward_frames(torch.zeros(2, 1, 64, 41))
    assert frames.shape[0] == 2 and frames.shape[-1] == 3 and frames.shape[1] > 1


def test_clink_cnn_v2_stages_and_gradients():
    from sed.train.clink_cnn_v2 import EMBEDDING_DIM, N_BANDS, N_FRAMES

    model = SedModel(
        "clink-cnn-v2", num_classes=3, pretrained=False,
        n_bands=N_BANDS, n_frames=N_FRAMES,
    )
    names = [name for name, _ in model.backbone_stage_modules()]
    assert names == [
        "stem",
        "stage0.block0",
        "stage1.block0", "stage1.block1",
        "stage2.block0", "stage2.block1",
        "stage3.block0", "stage3.block1",
        "pool",
    ]
    grouped = [p for _, module in model.backbone_stage_modules() for p in module.parameters()]
    assert len({id(p) for p in grouped}) == len(list(model.backbone.parameters()))
    assert model.backbone.out_channels == EMBEDDING_DIM
    loss = model(torch.randn(2, 1, N_BANDS, N_FRAMES)).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.backbone.parameters())


def test_clink_pooling_dropout_is_inactive_when_frozen_and_active_when_unfrozen():
    """The auto classifier default depends on the real module mode, not merely
    on Clink having a Dropout layer in its topology."""
    from sed.train.clink_cnn_v2 import DropPath, N_BANDS, N_FRAMES

    model = SedModel(
        "clink-cnn-v2", num_classes=3, pretrained=False,
        n_bands=N_BANDS, n_frames=N_FRAMES,
    )
    pool_dropout = model.backbone.pool.projection[-1]
    drop_paths = [m for m in model.backbone.modules() if isinstance(m, DropPath)]

    _, frozen_modules = apply_train_depth(model, "head")
    model.train()
    for module in frozen_modules:
        module.eval()
    assert pool_dropout.training is False
    assert all(module.training is False for module in drop_paths)

    _, frozen_modules = apply_train_depth(model, "last_block")
    model.train()
    for module in frozen_modules:
        module.eval()
    assert pool_dropout.training is True
    assert all(module.training is False for module in drop_paths)

    _, frozen_modules = apply_train_depth(model, "full")
    model.train()
    for module in frozen_modules:
        module.eval()
    assert pool_dropout.training is True
    assert all(module.training is True for module in drop_paths)


@pytest.mark.parametrize(
    "backbone,expected_channels", [("efficient-sed-fmn10", 960), ("efficient-sed-fmn20", 1920)]
)
def test_efficient_sed_frame_mobilenets_build_and_keep_time(backbone, expected_channels):
    model = SedModel(
        backbone, num_classes=3, pretrained=False, n_bands=128, n_frames=40, task="frame",
    )
    features = model.backbone(torch.zeros(2, 1, 128, 40))
    assert features.shape == (2, expected_channels, 1, 10)
    frames = model.forward_frames(torch.zeros(2, 1, 128, 40))
    assert frames.shape == (2, 10, 3)


def test_efficient_sed_official_checkpoint_prefix_and_checksum_load_strictly(
    tmp_path, monkeypatch,
):
    from sed.train import efficient_sed as efficient_sed_module
    from sed.util import sha256_file

    source = efficient_sed_module.EfficientSEDfMN10(
        pretrained=False, n_bands=128, n_frames=40,
    )
    with torch.no_grad():
        source.features[0][0].weight.fill_(0.125)
    official_state = {
        f"net.model.{key}": value.clone()
        for key, value in source.state_dict().items()
    }

    cache_dir = tmp_path / "checkpoints"
    cache_dir.mkdir()
    checkpoint_name = "fmn10_test.pt"
    cache_path = cache_dir / checkpoint_name
    torch.save(official_state, cache_path)

    class TestCheckpointModel(efficient_sed_module.EfficientSEDfMN10):
        pass

    TestCheckpointModel.checkpoint_name = checkpoint_name
    TestCheckpointModel.checkpoint_sha256 = sha256_file(cache_path)
    monkeypatch.setattr(
        efficient_sed_module,
        "load_state_dict_from_url",
        lambda *_args, **_kwargs: official_state,
    )
    monkeypatch.setattr(efficient_sed_module.torch.hub, "get_dir", lambda: str(tmp_path))

    loaded = TestCheckpointModel(pretrained=True, n_bands=128, n_frames=40)
    assert torch.equal(loaded.features[0][0].weight, source.features[0][0].weight)


def test_ced_family_wrappers_select_their_official_hub_models():
    from sed.train.external_audio import (
        CEDBaseEncoder,
        CEDMiniEncoder,
        CEDSmallEncoder,
        CEDTinyEncoder,
    )

    assert CEDTinyEncoder.model_name == "mispeech/ced-tiny"
    assert CEDMiniEncoder.model_name == "mispeech/ced-mini"
    assert CEDSmallEncoder.model_name == "mispeech/ced-small"
    assert CEDBaseEncoder.model_name == "mispeech/ced-base"


def test_ced_external_adapter_missing_dependency_message_is_clear():
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed; avoid network/weight load in unit test")
    from sed.train.external_audio import CEDTinyEncoder

    with pytest.raises(ImportError, match="transformers.*experimental external model"):
        CEDTinyEncoder(in_chans=1, n_bands=64, pretrained=True)


def test_ced_base_model_unwraps_encoder_not_base_model_self():
    # transformers' PreTrainedModel.base_model returns the model ITSELF when the
    # base_model_prefix attribute is absent (CED's prefix is "ced" but its
    # submodule is "encoder") — unwrapping must not hand back the classifier.
    from sed.train.external_audio import CEDTinyEncoder

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)

        @property
        def base_model(self):
            return self

    clf = Classifier()
    assert CEDTinyEncoder._base_model(clf) is clf.encoder


def test_ced_load_encoder_prefers_classifier_checkpoint():
    # The mispeech/ced-tiny checkpoint stores CedForAudioClassification keys
    # (encoder.* + outputlayer.*); loading via AutoModel matches nothing and
    # silently returns a random encoder, so the classifier path must win.
    from sed.train.external_audio import CEDTinyEncoder

    encoder = torch.nn.Linear(4, 4)

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

    calls: list[str] = []

    class AutoClassifier:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append("classifier")
            return Classifier()

    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append("automodel")
            return torch.nn.Linear(4, 4)

    out = CEDTinyEncoder._load_encoder(AutoModel, AutoClassifier, "mispeech/ced-tiny", {})
    assert calls == ["classifier"]
    assert out is encoder


def test_custom_class_path_backbone(tmp_path):
    mod = tmp_path / "my_custom_net.py"
    mod.write_text(textwrap.dedent("""
        import torch.nn as nn
        class TinyNet(nn.Module):
            def __init__(self, in_chans=1, n_bands=64, n_frames=80, pretrained=False):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Conv2d(in_chans, 8, 3, stride=2, padding=1),
                    nn.Conv2d(8, 16, 3, stride=2, padding=1),
                ])
            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return x
    """))
    sys.path.insert(0, str(tmp_path))
    try:
        spec = B.BackboneSpec(name="tinynet", family="custom",
                              source="my_custom_net:TinyNet", input_channels=1)
        adapter = B.CustomAdapter(spec)
        bb = adapter.build_backbone(pretrained=False, n_bands=64, n_frames=80)
        emb = adapter.forward_features(bb, torch.zeros(1, 1, 64, 80))
        assert emb.dim() == 2  # pooled to (B, D)
        assert len(adapter.stages(bb)) >= 1
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("my_custom_net", None)


def test_direct_parameter_backbone_has_a_stage():
    class DirectBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

        def forward(self, x):
            return x

    adapter = B.BackboneAdapter(B.BackboneSpec(name="direct", family="custom"))
    backbone = DirectBackbone()
    stages = adapter.stage_modules(backbone)
    assert stages == [("backbone", backbone)]


def test_custom_adapter_rejects_declared_value_hidden_by_catch_all_kwargs():
    class InertKwargsBackbone(torch.nn.Module):
        def __init__(self, in_chans=1, **_ignored):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_chans, 4, 1)

        def forward(self, x):
            return self.conv(x)

    adapter = B.CustomAdapter(B.BackboneSpec(name="inert", family="custom"))
    with pytest.raises(ValueError, match="not accepted.*width_mult"):
        adapter._construct(
            InertKwargsBackbone,
            pretrained=False, n_bands=64, n_frames=80,
            params={"width_mult": 1.5}, input_channels=1,
        )


@pytest.mark.parametrize("backbone,nb,nf", [("mn10_as", 128, 100), ("convnext_tiny", 128, 100)])
def test_freeze_depths_monotonic(backbone, nb, nf):
    if backbone.startswith("convnext"):
        pytest.importorskip("timm")
    counts = {}
    for depth in ("head", "last_block", "full"):
        m = SedModel(backbone, num_classes=4, pretrained=False, n_bands=nb, n_frames=nf)
        info, frozen = apply_train_depth(m, depth)
        counts[depth] = info["trainable_params"]
        assert info["trainable_params"] == _trainable(m)
    assert counts["head"] < counts["last_block"] < counts["full"]
    # "head" trains only the classifier head.
    m = SedModel(backbone, num_classes=4, pretrained=False, n_bands=nb, n_frames=nf)
    apply_train_depth(m, "head")
    head_params = sum(p.numel() for p in m.head.parameters())
    assert _trainable(m) == head_params


def test_describe_train_depth_matches_apply_without_mutating():
    m = SedModel("mn01", num_classes=2, pretrained=False, n_bands=128, n_frames=100)
    before = [p.requires_grad for p in m.parameters()]
    expected = describe_train_depth(m, "last_n:3")
    assert [p.requires_grad for p in m.parameters()] == before

    info, _frozen = apply_train_depth(m, "last_n:3")
    assert info["trainable_params"] == expected["trainable_params"]
    assert info["trainable_stages"] == expected["trainable_stages"]
    assert _trainable(m) == expected["trainable_params"]


def test_select_trainable_stages():
    assert select_trainable_stages(10, "full") == 10
    assert select_trainable_stages(10, "head") == 0
    assert select_trainable_stages(10, "last_block") == 1
    assert select_trainable_stages(10, "last_n:3") == 3
    assert select_trainable_stages(4, "last_n:99") == 4
    # Legacy spellings from old saved configs must keep resolving.
    assert select_trainable_stages(10, "half") == 5
    assert select_trainable_stages(10, "last_n=2") == 2
    assert select_trainable_stages(10, "last4") == 4


def test_registry_register_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "USER_MODEL_DIR", tmp_path / "models")
    pytest.importorskip("timm")
    spec = B.BackboneSpec(name="convnext_atto_test", family="timm", source="convnext_atto")
    info = B.probe_spec(spec)
    spec.feat_dim = info["feat_dim"]
    B.save_user_spec(spec)
    names = [s["name"] for s in B.registry_payload()]
    assert "convnext_atto_test" in names
    assert not B.all_specs()["convnext_atto_test"].builtin
    assert B.delete_user_spec("convnext_atto_test")
    assert "convnext_atto_test" not in [s["name"] for s in B.registry_payload()]


def test_generic_model_fit_report():
    pytest.importorskip("timm")
    from types import SimpleNamespace

    from sed.audio.features import model_fit_report

    feats = SimpleNamespace(feature_type="mel", frontend="efficientat", n_fft=1024,
                            win_length=800, hop_length=320, n_mels=128, fmin=0,
                            fmax=15000, window_seconds=1.0, pcen=False)
    rep = model_fit_report(feats, "convnext_tiny", False, 32000)
    assert rep["feat_dim"] > 0
    assert rep["feature_map"] and rep["feature_map"][0] >= 1
    assert rep["verdicts"][0]["level"] in ("ok", "warn", "error")


def test_generic_model_fit_report_uses_selected_params():
    from types import SimpleNamespace

    from sed.audio.features import model_fit_report

    feats = SimpleNamespace(feature_type="mel", frontend="efficientat", n_fft=1024,
                            win_length=800, hop_length=320, n_mels=128, fmin=0,
                            fmax=15000, window_seconds=1.0, pcen=False,
                            stft_engine="librosa", colormap="none", image_size=96)
    stride1 = model_fit_report(feats, "kws-bcresnet", False, 32000,
                               params={"time_stride": 1})
    stride2 = model_fit_report(feats, "kws-bcresnet", False, 32000,
                               params={"time_stride": 2})
    assert stride1["feature_map"][1] > stride2["feature_map"][1]


def test_efficientat_param_count_width_dependent_and_variant_normalized():
    from sed.train.backbones import efficientat_param_count

    mn04 = efficientat_param_count("mn04_as")
    mn10 = efficientat_param_count("mn10_as")
    assert mn04 and mn10 and mn04 < mn10
    # Checkpoint-variant suffixes describe the training feature, not the
    # network — they normalize to the base width and share its count.
    assert efficientat_param_count("mn10_as_mels_256") == mn10
    assert efficientat_param_count("mn10_as_hop_5") == mn10


def test_head_dropout_is_a_training_knob_not_part_of_the_recipe():
    """Head dropout must be adjustable without making older runs incomparable.

    It is a ``TrainingCfg`` field, so it must not reach the recipe signature, and
    it must be an identity at eval so export is unaffected.
    """
    from sed.config import Config
    from sed.train.model_lab import recipe_from_config, recipe_signature

    model = SedModel("transient-resnet", num_classes=3, pretrained=False,
                     n_bands=128, n_frames=81, head_dropout=0.55)
    assert model.head[0].p == pytest.approx(0.55)
    # Identity in eval: the exported graph is unchanged by the value.
    model.eval()
    x = torch.randn(4, 1, 128, 81)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x))

    base = Config.model_validate({"model": {"backbone": "transient-resnet",
                                            "pretrained": False,
                                            "initialization": "random"}})
    other = base.model_copy(deep=True)
    other.training.head_dropout = 0.7
    assert recipe_signature(recipe_from_config(base)) == recipe_signature(recipe_from_config(other))


def test_head_dropout_is_validated():
    from sed.config import TrainingCfg

    assert TrainingCfg().head_dropout == pytest.approx(0.2)
    assert TrainingCfg(head_dropout=0.0).head_dropout == 0.0
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="head_dropout"):
            TrainingCfg(head_dropout=bad)
