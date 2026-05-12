"""Focused contract tests for the ClinkCNN-v2 encoder."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sed.train.clink_cnn_v2 import (  # noqa: E402
    EMBEDDING_DIM,
    N_BANDS,
    N_FRAMES,
    ClinkCNNv2,
    DepthwisePointwiseDownsample,
    FrequencyAwarePool,
    GlassBlock,
    PointwiseProjection,
)
from sed.train.model import SedModel  # noqa: E402


def test_forward_shape_exact_input_guards_and_gradients():
    # Use a non-native but supported geometry so this also proves that the
    # instance-specific input contract participates in autograd end-to-end.
    n_bands, n_frames = 64, 25
    net = ClinkCNNv2(n_bands=n_bands, n_frames=n_frames).train()
    x = torch.randn(2, 1, n_bands, n_frames, requires_grad=True)
    out = net(x)
    assert out.shape == (2, EMBEDDING_DIM)
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in net.parameters())

    for shape in ((2, n_bands, n_frames), (2, 2, n_bands, n_frames),
                  (2, 1, n_bands - 1, n_frames), (2, 1, n_bands, n_frames - 1)):
        with pytest.raises(ValueError, match="input must have shape"):
            net(torch.zeros(shape))


@pytest.mark.parametrize("kwargs", [
    {"pretrained": True},
    {"in_chans": 2},
    {"n_bands": 31},
    {"n_bands": 1_025},
    {"n_frames": 24},
    {"n_frames": 202},
])
def test_geometry_and_initialization_guards(kwargs):
    with pytest.raises(ValueError):
        ClinkCNNv2(**kwargs)


@pytest.mark.parametrize(
    ("n_bands", "n_frames", "final_map"),
    [
        (128, 81, (96, 16, 11)),
        (256, 81, (96, 32, 11)),
        (128, 161, (96, 16, 21)),
        (463, 81, (96, 58, 11)),
        (925, 81, (96, 116, 11)),
    ],
)
def test_frontend_geometries_keep_one_topology_and_expected_final_map(
    n_bands, n_frames, final_map,
):
    net = ClinkCNNv2(n_bands=n_bands, n_frames=n_frames).eval()

    assert sum(parameter.numel() for parameter in net.parameters()) == 241_025
    assert len(net.fine_tune_stages()) == 9
    assert net.frequency_coordinate.shape == (1, 1, n_bands, 1)
    assert net.frequency_coordinate[0, 0, 0, 0].item() == pytest.approx(-1.0)
    assert net.frequency_coordinate[0, 0, -1, 0].item() == pytest.approx(1.0)

    with torch.inference_mode():
        mapped = net.forward_map(torch.zeros(1, 1, n_bands, n_frames))
        embedding = net.pool(mapped)
    assert tuple(mapped.shape[1:]) == final_map
    assert embedding.shape == (1, EMBEDDING_DIM)


def test_input_planes_are_raw_positive_delta_and_frequency_coordinate():
    net = ClinkCNNv2().eval()
    raw = torch.zeros(2, 1, N_BANDS, N_FRAMES)
    raw[:, :, :, 1] = 2.0
    raw[:, :, :, 2] = 1.0
    planes = net.input_planes(raw)

    assert planes.shape == (2, 3, N_BANDS, N_FRAMES)
    assert torch.equal(planes[:, 0:1], raw)
    assert torch.all(planes[:, 1, :, 0] == 0.0)
    assert torch.all(planes[:, 1, :, 1] == 2.0)
    assert torch.all(planes[:, 1, :, 2:] == 0.0)
    coordinate = planes[:, 2]
    assert torch.all(coordinate[:, 0] == -1.0)
    assert torch.all(coordinate[:, -1] == 1.0)
    assert torch.all(coordinate[:, 1:, :] > coordinate[:, :-1, :])
    assert torch.equal(coordinate[0], coordinate[1])
    assert torch.equal(coordinate[..., 0], coordinate[..., -1])


def test_trunk_shape_schedule_and_block_primitives():
    net = ClinkCNNv2().eval()
    z = net.stem(net.input_planes(torch.randn(1, 1, N_BANDS, N_FRAMES)))
    assert tuple(z.shape[1:]) == (32, 128, 41)
    shapes = []
    for stage in net.stages:
        z = stage(z)
        shapes.append(tuple(z.shape[1:]))
    assert shapes == [
        (32, 64, 41),
        (64, 32, 21),
        (96, 16, 11),
        (96, 16, 11),
    ]

    blocks = [m for m in net.modules() if isinstance(m, GlassBlock)]
    assert len(blocks) == 7
    assert [b.expand.out_channels // b.expand.in_channels for b in blocks] == [2] * 7
    assert max(b.drop_path.probability for b in blocks) == pytest.approx(0.1)
    assert all(b.frequency_branch.kernel_size == (7, 1) for b in blocks)
    assert all(b.temporal_branch.kernel_size == (1, 7) for b in blocks)
    # CaiT layer scale for a shallow stack: 1e-6 leaves every branch dead at init.
    assert all(b.layer_scale.detach().mean().item() == pytest.approx(0.1) for b in blocks)
    # Only the three striding stages downsample; the final stage keeps 16x11 and
    # therefore uses a plain pointwise projection instead of a 3x3 depthwise.
    downsamples = [m for m in net.modules() if isinstance(m, DepthwisePointwiseDownsample)]
    assert len(downsamples) == 3
    assert all(d.depthwise.groups == d.depthwise.in_channels for d in downsamples)
    assert len([m for m in net.modules() if isinstance(m, PointwiseProjection)]) == 1
    assert "checkpoint" not in inspect.getsource(GlassBlock.forward)


def test_frequency_aware_pool_attends_time_per_frequency_then_uses_pyramid():
    pool = FrequencyAwarePool(96).eval()
    x = torch.randn(2, 96, 16, 11)
    weights = pool.attention_weights(x)
    assert weights.shape == (2, 1, 16, 11)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 1, 16), atol=1e-6)

    stats = pool.temporal_statistics(x)
    assert stats.shape == (2, 288, 16)
    assert pool.pyramid_bins == (1, 2, 4)
    # The mixer width is the whole trunk->embedding path; 64 keeps compression
    # at ~4.5x instead of the 12x a 32-channel mixer imposed.
    mixer_out = next(m for m in pool.frequency_mixer.modules()
                     if isinstance(m, torch.nn.Conv1d)).out_channels
    assert mixer_out == 64
    assert mixer_out * (sum(pool.pyramid_bins) + 1) == EMBEDDING_DIM
    assert any(isinstance(m, torch.nn.Conv1d) and m.kernel_size == (3,)
               for m in pool.frequency_mixer.modules())
    assert pool(x).shape == (2, EMBEDDING_DIM)

    # Temporal statistics never merge frequency rows. A change in one row can
    # only alter that row before the explicit Conv1d frequency mixer.
    baseline = torch.zeros(1, 96, 16, 11)
    changed = baseline.clone()
    changed[:, :, 7, 4] = 3.0
    difference = pool.temporal_statistics(changed) - pool.temporal_statistics(baseline)
    assert torch.count_nonzero(difference[..., :7]) == 0
    assert torch.count_nonzero(difference[..., 8:]) == 0
    assert torch.count_nonzero(difference[..., 7]) > 0

    # The explicit static-bin implementation is numerically identical to
    # adaptive pooling even for a non-divisible 58 -> 4 export geometry.
    rows = torch.randn(2, 64, 58)
    for bins in pool.pyramid_bins:
        expected = torch.nn.functional.adaptive_avg_pool1d(rows, bins)
        actual = pool._fixed_adaptive_average(rows, bins)
        torch.testing.assert_close(actual, expected)


def test_nine_fine_tune_stages_cover_parameters_and_last_two_are_bounded():
    net = ClinkCNNv2()
    groups = net.fine_tune_stages()
    assert [name for name, _ in groups] == [
        "stem",
        "stage0.block0",
        "stage1.block0",
        "stage1.block1",
        "stage2.block0",
        "stage2.block1",
        "stage3.block0",
        "stage3.block1",
        "pool",
    ]
    grouped = [p for _, module in groups for p in module.parameters()]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in net.parameters()}
    last_two = sum(p.numel() for _, module in groups[-2:] for p in module.parameters())
    total = sum(p.numel() for p in net.parameters())
    assert last_two / total <= 0.30


def test_parameter_and_mac_budgets():
    net = ClinkCNNv2().eval()
    params = sum(p.numel() for p in net.parameters())
    assert params == 241_025

    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        net(torch.randn(1, 1, N_BANDS, N_FRAMES))
    # PyTorch counts a multiply and an add as two FLOPs. The design budget is
    # stated in multiply-accumulate operations, so divide the counter by two.
    gmac = counter.get_total_flops() / 2e9
    assert gmac <= 0.25, f"MAC budget exceeded: {gmac:.3f} GMAC"


def test_catalog_default_extracts_exact_17640_to_128_by_81_contract():
    from sed.audio.features import build_feature_extractor, window_sample_count
    from sed.config import Config, FeaturesCfg
    from sed.train.model_lab import architecture_by_id, recommended_feature_for

    feature = recommended_feature_for(architecture_by_id("clink-cnn-v2"))
    assert feature is not None
    cfg = Config.model_validate({
        "model": {
            "backbone": "clink-cnn-v2",
            "pretrained": False,
            "initialization": "random",
            "sample_rate": feature["sample_rate"],
        },
        "features": {
            key: value
            for key, value in feature.items()
            if key in FeaturesCfg.model_fields
        },
    })
    assert cfg.model.sample_rate == 44_100
    assert cfg.features.n_fft == 1024
    assert cfg.features.win_length == 882
    assert cfg.features.hop_length == 220
    assert cfg.features.n_mels == N_BANDS
    assert cfg.features.fmin == 80
    assert cfg.features.fmax == 20_000
    samples = window_sample_count(cfg.model.sample_rate, cfg.features.window_seconds)
    assert samples == 17_640
    features = build_feature_extractor(cfg)(torch.zeros(samples))
    assert tuple(features.shape) == (1, N_BANDS, N_FRAMES)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "sample_rate", 48_000),
        ("model", "channels", 2),
        ("features", "window_seconds", 0.5),
        ("features", "log_power", False),
        ("features", "stft_engine", "scipy"),
        ("features", "db_norm", True),
        ("features", "colormap", "jet"),
        ("features", "pcen", True),
    ],
)
def test_supported_frontend_family_rejects_incompatible_semantics(section, field, value):
    from sed.config import (
        Config,
        FeaturesCfg,
        clink_cnn_v2_feature_mismatches,
        validate_clink_cnn_v2_feature_contract,
    )
    from sed.train.model_lab import architecture_by_id, recommended_feature_for

    feature = recommended_feature_for(architecture_by_id("clink-cnn-v2"))
    cfg = Config.model_validate({
        "model": {
            "backbone": "clink-cnn-v2",
            "pretrained": False,
            "initialization": "random",
            "sample_rate": feature["sample_rate"],
        },
        "features": {
            key: item
            for key, item in feature.items()
            if key in FeaturesCfg.model_fields
        },
    })
    bad = cfg.model_copy(deep=True)
    setattr(getattr(bad, section), field, value)

    mismatches = clink_cnn_v2_feature_mismatches(bad)
    assert any(f"{section}.{field}=" in mismatch for mismatch in mismatches)
    with pytest.raises(ValueError, match="ClinkCNN-v2 frontend rejected"):
        validate_clink_cnn_v2_feature_contract(bad, phase="test")


@pytest.mark.parametrize(
    "updates",
    [
        {"n_fft": 2_048, "win_length": 882, "n_mels": 256},
        {"n_fft": 2_048, "win_length": 1_764, "n_mels": 256},
        {"hop_length": 110},
        {
            "feature_type": "linear", "n_fft": 1_024, "win_length": 882,
            "standardize": True, "standardize_stats_resolved": False,
        },
    ],
)
def test_supported_frontend_variations_validate(updates):
    from sed.config import Config, FeaturesCfg
    from sed.train.model_lab import architecture_by_id, recommended_feature_for

    feature = recommended_feature_for(architecture_by_id("clink-cnn-v2"))
    feature.update(updates)
    cfg = Config.model_validate({
        "model": {
            "backbone": "clink-cnn-v2",
            "pretrained": False,
            "initialization": "random",
            "sample_rate": feature["sample_rate"],
        },
        "features": {
            key: item
            for key, item in feature.items()
            if key in FeaturesCfg.model_fields
        },
    })
    assert cfg.model.backbone == "clink-cnn-v2"


def test_shared_feature_extractor_rejects_in_place_contract_drift():
    from sed.audio.features import build_feature_extractor
    from sed.config import Config, FeaturesCfg
    from sed.train.model_lab import architecture_by_id, recommended_feature_for

    feature = recommended_feature_for(architecture_by_id("clink-cnn-v2"))
    cfg = Config.model_validate({
        "model": {
            "backbone": "clink-cnn-v2",
            "pretrained": False,
            "initialization": "random",
            "sample_rate": feature["sample_rate"],
        },
        "features": {
            key: item
            for key, item in feature.items()
            if key in FeaturesCfg.model_fields
        },
    })
    cfg.features.colormap = "jet"
    with pytest.raises(ValueError, match="feature extraction"):
        build_feature_extractor(cfg)


def test_builds_through_sedmodel_and_probes_nine_stages():
    model = SedModel(
        "clink-cnn-v2",
        num_classes=3,
        pretrained=False,
        n_bands=N_BANDS,
        n_frames=N_FRAMES,
        params={},
    ).eval()
    assert model(torch.zeros(2, 1, N_BANDS, N_FRAMES)).shape == (2, 3)
    assert len(model.backbone_stage_modules()) == 9


@pytest.mark.parametrize(
    ("profile", "n_bands", "n_frames"),
    [
        ("native", 128, 81),
        ("dense-mel256", 256, 81),
        ("resolved-mel256", 256, 81),
        ("temporal-mel128", 128, 161),
        ("linear-fft1024", 463, 81),
        ("linear-fft2048", 925, 81),
    ],
)
def test_exports_to_openvino_with_dynamic_batch(
    tmp_path, profile, n_bands, n_frames,
):
    ov = pytest.importorskip("openvino")
    from sed.export.openvino import _export_onnx, convert_onnx_to_ir

    model = SedModel(
        "clink-cnn-v2",
        num_classes=3,
        pretrained=False,
        n_bands=n_bands,
        n_frames=n_frames,
        params={},
    ).eval()
    onnx_path = tmp_path / f"clink-cnn-v2-{profile}.onnx"
    _export_onnx(model, onnx_path, n_bands, n_frames)
    ir = convert_onnx_to_ir(onnx_path, n_bands, n_frames)
    compiled = ov.Core().compile_model(ir, "CPU")
    assert compiled.input(0).partial_shape[0].is_dynamic
    assert compiled.input(0).partial_shape[1:].to_shape() == [1, n_bands, n_frames]
    for batch_size in (1, 2):
        inputs = np.zeros((batch_size, 1, n_bands, n_frames), dtype=np.float32)
        logits = next(iter(compiled({0: inputs}).values()))
        assert logits.shape == (batch_size, 3)
        with torch.no_grad():
            expected = model(torch.from_numpy(inputs)).numpy()
        np.testing.assert_allclose(logits, expected, rtol=1e-4, atol=1e-5)
