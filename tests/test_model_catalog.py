"""Every model listed in the curated catalog must actually build and probe.

The catalog carries no aspirational entries: anything shown in the Model tab
builds from scratch with its declared default feature geometry (HF-backed
entries are skipped when transformers or the network is unavailable, never
failed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sed.train.model_lab import curated_architectures  # noqa: E402

_HF_BACKED = {"ced-tiny", "ced-mini", "ced-small", "ced-base"}


def _probe_geometry(spec) -> tuple[int, int, int]:
    """(n_bands, n_frames, input_channels) from the architecture's default feature."""
    from sed.audio.features import feature_frame_count

    f = dict(spec.default_feature or {})
    if str(f.get("colormap", "none")) != "none":
        size = int(f.get("image_size", 96))
        return size, size, int(spec.input_channels)
    n_bands = int(f.get("n_mels") or 128)
    if f.get("n_fft") and f.get("hop_length") and f.get("sample_rate"):
        n_frames = feature_frame_count(
            sample_rate=int(f["sample_rate"]),
            window_seconds=float(f.get("window_seconds", 1.0)),
            hop_length=int(f["hop_length"]),
            n_fft=int(f["n_fft"]),
            frontend=str(f.get("frontend", "efficientat")),
            win_length=int(f.get("win_length", 0) or 0) or None,
            stft_engine=str(f.get("stft_engine", "librosa")),
        )
    else:
        n_frames = 100
    return n_bands, int(n_frames), int(spec.input_channels)


def _backbone_name(spec) -> str:
    # Mirrors the server's resolution in _handle_model_recipe_select.
    if spec.provider in ("efficientat", "timm"):
        return spec.source
    return spec.id


@pytest.mark.parametrize("spec", curated_architectures(), ids=lambda s: s.id)
def test_catalog_architecture_builds_scratch(spec):
    from sed.train.backbones import get_spec, probe_spec

    if spec.id in _HF_BACKED:
        pytest.importorskip("transformers")
    n_bands, n_frames, in_ch = _probe_geometry(spec)
    backbone = _backbone_name(spec)
    try:
        probe = probe_spec(get_spec(backbone), n_bands=n_bands, n_frames=n_frames,
                           params={}, input_channels=in_ch)
    except (OSError, ConnectionError) as exc:  # HF config fetch needs the network once
        if spec.id in _HF_BACKED:
            pytest.skip(f"offline: {exc}")
        raise
    assert probe["feat_dim"] > 0, f"{spec.id} probed with no embedding"
    assert probe.get("n_stages", 1) >= 1


def test_catalog_has_no_removed_or_superseded_entries():
    ids = {spec.id for spec in curated_architectures()}
    assert ids.isdisjoint({
        "convnext-audio-tiny", "passt-external", "hf-ast-base",
        "tiny-audio-transformer", "passt-lite", "cnn-conformer-tiny",
        "transient-gridnet",
    })


def test_hf_entries_declare_contracts():
    by_id = {spec.id: spec for spec in curated_architectures()}
    for arch_id in _HF_BACKED:
        ced = by_id[arch_id].weights[0].expected_feature_contract
        assert ced["features.n_mels"] == 64 and ced["model.sample_rate"] == 16000
        assert ced["features.hop_length"] == 160

    for arch_id in ("efficient-sed-fmn10", "efficient-sed-fmn20"):
        weight = by_id[arch_id].weights[0]
        contract = weight.expected_feature_contract
        assert contract["features.frontend"] == "efficientat"
        assert contract["features.n_mels"] == 128
        assert contract["model.sample_rate"] == 16000
        assert len(weight.sha256) == 64


def test_archived_entries_hidden_from_catalog_payload_but_still_resolve():
    from sed.train.model_lab import catalog_payload, curated_architectures

    archived = {spec.id for spec in curated_architectures() if spec.archived}
    assert archived == {
        "timm-mobilenetv3-small",
        "kws-tcresnet", "kws-dscnn",
    }
    visible = {a["id"] for a in catalog_payload()["architectures"]}
    assert visible.isdisjoint(archived)
    # The active recipe keeps rendering a retained archived selection (badged).
    kept = {a["id"]: a for a in catalog_payload(keep_ids={"kws-dscnn"})["architectures"]}
    assert kept["kws-dscnn"]["archived"] is True
    # Old manifests naming an archived backbone still resolve to a spec.
    from sed.train.backbones import get_spec

    assert get_spec("kws-dscnn").source.endswith(":DSCNN")


def test_hf_entries_declare_optional_dependency():
    from sed.train.model_lab import curated_architectures

    by_id = {spec.id: spec for spec in curated_architectures()}
    for arch_id in _HF_BACKED:
        assert by_id[arch_id].requires == ("transformers",)
        payload = by_id[arch_id].to_payload()
        assert isinstance(payload["available"], bool)


def test_removed_model_ids_and_classes_are_absent_from_runtime_sources():
    root = Path(__file__).resolve().parents[1]
    runtime = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/sed/train/model_lab.py",
            "src/sed/train/external_audio.py",
            "src/sed/train/short_audio.py",
        )
    )
    for token in (
        "hf-ast-base", "HFASTEncoder", "tiny-audio-transformer", "TinyAudioTransformer",
        "passt-lite", "PaSSTLite", "cnn-conformer-tiny", "CNNConformerTiny",
        "transient-gridnet", "TransientGridNet",
    ):
        assert token not in runtime
