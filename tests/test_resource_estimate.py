"""Focused precision-contract tests for the Train-tab compute estimator."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("precision", ["16-mixed", "bf16-mixed"])
def test_cpu_estimate_falls_back_to_fp32_without_invalid_autocast(precision):
    torch = pytest.importorskip("torch")
    from sed.train.resource_estimate import _autocast_setup

    context, scaler, note = _autocast_setup(precision, torch.device("cpu"))

    assert context is None
    assert scaler is None
    assert note == "no GPU — measured in fp32 on CPU"


def test_cpu_estimate_accepts_canonical_full_precision():
    torch = pytest.importorskip("torch")
    from sed.train.resource_estimate import _autocast_setup

    context, scaler, note = _autocast_setup("32-true", torch.device("cpu"))

    assert context is None
    assert scaler is None
    assert note is None


def test_old_cuda_devices_do_not_present_emulated_bf16_as_native(monkeypatch):
    torch = pytest.importorskip("torch")
    from sed.train.resource_estimate import _autocast_setup

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (5, 2))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    context, scaler, note = _autocast_setup("bf16-mixed", torch.device("cuda"))

    assert context is None
    assert scaler is None
    assert note == "GPU has no native bf16 support — measured in fp32"
