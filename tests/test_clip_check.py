"""On-the-spot recording quality checks.

These exist to answer "is this take usable?" while it can still be re-recorded,
so each test is one fault an operator would actually hit on site.
"""

from __future__ import annotations

import numpy as np
import pytest

from sed.data.clip_check import analyze_clip, render_spectrogram

SR = 48_000


def _room(seed: int = 0, scale: float = 0.02, seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * scale).astype("float32")


def _tone(hz: float, amp: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype("float32")


def test_a_clean_take_reports_no_faults():
    result = analyze_clip(_room(), SR)
    assert result["warnings"] == []
    assert -40 < result["peak_dbfs"] < -5
    assert abs(result["dc_offset"]) < 0.01


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_mains_hum_is_found_and_named(mains):
    """The fault a badly seated cable actually produces."""
    hummy = _room() + _tone(mains, 0.05) + _tone(mains * 2, 0.02)
    result = analyze_clip(hummy, SR)
    assert result["hum"]["mains_hz"] == mains
    assert result["hum"]["excess_db"] > 12.0
    assert any("hum" in w for w in result["warnings"])
    assert any(f"{mains:.0f} Hz" in w for w in result["warnings"])


def test_a_quiet_room_is_not_called_hum():
    # Every room has some mains energy; only a spike well above the local
    # spectrum is a fault worth interrupting a session for.
    assert analyze_clip(_room(scale=0.001), SR)["hum"]["excess_db"] < 12.0


def test_clipping_is_reported_with_a_count():
    result = analyze_clip(np.clip(_room() * 60, -1, 1), SR)
    assert result["clipped_samples"] > 0
    assert any("clipping" in w for w in result["warnings"])


def test_dc_offset_is_reported():
    result = analyze_clip(_room() + 0.05, SR)
    assert result["dc_offset"] == pytest.approx(0.05, abs=0.005)
    assert any("DC offset" in w for w in result["warnings"])


def test_a_dead_channel_is_called_out():
    assert any("silent" in w for w in analyze_clip(_room(scale=1e-6), SR)["warnings"])


def test_a_very_quiet_take_suggests_more_gain():
    # DESIGN.md targets peaks around -18 to -6 dBFS; a peak below -45 is a
    # gain problem, not a quiet room. -40 dBFS is still usable and must not nag.
    assert any("quiet" in w for w in analyze_clip(_room(scale=1e-3), SR)["warnings"])
    assert analyze_clip(_room(scale=2e-3), SR)["warnings"] == []


def test_an_empty_clip_does_not_crash():
    assert analyze_clip(np.zeros(0, dtype="float32"), SR)["warnings"]


def test_spectrogram_renders_png_at_both_sizes():
    audio = _room() + _tone(50.0, 0.05)
    big = render_spectrogram(audio, SR, width=900, height=380, with_axes=True)
    small = render_spectrogram(audio, SR, width=260, height=90, with_axes=False)
    for png in (big, small):
        assert png[:4] == b"\x89PNG"
    # Axes and tick labels make the large one substantially bigger.
    assert len(big) > len(small)


def test_spectrogram_handles_a_clip_shorter_than_one_fft_frame():
    png = render_spectrogram(np.zeros(200, dtype="float32"), SR, width=200, height=80)
    assert png[:4] == b"\x89PNG"
