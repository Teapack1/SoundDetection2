"""Tests for the onset gate — the ceiling on system recall.

With `[inference] onset_gating_enabled`, the detector decides what the model
ever sees, so these cover the merge rule production uses and the offline scan
that measures it.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from sed.audio.onset import OnsetMerger
from sed.config import Config
from sed.runtime.onset_scan import aggregate, flux_histogram, scan_file


def test_onset_merger_collapses_onsets_inside_the_cooldown():
    """The same physical transient reaches several mics within milliseconds and
    must be scored once — this is the rule `infer._on_audio_block` applies."""
    merger = OnsetMerger(cooldown_samples=100)

    # Three mics hearing one transient, plus a second transient well clear of it.
    assert merger.push([1000, 1010, 1040]) == [1000]
    assert merger.push([1500]) == [1500]


def test_onset_merger_cooldown_runs_from_the_last_accepted_onset():
    """A rejected onset must not extend the cooldown — otherwise a burst of
    near-simultaneous mic hits would suppress the next real transient."""
    merger = OnsetMerger(cooldown_samples=100)

    assert merger.push([0]) == [0]
    assert merger.push([50]) == []        # inside the cooldown, dropped
    assert merger.push([100]) == [100]    # exactly one cooldown after ACCEPTED 0
    assert merger.push([199]) == []
    assert merger.push([200]) == [200]


def test_onset_merger_sorts_within_a_batch_and_survives_reset():
    merger = OnsetMerger(cooldown_samples=10)
    # Mics are iterated in channel order, so a later mic can report an earlier index.
    assert merger.push([30, 5, 31]) == [5, 30]

    merger.reset()
    assert merger.push([0]) == [0], "reset must clear the accepted-onset history"


def test_onset_merger_with_zero_cooldown_accepts_everything():
    assert OnsetMerger(cooldown_samples=0).push([5, 5, 6]) == [5, 5, 6]


def _click_track(path, *, sr=32000, seconds=4.0, click_times=(0.5, 1.5, 2.5), channels=1):
    """Silence with sharp broadband clicks — an unmistakable onset for the gate."""
    rng = np.random.default_rng(0)
    n = int(sr * seconds)
    sig = (rng.standard_normal(n) * 1e-4).astype(np.float32)  # dither, no onsets
    for t in click_times:
        i = int(t * sr)
        burst = rng.standard_normal(int(sr * 0.01)).astype(np.float32)
        decay = np.exp(-np.linspace(0, 6, burst.size)).astype(np.float32)
        sig[i:i + burst.size] += burst * decay
    data = np.column_stack([sig] * channels) if channels > 1 else sig
    sf.write(str(path), data, sr)
    return sig


def _scan_cfg(sr=32000) -> Config:
    cfg = Config()
    cfg.model.sample_rate = sr
    cfg.audio.record_sample_rate = sr
    cfg.audio.blocksize = 1024
    cfg.inference.inference_channels = [0]
    return cfg


def test_scan_file_finds_clicks_and_reports_an_hourly_rate(tmp_path):
    wav = tmp_path / "clicks.wav"
    _click_track(wav, seconds=5.0, click_times=(1.5, 2.5, 3.5))

    result = scan_file(_scan_cfg(), wav)

    assert result.duration_s == pytest.approx(5.0, abs=0.01)
    assert result.merged_onset_count >= 3, "the three clicks must clear the gate"
    # Rate is derived from the real duration, so it is directly comparable
    # across recordings of different lengths.
    assert result.onsets_per_hour == pytest.approx(
        result.merged_onset_count / 5.0 * 3600.0
    )
    assert all(0.0 <= t <= 5.0 for t in result.merged_onsets_s)


def test_scan_file_cannot_fire_during_the_detector_warm_up(tmp_path):
    """`StreamingSuperflux` needs 32 flux frames before it has a baseline —
    about 1.0 s at the default 32 ms hop. Anything earlier is invisible to it.

    This is why the onset gate cannot be measured on the 1.0 s onset-triggered
    library clips: a whole clip fits inside the warm-up. Continuous audio is
    required for an onsets/hour figure that means anything.
    """
    wav = tmp_path / "early.wav"
    _click_track(wav, seconds=1.0, click_times=(0.5,))

    assert scan_file(_scan_cfg(), wav).merged_onset_count == 0


def test_scan_file_merges_across_channels_rather_than_counting_each_mic(tmp_path):
    """Three mics hearing the same clicks must not triple the scored count —
    otherwise every FP/hour estimate is inflated by the mic count."""
    wav = tmp_path / "clicks3.wav"
    _click_track(wav, seconds=5.0, click_times=(1.5, 2.5, 3.5), channels=3)

    cfg = _scan_cfg()
    cfg.inference.inference_channels = [0, 1, 2]
    result = scan_file(cfg, wav)

    assert result.channels == [0, 1, 2]
    assert result.raw_onset_count > result.merged_onset_count
    assert result.merge_ratio > 1.0
    assert result.merged_onset_count == pytest.approx(3, abs=1)


def test_scan_file_falls_back_when_the_file_has_fewer_channels(tmp_path):
    """A mono capture of a multi-mic venue is still worth scanning."""
    wav = tmp_path / "mono.wav"
    _click_track(wav, channels=1)

    cfg = _scan_cfg()
    cfg.inference.inference_channels = [0, 1, 2]
    assert scan_file(cfg, wav).channels == [0]


def test_aggregate_pools_duration_and_onsets_across_files(tmp_path):
    files = []
    for i in range(2):
        wav = tmp_path / f"c{i}.wav"
        _click_track(wav, seconds=5.0, click_times=(1.5, 2.5, 3.5))
        files.append(wav)
    results = [scan_file(_scan_cfg(), f) for f in files]

    agg = aggregate(results)
    assert agg["files"] == 2
    assert agg["duration_s"] == pytest.approx(10.0, abs=0.05)
    assert agg["merged_onsets"] == sum(r.merged_onset_count for r in results)
    assert agg["onsets_per_hour"] == pytest.approx(
        agg["merged_onsets"] / agg["duration_s"] * 3600.0
    )


def test_flux_histogram_summarises_gate_headroom(tmp_path):
    wav = tmp_path / "clicks.wav"
    _click_track(wav, seconds=5.0, click_times=(1.5, 2.5, 3.5))

    hist = flux_histogram([scan_file(_scan_cfg(), wav)], bins=6)

    assert hist["n"] > 0
    assert len(hist["bins"]) == 6
    assert sum(b["count"] for b in hist["bins"]) <= hist["n"]
    assert hist["median_flux_db"] is not None
    # Mostly silence, so only a small share of frames sit above the gate.
    assert 0.0 <= hist["above_gate_fraction"] <= 1.0


def test_flux_histogram_handles_an_empty_scan():
    assert flux_histogram([])["bins"] == []
