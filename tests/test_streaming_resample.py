"""The streaming resampler, and the onset artifact it exists to remove.

`resample_audio` is stateless, so calling it once per audio block filters each
block as an isolated signal and zero-pads both of its edges. Feeding that to a
spectral-flux onset detector puts an artificial transient at every block
boundary — which is how a room with nothing but a hum in it produced clips.
"""

from __future__ import annotations

import numpy as np
import pytest

from sed.audio.onset import StreamingSuperflux
from sed.audio.resample import StreamingResampler, resample_audio

RATES = [(48000, 32000), (44100, 32000), (32000, 32000), (96000, 32000), (16000, 32000)]


def _detector(sr: int) -> StreamingSuperflux:
    return StreamingSuperflux(sample_rate=sr, hop_ms=32, n_mels=64,
                              median_window_s=5.0, mad_k=3.0,
                              refractory_ms=150, min_flux=1e-4)


@pytest.mark.parametrize(("orig", "target"), RATES)
@pytest.mark.parametrize("block", [1024, 512, 4096, 333])
def test_streaming_matches_whole_file_resampling(orig, target, block):
    sig = np.random.default_rng(0).standard_normal(3 * orig).astype("float32")
    reference = resample_audio(sig, orig, target)

    r = StreamingResampler(orig, target)
    out = np.concatenate([r.process(sig[i:i + block])
                          for i in range(0, len(sig), block)])

    # Output lags by the filter's group delay — a fixed cost, not a backlog —
    # plus at most the decimation step still buffered.
    shortfall = reference.shape[0] - out.shape[0]
    assert 0 <= shortfall <= r.latency + r.down, shortfall
    assert np.max(np.abs(out - reference[:out.shape[0]])) < 1e-5


@pytest.mark.parametrize(("orig", "target"), RATES)
def test_output_is_independent_of_how_the_stream_is_chunked(orig, target):
    """The whole point: the same audio must resample the same way whatever
    block size the sound card happens to deliver."""
    sig = np.random.default_rng(1).standard_normal(2 * orig).astype("float32")

    def run(block):
        r = StreamingResampler(orig, target)
        return np.concatenate([r.process(sig[i:i + block])
                               for i in range(0, len(sig), block)])

    base = run(1024)
    for block in (256, 512, 700, 4096):
        other = run(block)
        n = min(len(base), len(other))
        assert n > 0
        assert np.max(np.abs(base[:n] - other[:n])) < 1e-6


@pytest.mark.parametrize(("orig", "target"), RATES)
def test_output_index_is_the_callers_timebase(orig, target):
    r = StreamingResampler(orig, target)
    emitted = 0
    for _ in range(20):
        emitted += r.process(np.zeros(512, dtype="float32")).shape[0]
    assert r.output_index == emitted
    # Passthrough must keep the same clock as a converting resampler, or a card
    # already at the model rate reports onsets against a frozen timebase.
    assert emitted > 0


def test_reset_forgets_the_stream():
    r = StreamingResampler(48000, 32000)
    r.process(np.ones(4096, dtype="float32"))
    assert r.output_index > 0
    r.reset()
    assert r.output_index == 0


@pytest.mark.parametrize("orig", [48000, 44100])
@pytest.mark.parametrize("block", [1024, 512])
def test_a_steady_tone_fires_no_onsets(orig, block):
    """30 s of a held 440 Hz tone has no transients in it at all.

    Per-block resampling fired 124 times at 48 kHz/1024 and 95 times at
    44.1 kHz/512 — junk clips, saved to the library as labelled positives.
    """
    t = np.arange(int(30 * orig)) / orig
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")

    det, res = _detector(32000), StreamingResampler(orig, 32000)
    fired = 0
    for i in range(0, len(tone) - block, block):
        chunk = res.process(tone[i:i + block])
        if chunk.size:
            fired += len(det.process_block(chunk, res.output_index))
    assert fired == 0


def test_per_block_resampling_is_what_used_to_fire(caplog):
    """Pin the mechanism, so nobody reintroduces it thinking it is equivalent."""
    orig, block = 48000, 1024
    t = np.arange(int(20 * orig)) / orig
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")

    det, fired, w = _detector(32000), 0, 0
    for i in range(0, len(tone) - block, block):
        w += block
        chunk = resample_audio(tone[i:i + block], orig, 32000)   # the old way
        fired += len(det.process_block(chunk, round(w * 32000 / orig)))
    assert fired > 20, "expected the per-block artifact to fire on a pure tone"


@pytest.mark.parametrize(("orig", "target"), RATES)
def test_latency_is_a_fixed_group_delay_not_a_growing_backlog(orig, target):
    r = StreamingResampler(orig, target)
    shortfalls = []
    fed = 0
    for _ in range(30):
        fed += 2048
        r.process(np.zeros(2048, dtype="float32"))
        shortfalls.append(round(fed * target / orig) - r.output_index)
    # The gap cycles with the remainder still buffered (only whole decimation
    # steps are consumed), but it stays inside one group delay plus one step no
    # matter how long the stream runs — it is a delay, not a backlog. A leak
    # would show as the bound being exceeded later in the run.
    assert max(shortfalls) <= r.latency + r.down, shortfalls
