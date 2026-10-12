"""On-the-spot quality checks for a freshly recorded clip.

The point of the recording console's preview is not to admire the sound — it is
to answer "is this take usable?" while you can still do something about it. The
faults that ruin a recording session are dull and mechanical: a poorly seated
cable humming at mains frequency, a preamp driven into clipping, a DC offset
from a bad input stage, a channel that is really just noise. All of them are
obvious in a linear-frequency spectrogram and trivial to measure, and none of
them are visible in a mel spectrogram, which throws away exactly the
high-frequency detail where cable hash lives and compresses the low end where
hum lives.
"""

from __future__ import annotations

import math

import numpy as np

# Mains hum and its harmonics. Both grids, because the library travels.
MAINS_HZ = (50.0, 60.0)
HUM_HARMONICS = 4
# A hum band this far above the local spectral background is audible and worth
# flagging; below it, every room has some.
HUM_FLAG_DB = 12.0
# Peak at or above this is clipping for practical purposes.
CLIP_DBFS = -0.1
# A DC offset this large points at a broken input stage, not at the room.
DC_FLAG = 0.01
# Below this the channel is preamp noise, not a signal.
SILENT_RMS_DBFS = -85.0


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def analyze_clip(wave: np.ndarray, sample_rate: int) -> dict:
    """Measure one channel of audio and report what is wrong with it.

    Returns levels, a hum estimate per mains grid, and a list of plain-language
    warnings ready to show next to the clip.
    """
    x = np.asarray(wave, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"warnings": ["empty clip"], "peak_dbfs": -120.0, "rms_dbfs": -120.0}

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    dc = float(np.mean(x))
    clipped = int(np.count_nonzero(np.abs(x) >= 0.999))

    spectrum, freqs = _spectrum(x, sample_rate)
    hum = _hum_estimate(spectrum, freqs)
    # Crest factor separates a transient recording from a wall of noise: a
    # clean clink peaks far above its own average, a hissy channel does not.
    crest_db = _dbfs(peak) - _dbfs(rms) if rms > 0 else 0.0

    warnings: list[str] = []
    if peak >= 10 ** (CLIP_DBFS / 20.0) or clipped > 2:
        warnings.append(
            f"clipping ({clipped} sample(s) at full scale) — lower the input gain"
        )
    if abs(dc) > DC_FLAG:
        warnings.append(f"DC offset {dc:+.3f} — check the input stage or cable")
    if hum["excess_db"] >= HUM_FLAG_DB:
        warnings.append(
            f"{hum['mains_hz']:.0f} Hz hum, {hum['excess_db']:.0f} dB above the "
            "surrounding spectrum — reseat the cable or move it off the mains lead"
        )
    if _dbfs(rms) < SILENT_RMS_DBFS:
        warnings.append("essentially silent — nothing connected to this channel?")
    elif _dbfs(peak) < -45.0:
        warnings.append("very quiet — raise the input gain or move the mic closer")

    return {
        "peak_dbfs": round(_dbfs(peak), 1),
        "rms_dbfs": round(_dbfs(rms), 1),
        "dc_offset": round(dc, 5),
        "clipped_samples": clipped,
        "crest_db": round(crest_db, 1),
        "hum": hum,
        "warnings": warnings,
    }


def _spectrum(x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Average magnitude spectrum over the clip, in dB."""
    n = min(len(x), 1 << 15)
    n_fft = 1 << max(10, int(n).bit_length() - 1)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    hop = n_fft // 2
    window = np.hanning(n_fft)
    frames = [
        np.abs(np.fft.rfft(x[i:i + n_fft] * window))
        for i in range(0, len(x) - n_fft + 1, hop)
    ] or [np.abs(np.fft.rfft(x[:n_fft] * window))]
    mag = np.mean(np.stack(frames), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return 20.0 * np.log10(mag + 1e-12), freqs


def _hum_estimate(spectrum_db: np.ndarray, freqs: np.ndarray) -> dict:
    """How far mains hum stands above the spectrum around it.

    Compared against neighbouring bins rather than an absolute level, because
    what makes hum a fault is that it is a narrow spike where the room is
    otherwise quiet — a loud room does not make a 50 Hz spike acceptable.
    """
    best = {"mains_hz": 0.0, "excess_db": 0.0, "harmonics": []}
    for mains in MAINS_HZ:
        excesses = []
        for k in range(1, HUM_HARMONICS + 1):
            target = mains * k
            if target >= freqs[-1]:
                break
            i = int(np.argmin(np.abs(freqs - target)))
            # Local background: nearby bins, excluding the spike itself.
            lo, hi = max(0, i - 12), min(len(spectrum_db), i + 13)
            neighbourhood = np.concatenate(
                [spectrum_db[lo:max(lo, i - 2)], spectrum_db[min(hi, i + 3):hi]]
            )
            if neighbourhood.size == 0:
                continue
            excesses.append(float(spectrum_db[i] - np.median(neighbourhood)))
        if not excesses:
            continue
        # The fundamental plus its strongest harmonic: a single spike can be a
        # real tone in the room, a harmonic series is a power problem.
        score = float(np.mean(sorted(excesses, reverse=True)[:2]))
        if score > best["excess_db"]:
            best = {
                "mains_hz": mains,
                "excess_db": round(score, 1),
                "harmonics": [round(e, 1) for e in excesses],
            }
    return best


def render_spectrogram(
    wave: np.ndarray, sample_rate: int, *, width: int = 900, height: int = 380,
    with_axes: bool = True,
) -> bytes:
    """A linear-frequency spectrogram of one channel, as PNG bytes.

    Linear and full-range on purpose. The dataset previews elsewhere are mel
    spectrograms because that is what the model sees, but for judging a
    recording that is the wrong picture: mel compresses the bottom of the
    spectrum, where hum lives, into a couple of bands, and spends most of its
    resolution where cable hash does not show. A linear axis from DC to Nyquist
    makes a mains spike a straight horizontal line you cannot miss, and shows
    whether the top octave is signal or hiss.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(wave, dtype=np.float64).reshape(-1)
    if x.size == 0:
        x = np.zeros(1024)

    n_fft = 1024 if x.size >= 1024 else 256
    hop = max(1, n_fft // 4)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft)
    frames = np.stack([
        np.abs(np.fft.rfft(x[i:i + n_fft] * window))
        for i in range(0, x.size - n_fft + 1, hop)
    ], axis=1)
    spec = 20.0 * np.log10(frames + 1e-10)
    # Fixed 70 dB span below the loudest bin: an absolute scale would make a
    # quiet take look empty and a loud one look saturated, and the comparison
    # between takes is the whole point.
    top = float(spec.max())
    spec = np.clip(spec, top - 70.0, top)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0.085, 0.16, 0.9, 0.8) if with_axes else (0, 0, 1, 1))
    ax.imshow(
        spec, origin="lower", aspect="auto", cmap="magma",
        extent=(0.0, x.size / sample_rate, 0.0, sample_rate / 2.0),
    )
    if with_axes:
        ax.set_xlabel("seconds", fontsize=8)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        # Mark the mains frequencies: hum is the fault people are actually
        # looking for, and it is much easier to see against a reference line.
        for mains in MAINS_HZ:
            ax.axhline(mains, color="#7dd3fc", lw=0.5, alpha=0.35)
    else:
        ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                facecolor="#0f172a", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()
