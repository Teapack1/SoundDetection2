"""Frame-task target alignment: interval annotations vs the window crop.

A clip longer than the feature window is cropped to it, and the frame-target
mapping must shift interval times by the crop offset (they are original-clip
coordinates). The crop is placed so the event sits `Config.window_lead_in_s()`
into the window — the geometry the onset-gated runtime scores — so an annotated
event is never cropped out, however late or early it sits in the recording.
"""

import json

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from sed.config import Config
from sed.train.datamodule import SedDataset

SR = 32000


def _write_clip(tmp_path, name: str, seconds: float, intervals: list[dict]):
    wav = tmp_path / f"{name}.wav"
    sf.write(str(wav), np.zeros(int(seconds * SR), dtype="float32"), SR)
    wav.with_suffix(".json").write_text(json.dumps({"event_intervals": intervals}))
    return wav


def _dataset(tmp_path, wav_path) -> SedDataset:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(f"path,class_name\n{wav_path},cheers\n")
    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers"]}})
    cfg.model.sample_rate = SR
    cfg.model.task = "frame"
    cfg.features.window_seconds = 1.0
    return SedDataset(csv_path, ["_negative", "cheers"], cfg)


def _positive_frame_bounds(target: torch.Tensor, class_idx: int) -> tuple[int, int]:
    hits = torch.nonzero(target[class_idx]).flatten()
    assert len(hits) > 0
    return int(hits[0]), int(hits[-1])


def test_short_clip_frame_targets_are_unshifted(tmp_path):
    wav = _write_clip(tmp_path, "short", 1.0, [
        {"label": "cheers", "onset_s": 0.25, "offset_s": 0.5, "confirmed": True},
    ])
    ds = _dataset(tmp_path, wav)
    _spec, target = ds[0]
    frames = target.shape[-1]
    first, last = _positive_frame_bounds(target, ds.class_index["cheers"])
    assert first == round(0.25 * frames)
    assert last == round(0.5 * frames) - 1


def test_custom_negative_role_class_gets_whole_clip_background_target(tmp_path):
    """A class with role="negative" that is NOT literally `_negative` (e.g. a
    project splitting its background pool into a named `hardneg` bucket) must
    get the same frame-task treatment `_negative` gets: no annotated interval
    required, and the whole clip painted as background for its own column.

    Before generalizing `_has_frame_target`/the frame-target painting beyond
    the literal `_negative` string, a row like this would fail
    `_has_frame_target` (no matching interval) and be silently dropped from
    frame-task training entirely.
    """
    wav = _write_clip(tmp_path, "hardneg_clip", 1.0, [])  # no interval annotation
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(f"path,class_name\n{wav},hardneg\n")
    cfg = Config.model_validate({
        "dataset": {"target_classes": ["_negative", "hardneg", "cheers"]},
        "targets": {"hardneg": {"role": "negative"}},
    })
    cfg.model.sample_rate = SR
    cfg.model.task = "frame"
    cfg.features.window_seconds = 1.0
    ds = SedDataset(csv_path, ["_negative", "hardneg", "cheers"], cfg)

    assert len(ds) == 1  # row survived the frame-task filter
    _spec, target = ds[0]
    hardneg_idx = ds.class_index["hardneg"]
    assert torch.all(target[hardneg_idx] == 1.0)
    other_idx = [i for i in range(target.shape[0]) if i != hardneg_idx]
    assert torch.all(target[other_idx] == 0.0)


def test_long_clip_frame_targets_follow_the_onset_placed_crop(tmp_path):
    # 2 s clip, 1 s window. The crop is placed so the annotated onset sits
    # `window_lead_in_s` into the window — derived from the config rather than
    # written out here, because it moves with [onset] preroll_ms.
    wav = _write_clip(tmp_path, "long", 2.0, [
        {"label": "cheers", "onset_s": 1.0, "offset_s": 1.2, "confirmed": True},
    ])
    ds = _dataset(tmp_path, wav)
    _spec, target = ds[0]
    frames = target.shape[-1]
    lead_in = ds.cfg.window_lead_in_s() / ds.cfg.features.window_seconds
    event_s = 0.2 / ds.cfg.features.window_seconds      # 1.0 s -> 1.2 s
    first, last = _positive_frame_bounds(target, ds.class_index["cheers"])
    assert first == round(lead_in * frames)
    assert last == round((lead_in + event_s) * frames) - 1


def test_event_near_the_clip_start_is_kept_not_cropped_away(tmp_path):
    """An early event used to be cropped out entirely by the centre crop.

    A 2 s clip centre-cropped to 1 s keeps [0.5, 1.5), so an event at
    [0.1, 0.3] produced an all-zero target — a positive clip trained as
    background. The crop now clamps to the clip start instead, keeping it.
    """
    wav = _write_clip(tmp_path, "early", 2.0, [
        {"label": "cheers", "onset_s": 0.1, "offset_s": 0.3, "confirmed": True},
    ])
    ds = _dataset(tmp_path, wav)
    _spec, target = ds[0]
    frames = target.shape[-1]
    first, last = _positive_frame_bounds(target, ds.class_index["cheers"])
    assert first == round(0.1 * frames)
    assert last == round(0.3 * frames) - 1


def test_event_near_the_clip_end_is_kept_not_cropped_away(tmp_path):
    """The mirror case: a late event must not fall off the right edge."""
    wav = _write_clip(tmp_path, "late", 2.0, [
        {"label": "cheers", "onset_s": 1.8, "offset_s": 1.95, "confirmed": True},
    ])
    ds = _dataset(tmp_path, wav)
    _spec, target = ds[0]
    frames = target.shape[-1]
    # Crop clamps to the last full window, [1.0 s, 2.0 s).
    first, last = _positive_frame_bounds(target, ds.class_index["cheers"])
    assert first == round(0.8 * frames)
    assert last == round(0.95 * frames) - 1


def test_window_lead_in_is_capped_to_a_fraction_of_the_window():
    """The recording preroll must not eat a short model window.

    `onset.record_seconds` is deliberately longer than the model window, so at
    0.4 s a 300 ms preroll would spend three quarters of the input on pre-onset
    background and truncate the decay of the transient being classified.
    """
    from sed.config import Config

    cfg = Config()
    cfg.features.window_seconds = 1.0
    cfg.onset.preroll_ms = 300
    assert cfg.window_lead_in_s() == pytest.approx(0.30)   # historical behaviour

    cfg.features.window_seconds = 0.4
    assert cfg.window_lead_in_s() == pytest.approx(0.12)   # 30% of the window

    # A preroll shorter than the cap is used as-is.
    cfg.onset.preroll_ms = 50
    assert cfg.window_lead_in_s() == pytest.approx(0.05)


def test_short_window_crops_around_the_onset_not_the_clip_centre(tmp_path):
    """The transient must survive a window shorter than the recording.

    Clips are recorded with a randomized preroll spread across the whole clip.
    Centre-cropping a 0.4 s window out of them removed the transient entirely
    from 52% of this project's own positive clips — training positives that
    contain only background.
    """
    import numpy as np
    import soundfile as sf

    from sed.config import Config
    from sed.train.datamodule import SedDataset

    sr = 32000

    def _clip(name: str, onset_s: float):
        wave = np.zeros(sr, dtype="float32")          # 1 s
        wave[int(onset_s * sr):int(onset_s * sr) + 400] = 0.9
        wav = tmp_path / f"{name}.wav"
        sf.write(str(wav), wave, sr)
        wav.with_suffix(".json").write_text(json.dumps({"onset_offset_s": onset_s}))
        return wav

    csv_path = tmp_path / "train.csv"
    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers"]}})
    cfg.model.sample_rate = sr
    assert cfg.features.window_seconds == 0.4      # the project default
    lead_in = cfg.window_lead_in_s()

    # A click at 0.5 s is outside a centre crop ([0.3, 0.7) would clip its tail
    # and a later one entirely) but the window is placed to include it.
    wav = _clip("mid_click", 0.5)
    csv_path.write_text(f"path,class_name\n{wav},cheers\n")
    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    cropped, crop_offset_s = ds._load(str(wav))

    assert len(cropped) == int(0.4 * sr)
    assert float(np.max(np.abs(cropped))) == pytest.approx(0.9, abs=1e-4)  # PCM_16
    assert crop_offset_s == pytest.approx(0.5 - lead_in, abs=1e-6)
    peak_at_s = int(np.argmax(np.abs(cropped))) / sr
    assert peak_at_s == pytest.approx(lead_in, abs=0.01)

    # A click too late for the full lead-in clamps to the last whole window
    # rather than running off the end — the transient is still kept.
    late = _clip("late_click", 0.9)
    csv_path.write_text(f"path,class_name\n{late},cheers\n")
    ds_late = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    cropped_late, offset_late = ds_late._load(str(late))
    assert offset_late == pytest.approx(1.0 - 0.4, abs=1e-6)
    assert float(np.max(np.abs(cropped_late))) == pytest.approx(0.9, abs=1e-4)


def test_onset_offset_falls_back_to_detection_then_centre(tmp_path):
    """No sidecar hint → detect it; nothing detectable → centre crop."""
    import numpy as np
    import soundfile as sf

    from sed.config import Config
    from sed.train.datamodule import SedDataset

    sr = 32000
    wave = np.zeros(sr, dtype="float32")
    wave[int(0.75 * sr):int(0.75 * sr) + 600] = 0.8
    wav = tmp_path / "no_sidecar.wav"
    sf.write(str(wav), wave, sr)          # deliberately no .json sidecar

    csv_path = tmp_path / "train.csv"
    csv_path.write_text(f"path,class_name\n{wav},cheers\n")
    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers"]}})
    cfg.model.sample_rate = sr
    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)

    cropped, _ = ds._load(str(wav))
    assert float(np.max(np.abs(cropped))) == pytest.approx(0.8, abs=1e-4)  # transient kept

    # The per-clip answer is memoized, so the ~30 ms detection runs once.
    assert str(wav) in ds._onset_cache
