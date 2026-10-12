# Recording Workflow & CLI Improvement Backlog

This document captures key architectural and UX improvements noted during dataset collection sessions.

---

## 1. Terminology: Rename `--class` (`-c`) to `--label` (`-l`)

* **Current State:** `sed record --class riedel-veloce`
* **Problem:** `--class` creates confusion with downstream model target classes (e.g. `cheers`, `_negative`). In raw audio capture, clips are organized by sound bucket / sound identity, not necessarily the final model classification head class.
* **Proposed Change:**
  * Rename the CLI flag to `--label` (`-l`).
  * Maintain backward compatibility alias (`--class` maps to `--label` with a deprecation note).
  * Update sidecars and documentation so clips in `data_library/sounds/<label>/` are consistently referred to as labeled raw sound buckets.

---

## 2. Dedicated `location` / `site` Metadata Field

* **Current State:** Environment/location info is manually passed in `--tags "rnd,office"`.
* **Problem:** Lacks standard structure across recording sessions, making it hard to programmatically filter clips by venue or acoustic site.
* **Proposed Change:**
  * Add `--location` / `--site` CLI argument to `sed record`.
  * Store `location: str` directly in `ClipMeta` sidecars alongside `mic_position` and `project_id`.
  * Expose location filters in dataset composition (`sed dataset`) to allow training models on specific subset locations or building cross-location validation splits.

---

## 3. Lightweight Web UI for Hardware & Recording — **done**

* **Was:** Multiple individual CLI subcommands (`sed audio devices`, `sed audio meter`, `sed check record`, `sed record`) to navigate during a live physical recording session.
* **Shipped:** `uv run sed record --serve` opens a single-page recording console
  on port 8790 (`src/sed/data/record_server.py` + `record_ui.py`), a sibling of
  the inspect server rather than a tab in it — it boots in about a second with
  no library scan, and the two cannot share one PortAudio input anyway.
  * Device enumeration and ordered channel selection, with a live level beside
    each channel so a mis-indexed or unplugged input is visible before the pass.
  * Per-microphone VU meters (peak, RMS, 60 s history) and the `flux vs gate`
    readout, plus `doctor_record` run inline as a pre-flight panel.
  * One-click onset-triggered sessions, with `Continuous` and `Manual` modes
    for material the onset detector should not gate.
  * Onset floor, sensitivity, cooldown, clip length, preroll (including both
    ends of the randomized jitter window) and trigger mode are live sliders —
    no restart — with an explicit *Save to config.toml* per group, and an
    auto-calibrate that proposes a floor from the room.
  * A run list for the console's lifetime, with one-action discard of a whole
    pass — the clips, their sidecars and their library rows — so an unusable
    pass never has to be hunted down in Inspect by its session id.
* **Not done:** the CLI itself was not streamlined; the subcommands all remain.
  The console covers the same ground, so the case for removing them is weaker
  than it was.

Both front-ends run the same loop (`src/sed/data/record_engine.py`), so the
terminal recorder and the console cannot produce different kinds of clip.

---

## 4. Recording Duration & Temporal Transparency — **mostly done**

* **Was:** Clip length was configured via `[onset] record_seconds` and never shown.
* **Shipped:**
  * `sed record` prints `clip duration: 1.00s (48,000 frames @ 48000 Hz)` in its
    startup header.
  * `--record-seconds` / `-rs` overrides it per session.
  * The recording console shows the same figure live under the clip-length
    slider, next to a small diagram of where the onset sits inside the clip
    (preroll │ onset │ postroll) — the geometry that decides what the model
    window crops out later.
* **Still open:** sample-exact frame sizing. Clip length is still
  `int(record_seconds * sample_rate)`, so a project mixing capture rates can
  hold clips whose frame counts differ by one. Nothing downstream depends on
  identical frame counts today (features are computed per clip and the model
  window is cropped around the onset), so this is a tidiness question rather
  than a correctness one.
