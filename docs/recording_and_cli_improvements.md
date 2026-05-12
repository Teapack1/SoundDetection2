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

## 3. CLI Streamlining & Lightweight Web UI for Hardware & Recording

* **Current State:** Multiple individual CLI subcommands (`sed audio devices`, `sed audio meter`, `sed check record`, `sed record`, `sed project`, etc.).
* **Problem:** Navigating many CLI subcommands can feel overwhelming during live physical recording sessions.
* **Proposed Change:**
  * Streamline CLI commands down to core essential entrypoints.
  * Build a fast, lightweight web UI module (extending the inspect server or running a micro-web app) to handle:
    * Interactive audio device enumeration & channel selection.
    * Real-time VU level meters and pre-flight health checks.
    * Quick one-click onset-triggered recording sessions with instant visual feedback.

---

## 4. Recording Duration, Frame Precision & Temporal Transparency

* **Current State:** Clip length (e.g. `0.4s`) is configured via `[onset] record_seconds` in `config.toml`, but isn't explicitly displayed in live recording logs or directly overridable on the CLI.
* **Problem:** Operators lack instant visibility into exact clip durations during recording, and small floating-point time conversions can introduce minor sample length variations.
* **Proposed Change:**
  * Display explicit clip duration (e.g. `0.40s / 19,200 frames`) in `sed record` startup headers and live log lines.
  * Add a `--record-seconds` / `--duration-ms` flag to `sed record` for session-level overrides.
  * Research sample-exact frame sizing (e.g., locking clips to exact frame counts like 19,200 samples @ 48 kHz or 17,640 @ 44.1 kHz) to guarantee all dataset clips share identical matrix shapes.
