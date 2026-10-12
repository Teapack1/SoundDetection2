# Workflow

Run this after completing [SETUP.md](SETUP.md). You record, train, and test live inference on the workstation, then deploy the model to the edge device.

- **`<project_id>`**: Project directory name under `projects/` (e.g., `temna`).
- **`<model_id>`**: Trained model bundle folder name (e.g., `2026-06-11_1508-baseline`).
- **`uv run sed`**: Prefix to execute CLI commands in the local virtual environment.

---

## 1. Create a Project

```bash
uv run sed project new <project_id> --name "Venue Name"
```
Edit `projects/<project_id>/config.toml` to specify classes and audio channels:

```toml
[dataset]
target_classes = ["_negative", "cheers_loud", "cheers_quiet"]

[deploy]
record_device = "Scarlett"      # Stable name substring of your audio device

[audio]
record_sample_rate = 48000
record_channels = [0, 2]        # Ordered, zero-based input channels to record

[onset]
multi_mic_trigger = "any"       # Each selected mic runs its own onset detector
recording_cooldown_ms = 1500    # Global delay before any mic can start another recording
cross_channel_merge_ms = 500    # Merge mic onsets into one multichannel clip

[inference]
inference_channels = [0]        # Ordered, zero-based channels to run the model on
```
*Channels are zero-based physical indexes (a bare integer is read as a count, e.g. `1` → `[0]`). `inference_channels` opens the multichannel device but runs the model on only those channels — no downmixing — independently of `record_channels`.*

```bash
uv run sed project list                 # List projects
uv run sed project info <project_id>    # Project info and stats
```

---

## 2. Check Audio Signal

```bash
uv run sed audio devices                         # List input devices and channel counts
uv run sed audio meter --project <project_id>    # Live channel level meter
uv run sed check record --project <project_id>   # Pre-recording check (mics and signals)
```

All three are also in the recording console below, which is usually the faster
way to run them on site.

---

## 3. Record Sounds

Record one class at a time. Clips are saved to `data_library/sounds/<label>/` and reused across projects.

### Recording console (recommended on site)

```bash
uv run sed record --serve                      # no project: fills the shared library
uv run sed record --serve --project <id>       # record for a venue
```

Opens a single-page UI on http://127.0.0.1:8790 — boots in about a second, no
library scan. Every flag below also works with `--serve` and prefills the form.

**A project is optional, and picked in the console.** Most recording just
expands `data_library/`, so *no project* is the default: those clips carry
`_library`, which every project's default source rule accepts alongside its own
id, so they are usable everywhere. Choosing a project in the top bar loads that
venue's device, channel and onset settings and stamps its id on subsequent
clips, so the project's own source rules pick them up — another project can
still use them by naming that project in a source rule. Switching mid-session
starts a new recording group rather than relabelling anything already recorded.
Saving settings to `config.toml` needs a project; without one, changes last for
the session.

* **Setup**: pick the interface, sample rate and block size; tick channels with
  their live level next to them; run the `check record` checklist inline.
* **Session**: label (with the library's existing labels and their clip counts),
  density, location and tags (both suggested from what has already been
  recorded, tags with their usage counts), mic position, microphone names,
  description, frozen-test and
  split-channels toggles, target count and time cap — all editable *while*
  recording. Changing the label starts a new recording group.
* **Health**: an unsupported sample rate or channel count fails at Start with the
  reason, instead of a session that reports "recording" with silent meters. A
  device dropout, a failing disk and dropped clips are reported in a banner.
* **Live**: per-microphone meters with peak/RMS, a 60 s level history, and the
  `flux vs gate` readout that decides whether a transient fires. Onset floor,
  sensitivity, cooldown, clip length, preroll and trigger mode are sliders that
  apply without dropping the audio stream; each group has its own
  **Save to config.toml**, which preserves the file's comments.
* **Clip geometry**: the clip-length slider shows the exact duration and frame
  count, and a diagram of where the transient sits inside the clip — guaranteed
  preroll, the window a randomized onset lands in, and the postroll. With
  *Randomize per clip* on, both ends of that window are their own sliders; all
  of them are bounded by the current clip length rather than a fixed ceiling.
* **Modes**: `Onset` (a clip per transient), `Continuous` (back-to-back clips,
  gate off — the fast way to collect room tone and negatives) and `Manual`.
* **Auto-calibrate** works before you start recording — press *Start meter*, then
  calibrate. It listens to the room and places the onset floor above what the
  room does on its own, at one of three strictnesses (`relaxed` / `normal` /
  `strict`). Move around while it listens: whatever it hears becomes what it
  ignores, so walking and glass handling stop triggering takes. The result
  reports the room's median, 95th and loudest flux so you can see the headroom.
* **Inspect a take**: click any spectrogram in *Recent takes* to enlarge it.
  These are linear and full-range (DC to Nyquist), not the mel previews the
  dataset UI shows — mains hum reads as a hard horizontal line, cable hash as a
  bright top octave, and neither is visible in mel. Alongside it the take is
  measured: peak, RMS, crest factor, DC offset, clipped samples and mains hum
  per channel, with plain warnings ("50 Hz hum, 24 dB above the surrounding
  spectrum — reseat the cable"). Multichannel takes get a per-channel switch.
* **Recent takes** shows the last clips with spectrogram, playback and a delete
  button, so a bad take is rejected on the spot.
* **This server run** lists every pass made since the console started — time,
  label, mode, density, duration and clip count. **Discard run** stops the
  current pass and deletes everything it recorded; the ✕ on a finished pass does
  the same for it. Both remove the WAVs, their sidecars and their library rows,
  so a pass that turned out wrong never reaches the dataset. The list itself is
  in memory only: closing the console forgets the list, not the clips.
* Shortcuts: `Space` capture now, `R` start/stop, `U` undo last, `1`–`5` density.

### Terminal recorder

```bash
uv run sed record \
  --project <project_id> \
  --class cheers_quiet \
  --num 200 \
  --density normal \
  --onset-threshold -80 \
  --tags "wine-glass,quiet" \
  --description "quiet wine glass near bar" \
  --channels 0,2 \
  --split
```

### Options
* `-c, --class <label>`: Sound label to save under.
* `-p, --project <id>`: Optional project config association.
* `-n, --num <count>`: Stop after this many onset-triggered clips (0 for unlimited). Default: 50.
* `-du, --duration <time>`: Wall-clock cap (e.g., `15m`).
* `-de, --density <level>`: Noise density context (`empty` | `quiet` | `normal` | `busy` | `peak`).
* `-ch, --channels <indexes>`: Override channel indexes to record (e.g., `0` or `0,2`).
* `-ot, --onset-threshold <db>`: Override onset trigger floor in dB.
* `--split`: Auto-split multichannel recordings into per-channel mono clips on save.
* `-t, --test`: Mark clips as frozen test material (requires `--project`).
* `--serve`: Open the recording console instead of recording in the terminal.
  `--port` changes the port (default 8790).

With `multi_mic_trigger = "any"`, each selected mic triggers independently; near-simultaneous onsets save as one multichannel WAV, and the cooldown blocks every mic from re-triggering until it expires.

---

## 4. Manage the Sound Library

Manage and process clips in the shared data library (`data_library/`).

### Import Existing Clips
```bash
uv run sed library import <src_dir> --class <label> [--project <project_id>] [--density normal] [--test] [--move]
```

### Split Multichannel WAVs
The original is kept; mono clips land under `sounds/<label>/<label>_ch<c>/` with sidecars linking back to their parent.
```bash
uv run sed library split-channels --all                  # Split all multichannel clips
uv run sed library split-channels --class <label>        # Split all clips under a label
uv run sed library split-channels --uuid <uuid>          # Split one clip by library UUID
uv run sed library split-channels --all --force          # Re-split, overwriting existing mono clips
```

### Index & Stats
```bash
uv run sed library stats                 # Label and category counts
uv run sed library rebuild-manifest      # Re-index the manifest from WAV sidecars
```

### Audit Augmentations (needs a trained model)
Once a model exists, score every derived clip with the live runtime gates to see which augmentations it recognizes — broken down per recipe, mask-dB bin, and effective-SNR bin. Results appear in the Inspect Augmented tab.
```bash
uv run sed evaluate augmented --project <project_id>             # active model
uv run sed evaluate augmented --project <project_id> --exported  # score the OpenVINO IR
```

---

## 5. Prepare the Dataset

The Inspect web tool is where almost all dataset work happens now:

```bash
uv run sed dataset inspect --project <project_id> --serve
```

* Review clips: delete bad samples, or assign them to target classes.
* Build augmentation recipes: pick base + overlay clips, tune the panel, then **Generate**.
* Browse frozen-test predictions, confusion matrix, and training curves by run.
* In-app tooltips explain the mask-filter and effective-SNR gates.

---

## 6. Train the Model

Compose the dataset, sanity-check it, then train:

```bash
uv run sed dataset compose --project <project_id>   # Build train/val/test CSVs from selections
uv run sed check train --project <project_id>       # Pre-train check (stats, leakage, negatives)
uv run sed train --project <project_id> --description "baseline"
```
A model bundle lands in `projects/<project_id>/models/<model_id>/`. Add `--no-test` to skip evaluation or `--no-export` to skip ONNX/OpenVINO export.

---

## 7. Evaluate and Select Model

Regenerate the frozen-test report, compare models, and pick the active one:

```bash
uv run sed evaluate --project <project_id>                   # Active model
uv run sed evaluate --project <project_id> --model-id <id>   # One model (full id or #N handle)
uv run sed evaluate --project <project_id> --all             # Every model
uv run sed models compare --project <project_id>             # Side-by-side metrics + epochs/artifacts; copy full IDs from the header
uv run sed models select --project <project_id> <model_id>   # Set the active model
```
After changing the frozen test set, re-`compose` then `evaluate --all` so every model is scored on the same test. To re-score a single model on the current frozen test without the CLI, use the **Evaluate** button in the Inspect **Frozen test** tab (after Compose) — it refreshes that model's predictions and confusion matrix.

---

## 8. Calibrate Thresholds

Sweep per-class thresholds against recorded background clips to find the lowest that stays under your false-positive budget.

```bash
uv run sed calibrate --project <project_id> --target-per-1000 1.0 --negative-source venue
```
The budget is **per 1000 negative clips**, not per hour: negatives are captured as onset-triggered clips, so clips are the denominator the recorded material actually supports. No continuous recording is needed.

`--negative-source venue|library` or `--negative-dir <path>` selects the background source.

---

## 9. Test Live Inference on the Workstation

Run the deployed OpenVINO model locally with ArtNet off — catches model-load, audio-device, and threshold issues while the training environment is still in front of you.

```bash
uv run sed check deploy --project <project_id>
uv run sed inference --project <project_id> --verbose live --no-artnet
```
Use `--target cheers_loud,cheers_quiet` to narrow active triggers. Stop with `Ctrl+C` once peaks, probabilities, and `DETECT` lines look right.

The live bars refresh every `[inference] hop_ms` (default 300 ms). With `onset_gating_enabled`, bars update each hop but a trigger fires only on a detected onset.

### Mine Detections (collect hard examples live)

Save detected clips during live inference for review. Set which classes to mine in `[targets]` (`mine = true`, including `_negative`); mining shares each class's `threshold`.

```toml
[targets]
cheers_loud = { enabled = true, threshold = 0.9, mine = true, channel = 1, value = 255 }
_negative   = { enabled = false, mine = true, threshold = 0.5 }   # capture false positives
```

```bash
uv run sed inference --project <project_id> --verbose live --no-artnet --mine
```

* Saves `data_library/mined/<predicted_class>/<clip>.wav` + a trainable mined sidecar labeled as the predicted class.
* Only onset-gated windows above the class threshold are saved (live bars show a `Mined:` counter).
* Mined buckets appear in Inspect Library as `bucket=mined` and are automatically assigned to `_negative` per inference session. Preview them in the Inspect **Mined** tab and remove or reassign the source rules as needed.

---

## 10. Deploy to Edge

1. **Rsync the model bundle:**
   ```bash
   rsync -av projects/<project_id>/models/<model_id>/ <edge-user>@<edge-host>:/home/sed/SoundDetection2/projects/<project_id>/models/<model_id>/
   ```
2. **Activate and check on the edge:**
   ```bash
   sudo -u sed /home/sed/SoundDetection2/.venv/bin/sed models select --project <project_id> <model_id>
   sudo -u sed /home/sed/SoundDetection2/.venv/bin/sed check deploy --project <project_id>
   ```

---

## 11. Run on Edge

```bash
systemctl status sed                      # Service status
journalctl -u sed -f                      # Follow logs
```

### Interactive Debugging
```bash
sudo systemctl stop sed
sudo -u sed /home/sed/SoundDetection2/.venv/bin/sed inference --project <project_id> --verbose live --no-artnet
```

---

## Troubleshooting

- **No input signal:** Run `sed audio devices` and `sed audio meter`, and verify `[deploy] record_device`.
- **No input signal / wrong channels:** `sed record --serve` shows a live level per channel next to its checkbox, so you can see which inputs are actually connected. Channel indexes are 0-based: interface input N is index N-1.
- **Trigger too sensitive / not firing:** Drag the onset floor in the recording console (or press Auto-calibrate) and watch the `flux vs gate` readout; adjust `-ot` when recording from the terminal, or `[onset] mad_k` / `min_flux_db` in config.
- **Empty Augmented list in Inspect:** Assign your generated clips to a class (bucket `derived`) in the Inspect UI.
- **Too many false positives:** Re-calibrate with a lower budget: `sed calibrate --project <id> --target-per-1000 0.5`. If the model is firing on cutlery or glass handling specifically, that is a class-space problem rather than a threshold one — see `docs/pretraining_guide.md` on suppressor classes.
- **Edge fails to load or runs the wrong model:** Run `sed check deploy` on the edge; set the model with `sed models select --project <id> <model_id>` (copy IDs from `sed models compare`).
