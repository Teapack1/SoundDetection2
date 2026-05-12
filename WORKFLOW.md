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

---

## 3. Record Sounds

Record one class at a time. Clips are saved to `data_library/sounds/<label>/` and reused across projects.

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

Sweep per-class thresholds against background clips to find the lowest that stays under your false-positive budget.

```bash
uv run sed calibrate --project <project_id> --target-fp-per-hour 1.0 --negative-source venue
```
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
- **Trigger too sensitive / not firing:** Adjust `-ot` (onset threshold) when recording, or `[onset] mad_k` / `min_flux_db` in config.
- **Empty Augmented list in Inspect:** Assign your generated clips to a class (bucket `derived`) in the Inspect UI.
- **Too many false positives:** Re-calibrate with a lower rate: `sed calibrate --project <id> --target-fp-per-hour 0.5`.
- **Edge fails to load or runs the wrong model:** Run `sed check deploy` on the edge; set the model with `sed models select --project <id> <model_id>` (copy IDs from `sed models compare`).
