# WORKFLOW

Two setups: **A** on the workstation, **B** on the edge device at the venue.

---

## A. Training workstation

### A1. Install

```bash
git clone <repo> sed && cd sed
git init && git submodule add https://github.com/fschmid56/EfficientAT vendor/EfficientAT
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra train --extra dev
uv run clinkd --help && uv run pytest
```

### A2. Build the library

Shared studio recordings, reused across all projects. Record once in a quiet room with each glass type, plus ambience and distractors from real venues.

```bash
# Studio glass clips — 60-100 per glass type
uv run clinkd library record-glass --type wine_thin  --num 60
uv run clinkd library record-glass --type wine_thick --num 60
uv run clinkd library record-glass --type champagne  --num 60

# Busy ambience — 3+ different venues, 20-30 min each
uv run clinkd library record-ambience --label friends_bar    --density busy --duration 30m
uv run clinkd library record-ambience --label hotel_lobby    --density normal --duration 30m
uv run clinkd library record-ambience --label coffee_morning --density quiet --duration 20m

# Hard-negatives — sounds that could be confused with clinks
uv run clinkd library record-hardneg --label cutlery      --mode auto --duration 15m
uv run clinkd library record-hardneg --label ice_in_glass --mode auto --duration 15m
uv run clinkd library record-hardneg --label bottle_clank --mode auto --duration 15m

# Check what you have
uv run clinkd library stats
uv run clinkd library ambience-coverage
```

Ctrl+C anywhere is safe — clips are saved per-clip, not at the end.

### A3. Create the project

Each venue/deployment gets its own project. The project config only contains venue-specific overrides — everything else inherits from `configs/default.toml`.

```bash
uv run clinkd project new the_hawthorn
$EDITOR projects/the_hawthorn/project.toml
```

Example `project.toml`:

```toml
[project]
id = "the_hawthorn"
name = "The Hawthorn"

[model]
classes = ["_negative", "clink_wine_thin", "clink_wine_thick", "clink_champagne"]

[targets]
clink_wine_thin  = { enabled = true, channel = 1, value = 85,  effect = "default" }
clink_wine_thick = { enabled = true, channel = 1, value = 170, effect = "default" }
clink_champagne  = { enabled = true, channel = 3, value = 255, effect = "default" }

[thresholds]
clink_wine_thin  = 0.600
clink_wine_thick = 0.600
clink_champagne  = 0.600

[audio]
record_device = "UMC404HD"

[artnet]
target_ip = "192.168.50.40"
```

### A4. On-site recordings

Done on the edge device at the venue (see **B3**). After recording, rsync files back to the workstation.

### A5. Inspect, compose, train, deploy

After recordings are on the workstation, run these steps in order.

```bash
# 1. Inspect recordings and flag/delete bad ones interactively
uv run clinkd inspect --project the_hawthorn --serve

# 2. Compose dataset (flagged clips are excluded automatically)
uv run clinkd dataset compose --project the_hawthorn

# 3. Pre-train check — validates counts, leakage, ambience pool
uv run clinkd check train --project the_hawthorn

# 4. Train — 30-90 min on a single GPU
uv run clinkd train --project the_hawthorn

# 5. Export to OpenVINO INT8 for edge deployment
uv run clinkd export --project the_hawthorn

# 6. Calibrate per-class thresholds
uv run clinkd calibrate --project the_hawthorn \
    --ambience-source simulated_busy --target-fp-per-hour 1.0

# 7. Final deploy check — model loads, latency OK, ArtNet reachable
uv run clinkd check deploy --project the_hawthorn
```

**Inspect** opens a local web UI where you can:
- **Flag** clips (click to toggle) — excluded from `dataset compose` but files stay on disk
- **Delete flagged** (red button) — permanently removes WAV + JSON from disk with confirmation
- **Done** (green button) — stops the server

**Calibrate** `--ambience-source` options:
- `simulated_busy` — mixes empty-venue room tone with library ambience (default, no real busy recording needed)
- `venue` — real busy recordings from `recordings/ambience/` (use after a second visit)
- `library` — library ambience pool only

### A6. Push model to edge

```bash
rsync -av projects/the_hawthorn/{models,project.toml} \
    edge.local:/opt/sed/projects/the_hawthorn/
```

The systemd path-unit auto-restarts the service when the model file changes.

---

## B. Edge device at the venue

### B1. Pick a distro

**Ubuntu Server 24.04 LTS, headless.** Install with only OpenSSH server selected.

### B2. Install

```bash
# From the workstation
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    ./ edge.local:/tmp/sed-source/
ssh edge.local
cd /tmp/sed-source
sudo PROJECT_ID=the_hawthorn ./deploy/install.sh
```

Creates the `sed` user, builds an inference-only venv (~150 MB), installs systemd units.

```bash
systemctl status sed
journalctl -u sed -f
```

### B3. On-site recording

Run before the venue opens, with mics at their final positions.

```bash
# 1. Find and verify audio interface
clinkd audio devices
clinkd audio meter --project the_hawthorn
clinkd check record --project the_hawthorn

# 2. Record empty-venue room tone
clinkd record ambience --project the_hawthorn --duration 30m --density empty

# 3. Record positives at 3-5 table positions
clinkd record positives --project the_hawthorn --class clink_wine_thin  --duration 15m
clinkd record positives --project the_hawthorn --class clink_wine_thick --duration 15m
clinkd record positives --project the_hawthorn --class clink_champagne  --duration 15m

# 4. Venue-specific distractors (optional — library covers generic ones)
clinkd record hardnegs --project the_hawthorn --class espresso --duration 10m
```

Pull recordings back to the workstation, then continue at **A5**:

```bash
rsync -av edge.local:/opt/sed/projects/the_hawthorn/recordings/ \
    projects/the_hawthorn/recordings/
```

### B4. Daily ops

```bash
systemctl status sed
journalctl -u sed -f                # live event log
sudo systemctl restart sed
```

Interactive debug (live probability bars):

```bash
sudo systemctl stop sed
sudo -u sed /opt/sed/.venv/bin/clinkd run --project the_hawthorn
# Ctrl+C to exit
sudo systemctl start sed
```

### B5. Fine-tune with real ambience

Come back during busy hours for better threshold calibration:

```bash
# On the edge, during service
clinkd record ambience --project the_hawthorn --duration 1h --density busy

# On the workstation
rsync -av edge.local:/opt/sed/projects/the_hawthorn/recordings/ambience/ \
    projects/the_hawthorn/recordings/ambience/
uv run clinkd calibrate --project the_hawthorn --ambience-source venue
rsync -av projects/the_hawthorn/project.toml \
    edge.local:/opt/sed/projects/the_hawthorn/project.toml
ssh edge.local sudo systemctl restart sed
```

---

## Tuning reference

### Onset sensitivity (recording)

Controls when `clinkd record positives/hardnegs` triggers and saves a clip. Full documentation in [`configs/default.toml`](configs/default.toml) under `[onset]`.

| Symptom | Fix |
|---|---|
| Fires on everything | Raise `mad_k` (→ 6-8) or `min_flux` (→ 0.03) |
| Clinks not detected | Lower `mad_k` (→ 3) or `min_flux` (→ 0.002) |
| Same clink saves 2-3 clips | Raise `refractory_ms` (→ 1000) |

### Training augmentations

All in [`configs/default.toml`](configs/default.toml) under `[augmentation]`. Override per-project in `project.toml`. Set any probability or amount to 0 to disable individually.

| Symptom | Fix in `project.toml` |
|---|---|
| FP on cutlery/dishes | `external_negative_mix_prob = 0.3` |
| Fine glasses buried under bar noise | `per_class_min_snr_db = { clink_wine_thin = 5.0 }` |
| Disable all waveform augmentation | Set `pitch_shift_prob`, `time_stretch_prob`, `time_shift_ms`, `gaussian_noise_amp` to 0 |
| Disable domain mixing entirely | `domain_mix_prob = 0` |

Preview augmented samples in `clinkd inspect --serve` under the "Augmentation preview" section.
