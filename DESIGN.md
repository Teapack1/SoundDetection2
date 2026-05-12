# Design Notes

This document explains the technical decisions in the codebase, especially
the ones tuned for **minimising false positives even at the cost of recall**.

## Configuration layering and snapshots

Settings live in two layers and a snapshot:

1. **`configs/default.toml`** — defaults for every knob (~80 settings).
   Edited rarely; changes apply to *new* training runs.
2. **`projects/<id>/project.toml`** — per-venue overrides: classes, ArtNet
   target, target → channel/value mapping, calibrated thresholds, per-class
   SNR floors. `load_config(project_id)` deep-merges (1) ← (2).
3. **`projects/<id>/models/training_config.json`** — full effective config
   snapshot written by `clinkd train` immediately before `Trainer.fit()`.
   This is the **contract**: it records exactly what the model was trained
   against so subsequent inference is reproducible against the same
   parameters, even if `default.toml` is later edited.

`clinkd run` calls `load_config_for_inference(project_id)`, which prefers
the snapshot over the live config when present. Operational settings
(`[artnet]`, `[audio]`, `[thresholds]`, `[logging]`) are still re-read from
`project.toml` on top of the snapshot so the lighting console IP or audio
device can change post-deployment without retraining.

This closes the silent-failure mode where editing `default.toml` after
training (say, bumping `n_mels` from 64 to 128) would produce a shape
mismatch or accuracy drop at inference. The runtime now uses the
trained-against values, full stop.

The OpenVINO `models/manifest.json` is the second half of the contract: it
records feature parameters (sample_rate, mel config) AND an
`inference_baseline` block (smoothing, gates, refractory) so the runtime
loads the model with the same downstream behaviour calibration was tuned
against. Per-class thresholds remain in `project.toml [thresholds]` —
they're venue-tuned and *should* change without retraining.

Tests: `tests/test_config_snapshot.py` covers round-trip, snapshot-wins-
over-live, operational-overrides-still-apply, and fallback-when-no-snapshot.

## False-positive minimization stack

Five independent layers, each tightening the gate:

### 1. Loss function: asymmetric focal loss (softened)

`src/sed/train/model.py :: AsymmetricFocalLoss`

Per-class focal loss with **γ_neg = 2, γ_pos = 1** plus a **1.5× weight on
the `_negative` class**. The asymmetry is real but deliberately mild:
- when the model is confidently wrong on a negative, the loss is amplified
  (modulator p² vs p¹) but not extreme
- when it's wrong on a positive (missing a fine quiet clink), the loss
  retains enough gradient that the model continues to learn the positive
  decision boundary
- the optimiser is biased toward "miss a clink" over "false positive on
  cutlery", but only by a factor of ~2× — not 8× as it was in iteration 1

**Why softened** (changed from γ_neg=4, neg_weight=2.0 in iteration 1): a fine
wine-glass clink at -5 dB SNR has only a few correctly-classifiable frames in
its window. With γ_neg=4 the gradient was so dominated by negatives that
those rare correctly-detected positive frames were getting essentially zero
backward signal — the model learned to be confident about silence and
ambivalent about real clinks. γ_neg=2 keeps the FP-prevention bias while
preserving learning signal on quiet positives. The runtime gates (negative
gate, margin, calibrated thresholds) handle the rest.

### 2. No class-balanced batches

`src/sed/train/datamodule.py :: SedDataModule.train_dataloader`

`shuffle=True`, no `WeightedRandomSampler`. The natural distribution
(heavily-negative) is preserved, so the model sees realistic class proportions
during training and the focal loss receives genuine gradient signal on
negatives. Class-balanced sampling would *over-sample* positives, which
inflates F1 in lab metrics but hurts deployed FP rate.

### 3. Multi-class model with explicit `_negative` head

`src/sed/train/lit_module.py`

Multi-label sigmoid (BCE) over all classes, but `_negative` is one of them.
At inference:
- the per-class smoother produces a smoothed score for `_negative` like any
  other class
- the arbiter (next layer) uses it as a **gate**: if smoothed `_negative` >
  `negative_max` (default 0.5), trigger is suppressed regardless of any
  positive score

This is more discriminative than a closed-set softmax: it lets the model say
"I see *something* but it doesn't match any target — go find another match."

### 4. Smoothing chain: median → EMA → optional consensus

`src/sed/runtime/smooth.py :: ClassSmoother`

```
raw probs  →  median(window=3)  →  EMA(α=0.4)  →  ≥ N consec above threshold
                                                  ↓
                                           hysteresis_low arms re-fire
```

A real wine-glass clink lasts 300–800 ms and lives in 3–8 inference windows
at hop=100 ms. The median + EMA already removes single-frame spikes from
music transients. By default we use **`consecutive_frames=1`** (immersive,
single-frame trigger) so a fine quiet clink fires immediately the first
window the smoothed score crosses threshold.

**Why the relaxed default** (was 2 in iteration 1): at consec=2 the user
perception was "too slow / sometimes missed" for fine clinks, because a
soft clink may only have 1-2 frames where its smoothed score actually clears
threshold before decaying. consec=1 restores that immersive feel; the
margin-to-second-best gate, the negative gate, and per-venue threshold
calibration carry the FP defence. Crank back to consec=2 in the project
TOML if a venue's specific FP profile demands it.

### 5. Arbiter gates — margin + negative gate + global refractory

`src/sed/runtime/smooth.py :: TriggerArbiter`

For a trigger to fire, ALL of these must hold simultaneously:

| Gate | Default | Purpose |
|------|---------|---------|
| smoothed score ≥ threshold | per-class, calibrated (~0.7–0.85) | basic confidence |
| `_negative` score ≤ `negative_max` | 0.5 | explicit "not other stuff" gate |
| winner ≥ runner-up + margin | margin = 0.08 | reject ambiguous events |
| consecutive_frames hit | 1 (immersive) | single-frame trigger |
| no fire in last `refractory_ms` | 300 ms | one event = one trigger |

Crucially, the smoother and arbiter are **decoupled**: an arbiter rejection
does NOT consume the smoother's "armed" state. Only an actual emitted
TriggerEvent disarms the smoother (via `consume_fire()`). This means a margin
rejection on frame N doesn't prevent a clean fire on frame N+1.

### 6. In-venue threshold calibration

`src/sed/runtime/calibrate.py`

The per-class thresholds in `project.toml` are NOT defaults — they are
**measured** by running the trained model over real recorded ambience and
sweeping thresholds to find the lowest per-class threshold that yields ≤ N
false-positives-per-hour. This is the single biggest contributor to deployed
FP rate.

Because the model has the asymmetric-focal training prior, calibrated
thresholds typically end up at 0.7–0.85 (not 0.6) for an FP/h target of 1.

## Multi-glass discrimination

The model learns each glass type as a separate class. Discrimination quality
depends on:

1. **Spectral separability** of the glass types. A thin wine glass vs. a thick
   tumbler differ a lot (peak frequency, decay shape). A thin wine vs.
   thin-walled water glass may not — accept this and treat them as one class.
2. **Inter-class margin** (`margin_to_second_best`). Default 0.10 — when two
   glass classes both score above threshold and are within 0.10 of each other,
   no fire. Increase to 0.15 if you keep seeing wrong-class triggers; recall
   drops accordingly.
3. **Consistent labelling**. The composer splits by session_id; if session A
   was "wine_thin" but the operator accidentally clinked a wine_thick once,
   that single mislabel poisons val. The HTML inspector + UMAP scatter
   surfaces these.

Realistic accuracy for 3-class wine discrimination in a noisy bar with
~500 positives per class:
- per-class F1 0.78–0.90 (the harder pairs lower)
- inter-class confusion 5–15 %
- false-positive rate 0.5–3 / hour at calibrated thresholds

## Domain mixing — the augmentation that matters most (now with masking check)

`src/sed/data/augment.py :: WaveformAugmenter`

For every training positive, with 95 % probability:

```
augmented_positive = clean_positive + ambience_clip * scale_factor(SNR_dB)
```

with SNR drawn uniformly from [-2, +25] dB. The negative side is the random
ambience clip drawn from `data_library/ambience/` (project-agnostic) plus
the project's own ambience.

This synthesises **unlimited** in-situ-realistic positives from a small pool
of clean recordings. It's the difference between "model works in a quiet
office" and "model works in a busy bar".

A second mix (~20 %) adds an external negative from
`data_library/negatives_external/` — this teaches the model that
"clink + cutlery in the same window = still target", which is a real
scenario at peak hours.

### Masking-aware safety net

A naive uniform-SNR mix has a pathological failure mode: if the clean
positive is a fine, quiet clink (a thin wine glass, soft strike) and the
ambience clip is loud crowd-talking, the talking can completely mask the
target. The label still says "positive" but the audio has no clink-like
content for the model to learn — over thousands of such samples the model
learns "loud talking = positive class", and you've destroyed your training
distribution from the inside.

Two defences, both on by default:

1. **Per-class minimum SNR floor** (`per_class_min_snr_db` in config). For
   fragile classes (fine glass, soft tap), set a higher floor than the
   global -2 dB minimum. Set in the project TOML:
   ```toml
   [augmentation.per_class_min_snr_db]
   clink_wine_thin = 3.0      # thin wine: ≥ +3 dB SNR
   clink_champagne = 1.0      # bigger transient, ≥ +1 dB OK
   ```

2. **Online masking check** (`masking_check = true` in config). Every domain
   mix is scored with `masking_score_db()` — a transient-dominance metric
   that finds the target's brightest time-frequency cell, then measures how
   much that band sticks above the median band-energy in the mixed frame at
   the same time. Higher = better (the target's transient survived).
   Threshold: `masking_min_db = 3.0` dB. If the score is below floor, the
   mix is **skipped** and the clean target is used instead. The augmenter
   tracks `mixes_skipped_masking` in stats so you can see if a class is
   producing too many bad mixes (re-record cleaner positives or raise the
   per-class SNR floor).

Both layers are visible in the dataset HTML report: `clinkd inspect` includes
an "Augmentation Preview" section that renders ~12 augmented samples per
target class, with side-by-side spectrograms (clean vs mixed), masking
score color-coded green/yellow/red, and audible WAV playback. If >30% of
samples are flagged red, a warning banner appears with concrete advice
(record more positives, raise the SNR floor, expand ambience pool).

## Multi-microphone inference

`src/sed/runtime/infer.py :: RealtimeInference`

Up to 4 microphones can stream simultaneously. The default in `default.toml`
is single-mic (`record_channels = 1`), but a venue with multiple zones can
configure 2–4 channels and they're all processed every frame:

1. Read `N_channels × window_samples` from the ring buffer per cycle.
2. Stack mel features into a batch of size N (one OpenVINO call, batched
   inference is cheap on Intel CPUs — INT8 mn10 batch=4 still fits in the
   100 ms hop budget on a CL260).
3. Pool per-class probabilities across mics by `multi_mic_mode`:
   - `"max"` (default) — any mic clearly hearing the target wins. Best for
     "any mic detects → trigger ArtNet" semantics.
   - `"mean"` — average; more conservative, requires multiple mics to agree.
   - `"single"` — only `inference_channel` is used; back-compat single-mic.
4. The single arbiter sees the maxed (or meaned) per-class probabilities and
   makes one decision per frame.

Per-mic stats are tracked (`per_mic_max_prob` EMA, `per_mic_triggers`) and
shown in the live runtime table — a dead mic shows up as 💀 instead of 🎤
within a few minutes. Useful for catching unplugged or failing microphones
without watching the audio interface.

## Library system: factory clean ↔ on-site reality

The library (`data_library/`) is the persistence layer for sounds that are
**not** project-specific:

- `glasses_clean/` — studio recordings of physical glasses; reused across
  every venue that uses the same glassware.
- `ambience/` — bar/restaurant ambience from any venue; pooled at training
  time as the domain-mixing source. More variety → better generalisation.
- `negatives_external/` — cutlery, ice, music, AudioSet samples; project
  agnostic.

A new venue typically composes its dataset from:
- 100 % of its on-site positives (small pool, ~200–500 per class)
- 100 % of its on-site hard-negatives + ambience
- ~50–100 library glass clips per class (provides spectral variety)
- the entire library ambience pool, used at augmentation time only

This mirrors the human intuition: the spectral signature of "wine_thin
clinking" comes from the glass (library), the realism of "wine_thin clinking
in this venue" comes from on-site recording.

## The empty-venue reality and why this workflow is sound

System install almost always happens **before the venue opens**: HVAC on,
dim lights, no crowd, music off. Nothing about that environment resembles
the deployed conditions the model will face. Anyone proposing "just record
during a busy night and train on that" hasn't tried to schedule install
around a venue's revenue hours.

The pipeline takes this constraint as the default case, not a fallback:

1. **At the venue (one short visit, before opening)**: clean on-site
   positives anchored in the venue's actual reverb, plus 20–40 minutes of
   empty-venue acoustics — room tone, HVAC, whatever's running with no
   crowd. That's all that needs to come from the venue itself.
2. **Off-site, ahead of time, anywhere with permission**: build a library
   of busy ambience and hard-negatives by recording at other bars,
   restaurants, cafés you visit normally. This pool is shared across every
   future project — record it deeply once.
3. **At training time**: domain-mixing combines clean positives with
   randomly drawn library ambience at SNRs in [-2, +25] dB, gated by the
   masking-aware safety net.
4. **At calibration time**: `--ambience-source simulated_busy` mixes the
   empty-venue room tone with library busy-ambience at SNRs in [-3, +12] dB
   to synthesise a 30-minute peak-hour stream, then sweeps thresholds
   against it. Realistic FP/h estimates without ever recording a real busy
   night at the target venue.
5. **After deployment** (optional second visit during real busy hours):
   recapture an hour of actual ambience and re-run `clinkd calibrate
   --ambience-source venue` to refine. The model is unchanged; only the
   thresholds shift, typically downward.

The closest analogue is far-field speech recognition: nobody records every
speaker in every café where Alexa might be deployed. Instead clean speech
is captured in studios, reverb impulse responses and crowd noise are
captured separately, and the training pipeline composes them stochastically.
The same logic applies to glass clink detection — the clink's transient is
the spectral signature, the venue contributes reverb and noise floor, and
the pool of library ambiences contributes the variability that makes the
model generalise to peak hours it never directly heard.

What makes this work without producing garbage training data is the
masking-aware safety net (per-class SNR floors + transient-dominance
check) — the augmenter cannot silently produce a "loud talking labelled
positive clink" sample even if the random ambience draw is unfortunate.
Combined with the dataset HTML report's augmentation preview (clean vs
mixed spectrograms with masking-score color coding), the operator can
visually verify that synthesis is producing audible targets before
training starts.

## Why log-mel and not PCEN

PCEN (per-channel energy normalisation) is excellent for stationary noise,
documented to reduce FAR 5–50× in bioacoustics. But:
- bars have non-stationary music + speech; PCEN's adaptive AGC can compress
  transient onsets (the very thing we want)
- EfficientAT is pretrained with log-mel; switching front-ends throws away
  the pretrained weights' usefulness
- the domain-mixing augmentation already covers most of what PCEN provides

If a specific venue is dominated by stationary HVAC noise and FP/h is high,
PCEN as a second branch is a future experiment — not a default.

## Why EfficientAT and not the alternatives

| Model | Why it's not the default |
|-------|-------------------------|
| YAMNet | mAP 0.31 too low; TF dependency awkward on Linux+OpenVINO |
| PANNs CNN14 | 80M params, slow on CL210 |
| AST / PaSST / BEATs | 86–90M params, attention layers don't quantise to INT8 cleanly, fragile to fine-tune on small data |
| BirdNET | non-commercial license |

EfficientAT `mn10_as` (4.9M params, ~0.47 mAP, MobileNetV3-based, distilled
from a PaSST teacher) is the Pareto frontier. INT8-quantises cleanly,
fine-tunes well on small datasets, MIT-licensed, runs in 10–30 ms on Intel.

## Latency budget

| Stage | Typical | Worst |
|-------|---------|-------|
| Mic acoustic flight (overhead → diaphragm) | 5–10 ms | 15 ms |
| ADC + USB mic buffer | 10–20 ms | 30 ms |
| ALSA period (1024 samples @ 48 kHz input) | 21 ms | 21 ms |
| Wait for next 100 ms inference tick | 50 ms (avg) | 100 ms |
| Mel spectrogram | 5–10 ms | 15 ms |
| OpenVINO INT8 mn10 forward | 10–25 ms | 50 ms |
| Smoothing + arbiter | <1 ms | <1 ms |
| ZMQ + ArtNet UDP send | 1–5 ms | 10 ms |
| DMX fixture latency | 25–50 ms | 100 ms |
| **Total** | **140–200 ms** | **350 ms** |

Sits comfortably under the 500 ms target.

## Things this design intentionally does NOT do

- **No real-time noise suppression / RNNoise**: would compress transients,
  defeating the point. The model is trained on noisy audio.
- **No automatic gain control**: SNR variability is part of the training
  distribution; AGC would shift the test distribution.
- **No two-stage cascade** (cheap detector → fancy classifier): adds latency
  and complexity. The model is small enough to run on every window.
- **No adaptive runtime threshold** based on recent ambience: tested in
  earlier iterations; unstable. The static-threshold + calibration approach
  is more reliable. If a venue's ambience drifts (new music system, new menu
  with different cutlery), recalibrate.
