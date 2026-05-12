# Research brief — glass-cheers detection: pretraining evaluation & next steps

*Written 2026-07-11 from the 22 comparable runs in `projects/pretraining/models/`.*

## 1. Run evaluation

All 22 runs share the frozen benchmark test hash `…3699eb6604ee` (regenerated
session-grouped on 2026-07-04, before every run), so their benchmark scores are
directly comparable. Threshold-*independent* AP is the architecture signal;
benchmark F1 additionally reflects each run's calibration.

Top runs (benchmark = frozen pretraining test):

| run | backbone | init | feature | bench F1 | bench AP | cheers AP | cheers P/R |
|---|---|---|---|---|---|---|---|
| 07-08_2150 | mn04_as | random | mel128 @5 ms | **0.9747** | 0.9970 | 0.9959 | 0.968 / 0.985 |
| 07-08_2058 | mn10_as | random | linear 96×96 | 0.9705 | 0.9953 | 0.9929 | 0.981 / 0.939 |
| 07-08_1437 | mn20_as | upstream | mel128 @10 ms | 0.9525 | 0.9977 | 0.9956 | 0.989 / 0.902 |
| 07-07_2207 | mn10_as | upstream | mel128 @10 ms | 0.9429 | 0.9960 | 0.9923 | 0.988 / 0.859 |
| 07-10_0739 | mn10_as | random | mel-441khz, CE | 0.9395 | 0.9952 | 0.9969 | 0.974 / 0.985 |
| 07-09_2037 | transient-resnet | random | mel-441khz | 0.9247 | **0.9982** | **0.9974** | **0.997** / 0.918 |

Findings:

- **AP is saturated and tight (0.990–0.998) across architectures** — the corpus
  is highly separable, and most of the F1 spread between runs is *threshold
  calibration*, not architecture. Runs evaluated before val-calibration existed
  (`config_defaults` thresholds) show artificially low recall (0.70–0.90) that
  the AudioSet-pretrained runs suffered most from.
- **transient-resnet (0.68 M params) has the best AP of any run** (macro 0.9982,
  cheers 0.9974) and the best cheers precision (0.997 at recall 0.918) — the
  strongest false-positive-averse profile in the table, at 1/7 the size of mn10.
- **The loss ablation is within noise**: identical mn10 configs scored bench F1
  0.915 (BCE) / 0.940 (CE) / 0.929 (CE alias) / 0.930 (asymmetric focal). No
  loss is clearly superior on this data; BCE stays the default because it is
  the only objective that composes with mixup and multi-label targets.
- The mel-441khz preset (128 mel, 5 ms hop @ 44.1 kHz) and efficientat-5ms
  (hop 160 @ 32 kHz) both preserve transient structure well; 5 ms hop variants
  consistently beat 10 ms on cheers recall.

## 2. Recommendation

**Primary track — `mn10_as`, AudioSet upstream init, full fine-tune with head
warm-up.** This is the strongest *transfer base* for the pipeline's real goal:
the AudioSet → domain-pretrain → venue-fine-tune chain, ending in a ~500
clips/class fine-tune on three glass types. Concretely:

- Feature: mel 128 bands @ 5 ms hop (`mn10_as_hop_5` checkpoint contract) —
  fine time resolution for glass transients while staying on a published
  AudioSet contract.
- Training: `domain_pretrain` preset — BCE + mixup 0.3 + SpecAugment,
  lr_backbone 3e-5 / lr_head 1e-4, layer decay 0.75; val-calibrated
  thresholds + auto-test (the preset now sets these automatically).
- The upstream runs' lower bench recall in the table is a *pre-calibration
  artifact*, not an architecture deficit — their APs are top-tier.

**Challenger track — `transient-resnet` from scratch.** Best AP + best
precision of any run at 0.68 M params. Keep training it in parallel on every
dataset iteration; it is the natural ensemble partner and a distillation
target for edge deployment.

**Venue fine-tune stage (3 glass types × ~500 clips):**

- Initialize *from the domain-pretrained run* (`initialization = from previous
  run`), never directly from AudioSet.
- `venue_finetune` preset: last ~30 % of encoder stages, head warm-up,
  freeze-BatchNorm, lr_backbone 5e-5 — now also enables val-calibrated
  auto-test, **BN recalibration** (re-measures BatchNorm statistics on venue
  data after training), and FP/h calibration on venue negatives.
- Augment aggressively: pitch ±1–2 semitones, venue-noise overlays at
  0–20 dB SNR, SpecAugment. 500 clips/class is enough for last-third
  fine-tuning of mn10; it is *not* enough for full fine-tuning without
  forgetting — compare both before trusting a full-depth run.
- Distinguishing three glass types is fine-grained: expect the confusion to
  concentrate between the two most similar glasses. Add hard-negative mining
  between glass classes (mined FPs of one class fed back as negatives for the
  other) before adding more raw data.

**Evaluate at FP/h operating points, not F1 alone.** The deployment constraint
is "quiet cheers in a loud venue with almost no false triggers" — track
recall @ FP/h ≤ 1 as the primary deployment number (the FP/h calibration sweep
already computes this trade-off).

## 3. AI research agent prompt

Copy everything between the lines into a capable research agent/LLM together
with the current Runs & Compare export.

---

You are a machine-learning research advisor for a production sound-event
detection system. Work from the facts below; when you recommend something,
give exact values/ranges, the reasoning, and the measurement that would
confirm or reject it.

**System.** Detect glass-cheers ("cheers", plus an "attention" class and an
explicit `_negative` class) in loud hospitality venues, on CPU (OpenVINO
preferred, ONNX/PyTorch fallbacks), from 1 s audio windows either sliding or
onset-gated by a spectral-flux trigger. Decision layer: per-class thresholds +
negative-gate + margin, calibrated (a) on the validation split for max-F1 and
(b) against negative audio to a target false-positives-per-hour.

**Data.** Domain corpus ≈ 66 k train clips (session-grouped splits, leakage
audited), frozen benchmark test regenerated 2026-07-04. Downstream task:
distinguish 3 glass types with only ~500 clips per class recorded on-venue.
Augmentation pipeline: pitch shift ±2 st, SNR-controlled noise overlays
(0–20 dB), positive/negative overlay composition, SpecAugment at train time.

**Current best results** (frozen benchmark; AP threshold-independent):
mn04_as scratch mel128@5ms F1 0.9747 / AP 0.9970; mn10_as AudioSet-pretrained
full-FT AP 0.9960; mn20_as pretrained AP 0.9977; transient-resnet (0.68 M,
scratch) AP 0.9982 with cheers precision 0.997 @ recall 0.918. Loss ablation
(BCE vs CE vs asymmetric focal) within ±0.02 F1. Baseline architecture chain:
AudioSet mn10_as → domain pretrain (full FT, BCE+mixup+SpecAugment) → venue
fine-tune (last ~30 % stages, frozen BN + post-hoc BN recalibration,
calibrated thresholds).

**Available knobs** (exact names): lr_backbone, lr_head, lr_layer_decay,
epochs, warmup_epochs, fine_tune_policy (head/last_stage/last_two/last_n:k/full)
+ fine_tune_warmup_epochs, freeze_batchnorm, bn_recalibrate, loss (bce_logits /
cross_entropy / asymmetric_focal with focal_gamma_pos/neg), negative_weight,
mixup_alpha, specaugment_freq_mask/time_mask, weight_decay, calibrate_thresholds,
calibrate_after (FP/h target + venue/library negatives). Feature knobs: mel
bands (40–256), hop (5–25 ms), sample rate, PCEN, standardized log-mel
(z-score), linear spectrograms, colormap-image features.

**Your tasks:**

1. **Hyperparameter plan for the 500-clip 3-glass fine-tune.** Propose a
   concrete search space (values, not vague ranges) over depth policy,
   lr_backbone/head, layer decay, mixup, SpecAugment mask sizes, and epochs
   for a ~1 500-clip 4-class (3 glass + `_negative`) fine-tune from a
   domain-pretrained mn10_as. Prioritize the 5–8 runs that give the most
   information first. State the expected failure mode of each corner
   (forgetting, overfit, under-adaptation).
2. **Loss choice.** Given the ablation showed ±0.02 F1 between BCE / CE /
   asymmetric focal on the (easy) domain corpus: argue which objective is the
   right default for the *fine-grained glass* stage where classes are similar
   and hard negatives dominate, and what γ+/γ− (or negative_weight) to start
   from. Include when NOT to use focal.
3. **Small-data protocol.** Design the evaluation protocol for the 500/class
   stage: how to split leakage-free at that scale (session-grouped), how many
   seeds/repeats are needed for a decision, early stopping metric, and how to
   compare against the do-nothing baseline (domain model + per-glass
   thresholds only).
4. **False-positive engineering.** Recommend an ordered plan for driving FP/h
   down at fixed recall: hard-negative mining from deployment logs, negative
   overlay composition, threshold/margin/negative-gate calibration, per-venue
   BN recalibration, and anything we are missing (e.g. energy/SNR gating,
   two-stage verification).
5. **Research roadmap.** Rank the next experiments by expected value:
   (a) BYOL/self-supervised pretraining on the unlabelled part of the corpus
   before supervised domain training; (b) knowledge distillation
   mn10 → transient-resnet for edge; (c) ensembling the two tracks;
   (d) CED Tiny/Mini/Small transformer fine-tunes at native 0.4 s and centered
   1.0 s context; (e) ClinkCNN-S scratch and EfficientSED fMN10/fMN20 challengers;
   (f) streaming/onset-conditioned evaluation improvements; (g) per-venue continual adaptation. For each: cost, expected gain, and the
   metric that would prove it.

Constraints: CPU-only inference; the decision layer (thresholds, margins,
smoothing) is calibrated separately from training — do not fold deployment
thresholds into the training objective; comparisons must use the frozen
benchmark + calibrated decision layer, and deployment claims must be stated
as recall @ FP/h.

---

## 4. Deployment note (engine fallbacks)

OpenVINO IR remains the preferred production engine. Runs without an export
now fall back automatically: **ONNX Runtime** (`model.onnx`, exported with
every run) → **PyTorch checkpoint** (universal, slower). PyTorch's own
TFLite-equivalent for mobile/edge is **ExecuTorch** (`.pte`); it targets
phones/microcontrollers and is not needed for this project's CPU boxes — the
OpenVINO → ONNX → checkpoint chain covers them.
