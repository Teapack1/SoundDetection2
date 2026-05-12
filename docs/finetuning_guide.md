# Fine-tuning guide — domain pretraining → fine-grained venue runs

*How to configure the second stage of the two-stage workflow: a small,
domain-specific run initialized from a previous run's encoder, whose job is not
"detect a clink" but "tell one glass from another".*

Everything here is reachable from the Train tab. The **Recommendations** panel at
the top of that tab reports the same findings live for your current config
(`sed.train.advice`), so treat this document as the reasoning and the panel as
the checklist.

---

## 1. Why the second stage is a different problem

Stage one (domain pretraining) asks: *is this a glass clink or the room?* The
negative class is everything else, the corpus is large, and the decision is
coarse. Almost any competent encoder solves it.

Stage two asks: *which glass?* The classes are variants of one event. They share
their onset, their envelope, their coarse spectral shape, and the room they were
recorded in. What separates them is the **partial comb** — a set of narrow, high
resonances whose spacing and decay depend on the vessel's geometry and wall
thickness.

Three consequences drive every recommendation below:

1. **The evidence is narrow and high.** Anything that blurs frequency (wide mel
   bands, a low `fmax`, wide SpecAugment frequency masks) removes the only thing
   that distinguishes the classes, while leaving detection accuracy intact. You
   get a run that detects clinks perfectly and cannot name them.
2. **The evidence is mid-level, not top-level.** The last encoder stage carries
   abstract "this is a clink" features. Partial structure lives further down. A
   head-only or single-stage fine-tune has no mechanism to represent the
   distinction, no matter how long it trains.
3. **The dataset is small and single-condition.** Every statistic estimated from
   it — BatchNorm running moments especially — describes the room at least as
   much as the sound.

---

## 2. Model and feature

**Architecture.** Stage two must use the *same* backbone and the same
architecture parameters as its parent: `parent_compatibility()` enforces it,
because the transferred weights are meaningless otherwise. So the architecture
decision belongs to stage one. For this task pick an encoder with enough stage
granularity to fine-tune partially — EfficientAT `mn*` exposes 17 stage groups,
ConvNeXt 5, ClinkCNN-v2 nine. More stages means the "adapt roughly the last
third" recommendation lands on a meaningful boundary rather than on "half the
network".

**Feature.** Also inherited from the parent, and also a stage-one decision. The
Train tab's *adopt feature* action exists for the case where you want to change
it: it copies the parent's contract onto the child so the transfer stays legal.
What matters for stage two:

| knob | why it matters here |
|---|---|
| `fmax` | Must reach ~16 kHz. Below that the upper comb is not in the input at all, and no amount of fine-tuning recovers it. |
| `sample_rate` | Caps `fmax`. 32 kHz keeps most of the band, 44.1 kHz keeps all of it. |
| `n_mels` / `feature_type` | At 128 mel bands each band near 8 kHz spans ~230 Hz, wide enough to merge partials from two different glasses. 192–256 bands, or a linear front-end, buys real resolution. |
| `hop_length` | 5–10 ms resolves the attack and the start of the decay. |
| `window_seconds` | Must contain the decay — decay rate is itself discriminative between a thin crystal glass and a thick tumbler. |

The Audio Features tab flags all of these against this task.

---

## 3. Depth: how much of the encoder to adapt

The Train tab's depth control is stage-based and model-aware. The recommendation
is a *share* of the encoder, not a fixed count, so "gentle transfer" means the
same thing on a 5-stage and a 17-stage model:

| dataset | recommended | rationale |
|---|---|---|
| < 300 clips | ~10% of stages, after head warm-up | barely enough data to move anything safely |
| 300 – 10 000 | ~30% of stages | the default supervised transfer recipe |
| ≥ 10 000 | full | enough signal to re-derive low-level filters |

**For sibling classes, do not go below two stages.** A head-only probe fits a
linear boundary in the parent's embedding, and the parent was never asked to
keep glass types apart — the distinction may not be represented there at all.
The advice panel raises `frozen_encoder_cannot_separate_siblings` for exactly
this case.

### Layer-wise LR decay

`lr_layer_decay` gives earlier stages a lower LR so transferred low-level
filters are protected. It is recommended as an **end-to-end ratio**, not a fixed
per-stage constant: the shallowest trainable stage lands at 10% of the backbone
LR for a full fine-tune and 50% for a partial one, and the per-stage number is
solved backwards from the stage count.

This matters more than it looks. A hardcoded 0.75 costs a 17-stage MobileNet
`0.75¹⁶ ≈ 1%` at the stem — effectively frozen, despite the run being labelled a
full fine-tune — while a 5-stage ConvNeXt keeps 32%. Same knob, two different
recipes. If you set the value by hand, compute it from the ratio you want:
`decay = ratio ** (1 / (trainable_stages - 1))`.

---

## 4. The classifier head

When initializing from a previous run, the head transfers **per class name**:
every class present in both runs starts from the parent's fitted weight row and
bias, and genuinely new classes start random. This is `training.transfer_head_init
= "matching_classes"` (the default).

The parent's rows were fitted against the exact embedding the child starts from,
so a shared class begins near its solution instead of spending the first third
of the budget getting back there. Set it to `"random"` only when you are
deliberately measuring a cold-head baseline. The training log names which classes
were warm-started.

**Head LR.** Keep `lr_head` within ~5× `lr_backbone`. A fresh classifier learning
20× faster fits the training set on the parent's *unchanged* features within a
few epochs; early stopping then fires and the encoder never adapts. The run looks
converged while nothing transferred — a common cause of "the fine-tune barely
improved on the parent".

---

## 5. BatchNorm — the most common silent failure

Two separate settings, often confused:

**`freeze_batchnorm`** decides what happens *during* training. On a small
fine-tune, freezing is right: a few hundred clips from one room are a worse
estimate of the deployment input distribution than the upstream corpus they would
replace, and re-estimating them every epoch is a common reason validation loss
looks noisy and converges slowly. The recommendation now fires from the *share of
the encoder being adapted* (< 75%) plus dataset size (< 5000 rows), rather than
from a depth string — which is why small multi-stage fine-tunes were previously
never advised to freeze it.

**`bn_recalibrate`** decides what gets *shipped*. It streams clean training
batches through the finished weights with only BatchNorm in train mode, then
writes a new checkpoint that calibration, auto-test and export all consume. Turn
it on whenever either is true:

- BatchNorm was frozen — the exported model would otherwise normalize with the
  parent's constants under weights that have moved.
- `mixup_alpha > 0` — BatchNorm saw convex blends of two clips, which never occur
  at inference.

It changes saved statistics only, never learned weights, and it is a no-op on
architectures without BatchNorm.

---

## 6. Regularization

| knob | fine-grained fine-tune |
|---|---|
| SpecAugment frequency mask | Keep under ~10% of the feature rows (≈8 of 128). The default 24/128 blanks 19% of the spectrum in one mask — that is the right trade for general tagging and the wrong one here, because the partial comb *is* the class label. |
| SpecAugment time mask | Light. The attack and early decay carry the evidence. |
| `mixup_alpha` | **Off.** A 50/50 blend of two glasses is not a meaningful training example; it teaches the model that the boundary between them is soft. Mixup earns its place on the large stage-one corpus, not here. |
| `weight_decay` | 1e-3 is a reasonable small-set default; 1e-4 barely constrains anything. |
| `head_dropout` | Left on *auto*, which resolves to 0 when the architecture already drops the pooled embedding at the selected depth and 0.2 otherwise. Raise it to 0.3–0.4 by hand only for a genuinely thin fit (frozen encoder, few hundred clips), where one Linear layer over a wide embedding is the whole fitted model. |
| `class_balance` | `balanced` when background and target counts differ; the sampler is the single place class frequency is corrected — losses stay unweighted. |

---

## 7. Loss and the decision layer

Keep `bce_logits`. It is the objective train, eval, calibration and export all
share, and switching to softmax cross-entropy would break the independent-sigmoid
runtime contract.

But understand what BCE does *not* give you: independent sigmoids apply no
competitive pressure between near-identical classes. Each class only has to
separate itself from everything else, which two variants of one sound do
trivially and identically. The competitive pressure comes from the **decision
layer** instead:

- `inference.margin_to_second_best` requires the winning class to beat the
  runner-up by that much. Checkpoint selection uses the same margin, so it is
  also what makes the *selected* checkpoint the one that commits rather than one
  that is confidently undecided.
- The default 0.08 is sized for detection. For sibling classes raise it to
  ~0.20 and let threshold calibration re-fit around it.

Enable `calibrate_thresholds` on every fine-tune: small runs shift probability
scales, and scoring an uncalibrated decision layer makes runs incomparable.

---

## 8. Schedule

- **Epoch budget** is a ceiling, not a promise — early stopping normally ends the
  run. 35–45 for a partial transfer, 60–80 for scratch.
- **`fine_tune_warmup_epochs`** (head-only epochs before the encoder unfreezes)
  defaults to `min(5, 10% of epochs)`. Leave it at 0/auto. If it ever exceeds
  ~40% of the budget the advice panel flags it — most of the run would be spent
  not adapting the encoder.
- **`warmup_epochs`** is the *LR ramp* and is unrelated. Keep it well under
  `patience`, or the run can be early-stopped before it reaches full learning
  rate.
- **`patience`** ≈ 20–25% of the epoch budget.

---

## 9. Reading a disappointing run

| symptom | first thing to check |
|---|---|
| Detects clinks, cannot name the glass | `fmax`, band count, SpecAugment frequency mask width — is the partial comb in the input and un-masked? |
| Val loss falls very slowly, noisily | BatchNorm re-estimating from a thin sample (`freeze_batchnorm`), or an LR ramp longer than patience |
| Barely better than the parent | head LR ≫ backbone LR, or a depth too shallow to represent the distinction |
| Good val, poor deployed | frozen BatchNorm shipped without `bn_recalibrate`; uncalibrated thresholds |
| Overfits within a few epochs | too few independent recording *groups*, not too few clips — check the split diagnostics before adding regularization |

The last row is worth stressing: the ceiling on generalization is the number of
independent recording sessions per class, not the row count. A corpus with
thousands of clips drawn from a handful of sessions is redundant, and the
regularization recommendations key off that redundancy rather than off size.
