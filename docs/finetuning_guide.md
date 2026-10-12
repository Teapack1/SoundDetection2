# Fine-tuning guide — domain pretraining → fine-grained venue runs

*How to configure the second stage of the two-stage workflow: a small,
domain-specific run initialized from a previous run's encoder, whose job is not
"detect a clink" but "tell one glass from another".*

Everything here is reachable from the Train tab. The **Recommendations** panel at
the top of that tab reports the same findings live for your current config
(`sed.train.advice`), so treat this document as the reasoning and the panel as
the checklist. Stage one — designing the corpus, the class space, the frontend
and the architecture this stage inherits — is
[pretraining_guide.md](pretraining_guide.md).

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

**Normalization is inherited too — and must be.** The parent's fitted scaling
constants are part of the encoder's numeric contract: refitting them on venue
data would move the input distribution the transferred weights expect. The
training launch gate enforces this, and the Audio Features tab disables
*Estimate from dataset* for a parent-initialized run and labels the state card
*inherited from parent*.

So the useful question here is the opposite one: **do the venue clips still land
inside the parent's scale?** *Check fit on this dataset* measures exactly that,
without changing anything — how far each band's mean sits from the fitted one, in
fitted standard deviations. A large systematic offset points at gain staging, a
different microphone, or a different distance, and the advice panel raises
`scaling_drifted_from_this_dataset`. The fix is on the audio side, never by
refitting.

---

## 2b. The venue class space

Same three roles as stage one (see
[pretraining_guide.md](pretraining_guide.md#2-classes-three-roles-not-a-taxonomy)):

- **targets** — one class per vessel, plus any other event the venue acts on;
- **`hardnegs`** — the near-misses this venue actually produces: cutlery, metal,
  glass handling and placement. Left out of `[targets]` (or `enabled = false`) it
  competes for every window without ever firing, so winning suppresses the
  trigger. Carry it over from stage one so its row warm-starts by name;
- **`_negative`** — everything else, uncategorised.

Recording targets, clips only:

| | per class |
|---|---|
| vessel classes | 200–400 clips across 4–8 sessions and 3–5 positions |
| `hardnegs` | several hundred, at venue density |
| `_negative` | 800–2000 |
| frozen test | 20–100 per class, never trained on |

There is no continuous-recording step: threshold calibration runs against
recorded negative *clips*, and the budget is expressed per 1000 clips.

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

## 4. The classifier head — the most common cause of a collapsed transfer

The encoder always transfers whole. What each **output row** of the classifier
starts from is a separate decision, and it is the one that decides whether the
run converges inside its epoch budget.

A row that starts from a fitted parent row begins near its solution — the
parent's rows were fitted against the exact embedding this run starts from. A row
that starts random has to be re-derived from a few hundred venue clips at a
*fine-tune* learning rate, which is a budget sized for a warm head. In this
project's own run history that is the difference between a `last_two` transfer
scoring 1.0 and the same configuration scoring 0.09 — and doubling the epochs
does not fix it, because the schedule was never the problem.

**Exact name matches are automatic.** The trap is that the two stages usually
share no positive class names at all: stage one labels the umbrella event
(`cheers`) and stage two labels the vessels (`riedel_wine`, `ikea_wine`,
`riedel_veloce`). The intersection is `{_negative}`, so every target row starts
random unless you say otherwise.

**Say otherwise in the Model tab.** The parent picker shows a *Classifier rows*
table — one row per class in this run, with the parent class it starts from:

| this run | starts from | |
|---|---|---|
| `_negative` | `_negative` | reuses the parent's own row |
| `riedel_wine` | `cheers` ▾ | starts from the parent's row |
| `riedel_veloce` | `cheers` ▾ | starts from the parent's row |
| `ikea_wine` | `cheers` ▾ | starts from the parent's row |
| `hardnegs` | `hardnegs` | reuses the parent's own row |

Rows that would start random are highlighted, and the panel raises
`finetune_positive_rows_cold` while any remain. Names with an obvious
relationship (`cheers` → `cheers_riedel_wine`, or the same name written with a
different separator) are pre-filled as a suggestion; unrelated names are left for
you rather than guessed at. The choices are saved as
`training.transfer_head_map` and recorded per class in the run manifest, and the
training log names every row's origin.

Several classes may start from the same parent row — that is the normal case
here. Each copy gets a small deterministic jitter (2 % of the row's own spread,
seeded from the training seed) so they do not begin bit-identical.

`training.transfer_head_init = "random"` ignores the table entirely and starts
every row cold. Use it only to measure a cold-head baseline deliberately.

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
| **Collapses to near-zero recall, and more epochs change nothing** | the **Classifier rows** table — how many target rows start random? A cold row at a fine-tune LR cannot be re-derived inside this budget. This is the single most common cause. |
| Detects clinks, cannot name the glass | `fmax`, band count, SpecAugment frequency mask width — is the partial comb in the input and un-masked? |
| Val loss falls very slowly, noisily | BatchNorm re-estimating from a thin sample (`freeze_batchnorm`), or an LR ramp longer than patience |
| Barely better than the parent | head LR ≫ backbone LR, or a depth too shallow to represent the distinction |
| Good val, poor deployed | frozen BatchNorm shipped without `bn_recalibrate`; uncalibrated thresholds |
| Fires on cutlery in the venue | is there a suppressor class at all, and does `sed check deploy` report its role as `suppressor` rather than `target`? |
| Everything looks shifted / oddly flat | *Check fit on this dataset* — the venue may sit outside the inherited scaling |
| Overfits within a few epochs | too few independent recording *groups*, not too few clips — check the split diagnostics before adding regularization |

The last row is worth stressing: the ceiling on generalization is the number of
independent recording sessions per class, not the row count. A corpus with
thousands of clips drawn from a handful of sessions is redundant, and the
regularization recommendations key off that redundancy rather than off size.
