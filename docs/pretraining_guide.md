# Pretraining guide — designing stage one so stage two can work

*The first of the two-stage workflow: shaping a general glass-clink encoder on
the shared corpus, in a way that the small venue fine-tune can actually build
on. Its companion is [finetuning_guide.md](finetuning_guide.md).*

Everything here is advisory. No class name is required, no sample rate is
enforced, and every architecture in the catalog stays trainable. The
**Recommendations** panel at the top of the Train tab (`sed.train.advice`)
reports the same findings live against your current configuration.

---

## 1. What each stage is for

| | stage 1 — domain pretrain | stage 2 — venue fine-tune |
|---|---|---|
| question | *is this a glass clink, or the room?* | *which glass?* |
| data | the shared corpus (tens of thousands of clips, many sessions) | a few hundred to a few thousand clips from one venue |
| initialization | random (or upstream) | **from previous run** — the stage-1 encoder |
| classes | coarse: the umbrella event and the sounds it gets confused with | the actual vessels |
| frontend | designed and fitted here | **inherited unchanged** |

The Train tab names which one you are in, above the recommendations:
`stage 1 · domain pretrain` or `stage 2 · venue fine-tune ← project/run`.

The decisive fact is that **stage two inherits far more than the weights.** The
architecture, the feature geometry and the fitted scaling constants are all part
of the encoder's numeric contract and cannot change between the stages. Stage one
is therefore where those decisions are actually made, whether or not you were
thinking about stage two at the time.

---

## 2. Classes: three roles, not a taxonomy

You do not need to sort your negatives into categories. There is exactly one
distinction worth making:

| role | what goes in it | why |
|---|---|---|
| **targets** | the sounds that should fire a cue. Stage 1: one umbrella class such as `cheers`, plus any genuinely different events you also act on (`clap`, `attention`). Stage 2: one class per vessel. | these are the classes with `enabled = true` in `[targets]` |
| **`hardnegs`** | near-misses only — cutlery, metal on metal, metal on wood, glass handling and placement | one `_negative` row has to model room tone, music, speech *and* cutlery at once. It does the first three easily and the last one badly, and cutlery is what actually fires in a venue |
| **`_negative`** | everything else: room tone, ambience, music, speech, TV, silence, crowd, and mixed clips | you do not have to categorise these — that is the point |

So: yes, put ambience, music, speech, TV, room tone and mixed clips all into
`_negative` and stop thinking about it. Keep exactly one extra bucket for sounds
a person would describe as "that almost sounded like a clink". When a clip
contains both, it belongs in `hardnegs` only if the confusable is the dominant
event; otherwise `_negative`.

Five classes over a large corpus is a reasonable stage-1 shape:
`cheers`, `clap`, `attention`, `hardnegs`, `_negative`.

### What a suppressor class actually does

`hardnegs` is a **suppressor**: a real model output that competes for every
window but never fires a cue. When it wins, the system abstains.

That behaviour is what makes the class worth having. Without it, a window where
the model says "cutlery, 0.95" and "wine glass, 0.80" would still fire the wine
glass, because the class that knew better was not allowed to speak. The same
comparison is used by checkpoint selection and by evaluation, so a suppressor
that works in production also shapes which checkpoint gets selected.

**Nothing here is enforced by name.** The role comes from the existing
`[targets]` block:

```toml
[targets]
riedel_wine = { enabled = true,  threshold = 0.9, channel = 1, value = 85 }
hardnegs    = { enabled = false }                    # competes, never fires
# hardnegs  = { enabled = false, veto_max = 0.5 }    # ...and hard-vetoes above 0.5
```

A class named `hardnegs` / `hard_negative` with no entry defaults to the
suppressor role; any class can be given either role explicitly. `sed check
deploy` prints the resolved role of every class before a bundle ships.

### Keep stage one coarse on purpose

Do **not** put vessel names in stage one. The corpus is not labelled by vessel,
and more importantly the single `cheers` row is what seeds every stage-2 vessel
row (§5). A coarse parent produces a general "this is a clink" direction in the
embedding, which is exactly the right starting point for three classes that all
have to sit near it.

Do not use loudness as a class either. Intensity is a nuisance variable to be
invariant to, and it is already captured as recording-density metadata.

---

## 3. Sample rate and the audio feature

A glass clink is a set of damped modal resonances: a fundamental around
400 Hz–3 kHz and a partial comb running to roughly 20 kHz. **The upper partials
carry the vessel identity**, and they sit exactly where venue noise is weakest —
speech, music and HVAC are all strongly low-frequency dominated.

| knob | suggested | why |
|---|---|---|
| `sample_rate` | 44 100, or 48 000 if the interface runs there natively | 32 kHz caps Nyquist at 16 kHz, and the AudioSet contract caps `fmax` at 15 kHz — the discriminative band is then simply absent from the input |
| `fmax` | ~20 000 | the top of the useful comb |
| `fmin` | 150–300 | below that there is almost no clink energy and plenty of HVAC, footfall and kick drum |
| `n_mels` | 192–256 for the identification task | see below |
| `hop_length` | ~5 ms | resolves the attack and the start of the decay |
| `win_length` | ~20 ms | shorter than the usual 25 ms; sharpens the onset |
| `window_seconds` | 0.4 | must contain the decay — decay rate itself separates a thin crystal glass from a thick tumbler |

Every one of these stays a free field. Avoid resampling where you can: prefer the
rate your interface actually runs at.

**On 128 mel bands.** At 44.1 kHz with `fmax` 20 000, each of 128 mel bands spans
roughly **232 Hz at 8 kHz**, while the partials that separate two vessels sit tens
to a couple of hundred Hz apart — so they merge. 128 bands detect a clink
perfectly well and identify the vessel marginally. 192–256 bands halve the smear
to about 115 Hz; a linear front-end resolves it fully at 3–7× the input rows. The
Audio Features tab computes this number for your exact geometry.

---

## 4. Normalization — the part that must not change later

### Two axes, not one list of modes

A feature is a **basis**, a **compression** and a **normalization**, and the
Audio Features tab has one control per axis:

| axis | what it decides | options |
| --- | --- | --- |
| basis | which frequency rows exist | mel bands, or linear STFT bins |
| **compression** | what the numbers *mean* | `log` = ln(power + 1e-5) · `db` = 10·log10(power) · `none` = raw power · `pcen` = per-band adaptive gain |
| **normalization** | what *range* those numbers arrive in | `none` · `z-score` (x − mean)/std · `min–max` clip((x − lo)/(hi − lo)) |

Compression is feature generation: log, dB and PCEN are different ways of
squeezing audio's enormous dynamic range into something a convolution can read,
and each one changes what a value means. Normalization changes no meaning at
all — it shifts and scales, so the network starts from numbers of a sensible
size. They are independent: every compression accepts every normalization, and
the same four compressions apply to mel and to linear.

Normalization constants come from one of two places, and the difference matters
more than the arithmetic:

- **fixed** — pinned by a checkpoint contract or typed by hand. Nothing was
  measured on your data.
- **fitted** (`dataset`) — measured on your composed training split.

Two frontends that used to look like "no normalization" are in fact fixed specs,
and the tab now says so:

| historical name | compression | normalization |
| --- | --- | --- |
| EfficientAT "fast-norm" `(x + 4.5) / 5` | `log` | fixed z-score, mean −4.5, std 5 |
| legacy soundDetection dB image | `db` | fixed min–max over [−20, 60] dB |
| CED checkpoint frontend | `db` | none |

AudioSet's constant was measured at 32 kHz / 128 mel / 15 kHz fmax. On any other
geometry it is an arbitrary offset and gain — which is exactly why seeing it
written down as "fixed" rather than as a blank is worth the extra control.

### What to choose

The standard practice, and the recommendation here, is to fit **per-band
z-score** constants on the composed training split:

- fitted on the *training* side of the live grouped split only — validation clips
  never contribute;
- balanced across classes and recording groups, so one large session cannot
  dominate;
- fitted in the compression-only domain — whichever compression is selected, so
  the constants land on exactly the values the normalization stage will consume;
- **frozen into the model bundle** and replayed unchanged by evaluation,
  threshold calibration, export and live inference.

Per band rather than global matters at this resolution: a mel spectrogram's band
energies differ by orders of magnitude between the low and high end, so a single
scale leaves the quiet upper partials — the ones carrying vessel identity — near
the bottom of the input range.

Fit them with **Estimate from dataset** in the Audio Features tab. The state card
next to it reports method, scope, band count, how many clips the fit consumed and
which dataset hash it came from.

### Across the two stages

Stage two **inherits these constants unchanged**. That is deliberate: refitting
them on venue data would move the input distribution the transferred weights
expect, and the encoder would be reading a different feature than the one it
learned. The training launch gate enforces this, and the Audio Features tab
disables *Estimate from dataset* for a parent-initialized run.

Which makes the opposite question the useful one in stage two: **are the venue
clips still inside the parent's scale?** *Check fit on this dataset* answers it,
without changing anything — it measures your clips in the same compression-only
domain and reports how far each band's mean sits from the fitted one, in fitted
standard deviations. A large systematic offset means a gain-staging, microphone
or distance difference, and it shows up here long before it shows up as a
confusing validation curve.

---

## 5. Architecture

Pick it in stage one; stage two cannot change it.

**Pretrained or random?** This project has measured it: ImageNet-upstream
`resnet18` reached 0.435 on the frozen-threshold validation metric where the same
backbone from scratch reached 0.983. A pretrained contract buys a prior worth
less than the frontend resolution it costs on this task. Train stage one from
random initialization on your own corpus.

**`mn10_as` without weights is not special.** The `_as` suffix names the AudioSet
*weight set*, not the architecture. With random initialization it is the
EfficientAT MobileNetV3-style spectrogram trunk at width 1.0, with random weights
and no AudioSet lineage at all. What you keep is the architecture and its **17
stage groups**, which is what makes "adapt the last ~30 %" land on a meaningful
boundary in stage two.

| candidate | params | stages | notes |
|---|---:|---:|---|
| `clink-cnn-v2` | 242 k | 9 | purpose-built for 0.4 s glass impacts — explicit resonance-position and attack/decay planes, anisotropic mixing, frequency-pyramid pooling. Reached 0.99 on this corpus. Deploys trivially on the edge CPU. Native geometry is 44.1 kHz over 0.4 s; other rates build and train and are reported as an advisory note, not refused. |
| `mn10` / `mn10_as` random | 4.9 M | 17 | choose for the finer fine-tune granularity, or as headroom if the corpus grows well past its current size with more independent sessions. Scratch scores 0.977–0.991. |
| `resnet18`, `convnext_nano` | 11–15 M | 5 | work from scratch, bring nothing task-specific, and their upstream weights are actively harmful here. |

**Capacity is not the bottleneck.** A 242 k-parameter model matches an
11 M-parameter one on this corpus. Telling vessels apart is a *frontend
resolution* problem, not a capacity problem — a small model separates glasses
reliably provided the partial comb is in its input. Above ~10 M the binding
limit is the venue set's small number of independent recording sessions, not its
clip count.

---

## 6. Schedule

The **Domain pretrain from scratch — large corpus** preset fills these in from
your dataset; the values below are what it is aiming at.

| knob | stage 1 | why |
|---|---|---|
| depth | full | random weights have nothing to protect |
| `lr_backbone` / `lr_head` | equal, ~3e-4 | the head is not special when everything is random |
| `lr_layer_decay` | 1.0 | no pretrained low-level features to preserve |
| `warmup_epochs` | ~8 | a long LR ramp stabilizes early scratch training |
| `epochs` | 60–80 | a ceiling, not a target — early stopping normally ends the run |
| `patience` | ~20 % of the budget | |
| `freeze_batchnorm` | off | BatchNorm must learn its statistics here |
| `bn_recalibrate` | on | this run becomes the parent for every fine-tune; ship it with statistics measured on un-mixed clips |
| SpecAugment | on, frequency mask under ~10 % of the rows | masking is the cheapest invariance on a large corpus, but the narrow partial comb *is* the class label |
| `mixup_alpha` | ~0.2–0.3 | earns its place here, on a large corpus with near-duplicate clips — and only here |
| `class_balance` | `sqrt` or `balanced` | the sampler is the single place class frequency is corrected; losses stay unweighted |
| `loss` | `bce_logits` | the objective training, evaluation, calibration and export all share |

The ceiling on generalization is the number of **independent recording sessions
per class**, not the row count. A corpus with tens of thousands of clips drawn
from a handful of sessions is redundant, and the recommendations key off that
redundancy rather than off size.

---

## 7. Handing off to stage two

Before you call a pretraining run finished, check that it is a usable parent:

1. **The frontend is fitted, not defaulted.** The scaling state card should read
   *fitted*, with a band count matching the feature and a clip count that looks
   like your corpus.
2. **BatchNorm was refreshed** (`bn_recalibrate`), so the exported statistics
   describe real clips.
3. **The class space is coarse and named for what the sound is**, because those
   names are what stage two maps onto — a vessel class can start from a `cheers`
   row, but only if such a row exists.
4. **The run completed and has a checkpoint**, or it will not appear in the
   parent picker at all.

Then continue in [finetuning_guide.md](finetuning_guide.md).
