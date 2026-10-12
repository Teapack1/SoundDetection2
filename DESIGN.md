# Design

SoundDetection2 is a venue-specific sound event detection system for interactive
lighting. It records real target sounds and negative/background sounds, trains a
per-venue model, calibrates thresholds for low false positives, and runs CPU
inference on an edge PC that can trigger ArtNet/DMX.

| Document | Purpose |
| --- | --- |
| [README.md](README.md) | Overview and documentation entrypoint. |
| [SETUP.md](SETUP.md) | Machine setup, audio devices, edge PC, and ArtNet networking. |
| [WORKFLOW.md](WORKFLOW.md) | Operational reference for project creation through deployment. |
| [DESIGN.md](DESIGN.md) | Design rationale plus recording and training guidance. |
| [docs/pretraining_guide.md](docs/pretraining_guide.md) | Stage one: class space, frontend, normalization and architecture for the shared-corpus encoder. |
| [docs/finetuning_guide.md](docs/finetuning_guide.md) | Stage two: adapting that encoder to one venue's glassware. |

## System Design

### Tech Stack

| Area | Project choice | Why |
| --- | --- | --- |
| Language/runtime | [Python](https://www.python.org/) 3.11-3.12, managed by [uv](https://docs.astral.sh/uv/) | Fast iteration, simple deployment, reproducible environments. |
| CLI and operator UI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) | Typed commands, readable tables, live terminal diagnostics. |
| Config/schema | [Pydantic](https://docs.pydantic.dev/) + TOML | Validated project configs with clear defaults. |
| Audio capture | [python-sounddevice](https://python-sounddevice.readthedocs.io/) / PortAudio | Cross-platform low-latency microphone input. |
| Audio files/features | [SoundFile](https://python-soundfile.readthedocs.io/), [librosa](https://librosa.org/doc/latest/index.html), NumPy/SciPy | Reliable WAV I/O, resampling, and DSP utilities. |
| Training | [PyTorch](https://pytorch.org/), [Lightning](https://lightning.ai/docs/pytorch/stable/), TorchMetrics | Standard fine-tuning loop with GPU support and metrics. |
| Pretrained audio model | [EfficientAT](https://github.com/fschmid56/EfficientAT) submodule | Efficient AudioSet-pretrained CNN embeddings for small venue datasets. |
| Augmentation | [audiomentations](https://iver56.github.io/audiomentations/) plus project overlay recipes | Mix real targets with realistic venue backgrounds. |
| Export/inference | [ONNX](https://onnx.ai/), [OpenVINO](https://docs.openvino.ai/), [NNCF](https://github.com/openvinotoolkit/nncf) | Portable export, CPU inference, optional quantization path. |
| Control output | [PyZMQ](https://pyzmq.readthedocs.io/) + [stupidArtnet](https://pypi.org/project/stupidArtnet/) | Decoupled detection events plus ArtNet/DMX triggering. |

### Design Priorities

| Priority | Project choice |
| --- | --- |
| Low false positives | Conservative classes, explicit `_negative`, suppressor classes for near-misses, runtime gates, calibration by false fires per 1000 negative clips. |
| Venue fit | One project under `projects/<id>` per room/use case. |
| Reusable data | Shared clip library; project-specific dataset selections. |
| Edge reliability | OpenVINO CPU runtime; no PyTorch required in production. |
| Traceability | WAV sidecars, config snapshots, dataset snapshots, model bundles. |

### Shared Library, Project Datasets

Recordings live in `data_library/sounds/<label>/` as WAV files with JSON
sidecars. The sidecars preserve provenance: label, project, density,
microphone/channel data, tags, review state, and recording details.

Projects compose datasets from that shared library through
`dataset_selection.json` and generated `train.csv`, `val.csv`, and `test.csv`
files. This keeps clips reusable while allowing each venue to define its own
model classes and training set.

### Workstation And Edge

The project separates model-building work from production inference. The
workstation owns the mutable work: recording, inspection, augmentation,
training, evaluation, calibration, and export. It installs the heavy training
stack such as PyTorch, Lightning, EfficientAT, and export tools.

The edge PC runs the selected bundle in place. It uses OpenVINO for CPU
inference and does not need PyTorch, which keeps production deployment smaller
and easier to maintain.

### Model And Features

| Part | Default | Why |
| --- | --- | --- |
| Backbone | EfficientAT `mn10_as`: MobileNetV3 width 1.0, AudioSet-pretrained, about 4.88M params / 0.54B MACs | Strong small-audio features from limited venue data. |
| Head | Project-specific classifier | Each venue owns its own class list. |
| Window | 0.4 s | The target transients are ~0.2 s; a longer window dilutes them in background. |
| Model rate | 32 kHz mono | Matches the pretrained EfficientAT contract. |
| Frontend | `efficientat` log-mel | Keeps training, export, and runtime feature shapes aligned. |

| Feature value | Default | Reason |
| --- | ---: | --- |
| `n_fft` | 1024 | Enough frequency detail for glass, speech, crowd, and music transients. |
| `win_length` | 800 samples | 25 ms at 32 kHz; standard short-time audio analysis window. |
| `hop_length` | 320 samples | 10 ms frame spacing; matches EfficientAT-style preprocessing. |
| `n_mels` | 128 | Good resolution without making the model input too large. |
| `fmin` / `fmax` | 0 / 15000 Hz | Covers most useful venue cues at 32 kHz sample rate. |

The feature contract is intentionally strict for pretrained EfficientAT models.
Changing sample rate, mel count, frequency range, or window geometry changes
the input distribution the pretrained backbone expects. When those values must
change, the safer design is to train without the pretrained contract.

#### Compression and normalization are separate

Beyond the geometry above, a feature has two independent value stages, one
control each in the Audio Features tab:

| Stage | Decides | Options |
| --- | --- | --- |
| `features.compression` | what the numbers **mean** | `log` (ln of power) · `db` (10·log10) · `none` (raw power) · `pcen` (per-band adaptive gain) |
| `features.scaling_method` | what **range** they arrive in | `none` · `zscore` · `minmax`, at `scaling_scope` `global`/`per_band` |

Compression is feature generation; normalization only shifts and scales. They
are orthogonal, and both apply identically to mel and linear bases — the earlier
design exposed them as one per-feature-type list of modes, which made
"log-mel" and "legacy dB" look like alternatives to "z-score" when they are
answers to different questions.

`features.scaling_source` records where the normalization constants came from:
`dataset` (fitted on the composed training split, validation excluded, frozen
into the bundle) or `fixed` (pinned by a checkpoint contract). Two frontends
that historically looked like "no normalization" are fixed specs and are stored
as such — EfficientAT's `(x + 4.5) / 5` is a fixed global z-score with mean
−4.5 / std 5, and the legacy soundDetection dB image is a fixed global min–max
over [−20, 60] dB. The older `log_power` / `db_norm` / `pcen` / `standardize`
fields remain accepted and are kept mirrored by config validation, so existing
projects, presets and frozen bundles produce byte-identical values.

### Recording

| Mechanism | Typical value | Purpose |
| --- | --- | --- |
| Clip length | 0.4 s | Matches the model window. Raise `record_seconds` when a library must also serve projects with a longer window — the window is cropped around the onset, so headroom is only useful if some project wants more of it. |
| Preroll | 120 ms or random 60-240 ms | Captures lead-in and adds timing variety. |
| Onset detector | Superflux / spectral flux | Finds transient events better than loudness alone. |
| Onset floor | `min_flux_db`, measured per room | The shipped -80 dB is a placeholder: a quiet office already sits near it, so it triggers on almost anything. Set it with the console's Auto-calibrate, which listens to the room and places the floor above what it does on its own — walk around while it listens and it learns to ignore you. |
| Cooldown | 600 ms | Avoids near-duplicate clips from one event. |
| Multi-mic merge | ~30-500 ms | Collapses the same physical event across mics. |
| Trigger mode | `any` for multi-mic venues | Lets each mic detect independently. |

The saved clip is longer than the model window on purpose. `record_seconds`
keeps generous context so the same library can train projects at different
window lengths, and the model window is cropped out of it around the detected
onset — never around the clip centre, which would discard the transient in most
recordings once the window is shorter than the clip. The onset sits
`min(preroll_ms, 30% of the window)` into the window, and the live onset-gated
runtime scores exactly that geometry, so training and production frame the same
event the same way.

The recording console (`sed record --serve`) is the operator surface for all of
this: it shows every microphone's level and the live `flux vs gate` comparison,
and lets the floor, cooldown, clip length and preroll be changed while the
stream runs. It drives the same loop as the terminal recorder, so the two
produce identical clips.

The onset detector runs at the model rate, so capture at any other rate is
converted on the way in. That conversion is **streamed**, not done per audio
block: resampling each block on its own zero-pads both of its edges, and a
spectral-flux detector reads the resulting step as a transient at every block
boundary. Measured on 30 s of a held 440 Hz tone with no transients in it, the
default gate fired 124 times at 48 kHz and none at 32 kHz, where no conversion
happens at all — the gate's sensitivity depended on the sound card and on the
buffer size. `sed.audio.resample.StreamingResampler` carries the filter state
across blocks so the result is the same however the audio is chunked, and the
recorder, the live runtime and `sed evaluate onset-gate` all now measure the
same gate.

Onset triggering records only acoustically interesting moments. Superflux uses
spectral change, so a quiet glass transient can trigger even when its absolute
loudness is low. The adaptive MAD baseline follows slow room changes; the
`min_flux_db` floor prevents tiny continuous background motion from becoming
recordings.

`any` is the preferred multi-mic trigger mode because a transient may be clear
on one microphone and masked on another. `mix` is cheaper but can dilute
single-mic events. `ch0` is useful only when one channel is the trusted source.

### Runtime

The live path is:

```text
PortAudio capture -> ring buffer -> onset-centred windows -> resample/features
-> OpenVINO CPU model -> probabilities -> multi-mic aggregation
-> smoother/arbiter -> ZMQ/ArtNet
```

The runtime reads one model window (0.4 s by default), resamples it to the model rate,
extracts log-mel features, and runs the exported OpenVINO model. Multi-mic
inference can combine channels with `max`, `best`, `mean`, or `single` behavior.

Before a trigger is emitted, probabilities pass through median smoothing, EMA,
hysteresis, consecutive-frame checks, per-class thresholds, `_negative` gating,
margin-to-second-best gating, suppressor gating, and a global refractory period.
These gates are intentional: they trade some recall for lower false-positive
risk.

The argmax that picks the winner spans **every** non-`_negative` class, including
the ones that cannot fire. A suppressor has to be allowed to win, or the decision
falls to the strongest target while the model was saying "cutlery". This is also
what makes the live decision identical to the one checkpoint selection and
evaluation measure, which have always compared every non-`_negative` class.

| Runtime gate | Default | Why |
| --- | ---: | --- |
| Inference hop | 300 ms | Frequent enough for lighting, low CPU cost. |
| Smoothing median | 3 frames | Removes isolated probability spikes. |
| EMA alpha | 0.4 | Stabilizes confidence without hiding real events for too long. |
| Consecutive frames | 3 | Requires persistence before firing. |
| Default threshold | 0.90 | Conservative until per-class calibration exists. |
| `_negative` max | 0.50 | Blocks positives when the no-trigger class is strong. |
| Class margin | 0.08 | Avoids firing when two positive classes are too close. |
| Refractory | 300 ms | Prevents double triggers on one event tail. |

With multiple inference microphones, `max` takes the per-class maximum so any
good mic can win, `best` forwards the whole probability vector of the mic that
heard the target loudest, `mean` requires broader agreement, and `single` is for
one trusted channel. With an explicit `_negative` class, `max` lets the quietest
off-axis mic's negative score veto a fire another mic is confident about, so
`best` is the better default for rooms with separated pickup zones; `max` remains
the safe choice for models without a `_negative` output.

### Bundles And Calibration

Each training run creates a self-contained bundle under
`projects/<id>/models/<model_id>/` with config snapshots, class order, dataset
artifacts, checkpoints, exports, reports, and runtime settings.

Calibration sweeps class thresholds against venue or library negative material
and writes the selected thresholds into runtime settings. The goal is practical:
meet a target false-fire rate per 1000 negative clips before lighting output is
enabled. Clips are the denominator because negatives are recorded as
onset-triggered clips, not as continuous audio.

### Avoided Choices

| Avoided choice | Reason |
| --- | --- |
| Automatic gain control | Moves the train/live level distribution and can hide useful loudness cues. |
| Runtime adaptive thresholds | Can drift during sudden crowd or music changes; calibration is more stable. |
| Noise suppression | Often damages the exact transients the model should detect. |
| Pure loudness trigger | Fails on quiet but spectrally sharp events and noisy rooms. |
| Transformer backbone on edge | Higher compute/memory cost than the EfficientAT CNN path. |
| Positive-heavy training set | Real venues are mostly no-trigger audio; overbalanced positives raise FP risk. |

## Venue Recording Strategy

### Class Semantics

Every model output has one of three **roles** at decision time. The role comes
from `[targets]`, not from the name, and `sed check deploy` prints the resolved
role of every class before a bundle ships.

| Role | Declared by | What it does |
| --- | --- | --- |
| Target | `[targets] <class> = { enabled = true }` (the default for an unrecognized name) | Competes for the window and may fire a cue. |
| Suppressor | `enabled = false`, or a name like `hardnegs` with no entry | Competes for the window but never fires. When it wins, the system abstains. Optional `veto_max` also discards the window outright above that score. |
| Negative | `_negative` by name, or any class given `[targets] <class> = { role = "negative" }` explicitly | Never competes for the window; only vetoes it, at `inference.negative_max` unless it sets its own `veto_max`. |

`_negative` stays the reserved name that gets the negative role automatically,
so an existing project needs no config changes. Any *other* class can be
promoted to the same role with an explicit `role = "negative"` — that is what
lets a project split one large `_negative` pool into named, individually
trainable buckets (`music`, `crowd`, a second hard-negative bucket) that all
still pool into the shared veto gate, the loss's negative focusing, and the
deployment false-alarm rate, while getting their own precision/recall for
diagnosis. Only do this when a bucket is large enough to matter (see Negative
Examples below) and you have a concrete reason to track it separately —
splitting for its own sake just dilutes each bucket.

Keep the class list small and tied to lighting behavior. In practice that means
one distinction inside the negatives and no more:

| Class type | Good shape | Avoid |
| --- | --- | --- |
| Negative | `_negative` = room tone, music, speech, crowd, mixed clips — uncategorised | Treating background as an afterthought; over-sorting it into many classes. |
| Suppressor | one `hardnegs` class for near-misses: cutlery, metal, glass handling | Folding near-misses into `_negative`, where one broad row must model them alongside room tone. |
| Object/event | `glass_clink`, `table_hit`, `applause`; per-vessel classes in a venue run | Splitting labels that trigger the same cue. |
| Intensity | recording-density metadata, not a class | Loudness levels as output rows — intensity is a nuisance variable to be invariant to. |
| Venue rule | One meaning an operator can explain | Labels based only on ML convenience. |

The suppressor role exists because a single `_negative` row has to model room
tone, music, speech *and* cutlery at once. It handles the first three easily and
the last one badly — and cutlery, sharing the target's attack and bright
partials, is what actually causes false fires. Because a suppressor competes in
the same argmax and margin comparison that checkpoint selection and evaluation
use, it shapes both what gets selected and what deploys.

See [docs/pretraining_guide.md](docs/pretraining_guide.md) for how the two
stages' class spaces relate.

### Microphone Geometry

| Venue shape | Mic plan | Data emphasis |
| --- | --- | --- |
| Quiet installation | 1 close mic, low threshold | HVAC, footsteps, speech as negatives. |
| Bar / club | 2-4 mics near action zones | Music, crowd, cutlery, bar work, peak density. |
| Stage / crowd | 1-2 audience-facing mics | Applause/cheer range, PA bleed, host speech. |
| Wide room | Multiple fixed zones | Per-zone metadata; split mono clips for training. |

`record_channels` is the physical capture set. `inference_channels` is the
production evaluation set. Keep channel order stable, and tag mic type/position
whenever channels differ.

### Recording Pass Shape

| Item | Practical target |
| --- | --- |
| Positive pass | 30-60 accepted clips per class/position/density. |
| Pass length | 5-15 min for positives; 10-30 min for background negatives. |
| Positions | 3-5 useful distances/angles per venue zone. |
| Levels | Loudest targets below clipping; typical peaks around -18 to -6 dBFS. |
| Review loss | Expect to reject 10-30% for handling noise, speech, bad labels, or duplicates. |
| Metadata | Always keep density, mic/channel, position, object/state, and session context. |

### Positive Examples

| Target | Minimum pilot | Reliable venue | Notes |
| --- | ---: | ---: | --- |
| Positive clips per class | 80-150 | 200-400 | Record one class at a time. |
| Distinct sessions | 2-3 | 4-8 | Different days, positions, and densities. |
| Density coverage | empty/quiet/normal | empty through peak | Avoid a model that only works in rehearsal. |
| Hard variants | 20-40/class | 50-100/class | Weak, distant, masked, or unusual examples. |
| Frozen test positives | 20-40/class | 50-100/class | Never train on these. |

Short onset-centered clips teach transient plus decay. Randomized preroll keeps
the event from always landing at the same frame.

### Negative Examples

| Negative source | Minimum pilot | Reliable venue | Why |
| --- | ---: | ---: | --- |
| Background clips | 300-600 | 800-2000 | Main defense against false positives. |
| Hard negatives | >= positives | 1-3x positives | Similar to target but should not fire. |
| Frozen test negatives | 80-150 | 200-500 | Honest FP check. |
| Calibration clips | 300-600 | 1000-3000 | Recorded background clips; the FP budget is per 1000 of them. |

Good negatives match production density and mic placement: room tone, music,
speech, crowd beds, bar work, cutlery, glass handling, setup noise, and peak
distractions.

### Test Discipline

| Split | Practical target |
| --- | --- |
| Train | Most accepted clips, balanced by class where possible. |
| Validation | 10-20% for model selection and early stopping. |
| Frozen test | Separate representative clips; never used for training decisions. |
| Failure response | Add similar new training material; keep test coverage honest. |

### Augmentation And Composition

| Technique | Use | Limit |
| --- | --- | --- |
| Background overlay | Best default; teaches targets under masking | Use real venue backgrounds. |
| Pitch/time variation | Small robustness gain | Do not create impossible sounds. |
| Channel splitting | Useful for multi-mic mono training | Keep parent/channel provenance. |
| Generated amount | 2-5x raw positives | More is not better if negatives are weak. |

Composition is a design step: the shared library is broad, but each project
selects only the clips that represent its venue.

### Model Selection And Calibration

| Training choice | Good default |
| --- | --- |
| Backbone | Pretrained `mn10_as` unless there is a measured reason to change. |
| Objective | `cross_entropy` with explicit `_negative` for single active class decisions. |
| Epochs | 60 default; rely on validation behavior, not just final epoch. |
| Precision | `16-mixed` on capable training hardware. |
| Class balance | Keep negatives deep; avoid a positive-heavy training set for real venues. |

| Decision | Prefer |
| --- | --- |
| Best model | Low false positives on hard negatives before high aggregate accuracy. |
| Positive recall | Good on clear events; acceptable misses on weak/ambiguous events. |
| Thresholds | Per-class calibration from venue-like negatives. |
| FP target | 0.1-0.5 per 1000 negative clips for disruptive cues; 0.5-1 normal; 1-3 for subtle ambient cues. |
| Recalibrate when | Mic placement, acoustics, classes, dataset, or model bundle changes. |

### Production Feedback

| Production signal | Dataset response |
| --- | --- |
| False trigger | Promote or record as hard negative. |
| Missed valid event | Add similar positives from the same acoustic condition. |
| Confused intensity | Merge labels or collect cleaner boundary examples. |
| New season/layout/show | Treat as a new venue condition and recalibrate. |
