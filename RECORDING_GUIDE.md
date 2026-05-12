# Recording Guide — How to Build a Good Dataset

This is the most important part of the project. The model can only be as
good as the audio you give it. Read this end-to-end before your first venue.

---

## 0. The mental model — and the reality

A SED model trained on your data answers one question per inference window:

> *"Is there a $TARGET clink in the last 1 second, given everything I've
> learned about how busy venues sound?"*

In an ideal world you'd record at the venue **during a busy Friday night**
and let the model learn directly from real conditions. In practice that's
almost never possible: install crews show up before opening, the place is
empty or near-empty, the music isn't on yet, the kitchen isn't running, and
the venue won't pay you to come back five times to capture peak hours.

So the workflow is *intentionally synthesis-heavy*: you record the **clean
ingredients** wherever you can capture them well, and the training pipeline
**combines them at training time** to produce realistic busy-venue audio.
This is exactly how state-of-the-art speech recognition handles noise — you
don't record every speaker in every café; you record clean speech and add
reverb + noise during training.

The four ingredients:

| Ingredient | Where | How often |
|---|---|---|
| **Clean glass clinks** (per glass type) | Studio, controlled — recorded once into the **library** | Once per glass type, ever |
| **Empty-venue acoustics** (room tone, HVAC, reverb signature) | At the venue, during install | Once per venue |
| **Busy ambience** (crowds, music, kitchen, conversations) | Off-site, anywhere genuinely busy — **library** | Build deeply once, reuse forever |
| **Hard-negatives** (cutlery, ice, bottle clanks, etc.) | Anywhere with that sound — **library** | Build deeply once, reuse forever |

Notice that **only one of four ingredients is venue-specific**. Everything
else is built into the shared library and amortised across every project
you'll ever do.

**Imbalance is good.** Real venues are 99.9 % "no clink". Reflecting that
ratio in training (lots of negatives, few positives) teaches the model the
prior and is what produces low FP rates. The pipeline does this for you;
you just need to make sure each *kind* of negative is well-represented.

---

## 1. Equipment & placement

### Microphones

- **Type**: small-diaphragm condenser or boundary mic with a wide cardioid
  pattern. Avoid shotgun mics (too directional) and dynamics (too quiet
  for distant clinks).
- **Phantom power on** for condensers (UMC404HD has +48V switch per pair).
- **Spacing**: aim mics at the bar / table area from 1–3 m. With 4 mics,
  cover the venue corners; the multi-mic inference takes the max across
  channels, so any one mic detecting fires the trigger.
- **Avoid**: speakers (gives PA bleed dominating the recording), kitchens
  (hood noise saturates everything), HVAC outlets (constant whoosh kills
  SNR for soft clinks).

### Interface

- **Behringer UMC404HD** is a fine cheap default (4 inputs, USB, Linux
  class-compliant). Set sample rate to 48 kHz on the device; the SED
  pipeline records WAVs at 48 kHz and resamples complete clips/windows to
  the model's 32 kHz feature rate internally. 44.1 kHz also works, but
  48 kHz is the better default for live/pro audio hardware.
- **Gain staging**: raise the gain knob until typical loud clinks peak
  around -10 to -3 dBFS in `clinkd audio meter`. This leaves headroom for
  unexpected loud events without clipping.
- Pin the interface in `projects/<id>/project.toml`; do not rely on the OS
  default input for deployed recording:

```toml
[audio]
record_device = "UMC404HD"   # stable name substring from `clinkd audio devices`
input_sample_rate = 48000    # hardware capture/WAV rate
sample_rate = 32000          # model feature rate for EfficientAT
record_channels = 4          # number of connected mics
```

- Run `clinkd audio meter --project <id>` before recording. Tap/clink near
  each mic and watch all channels move in the terminal at comparable levels.
- Run `clinkd check record --project <id>` BEFORE recording — it uses the
  same configured device/channels and surfaces silent channels, clipping,
  identical channels (interface mis-routing), and unpinned `default` input.

### Operator hygiene

- Don't talk during recording. Don't move chairs.
- Tell venue staff what you're doing and ask them to act normally.
- Keep a notebook: which class, which session, what was happening, any
  anomalies. The session_id in sidecars helps but human notes catch
  weirdness.

---

## 2. The four ingredients — per-class playbook

### A. Studio glass library — clean spectral references

`clinkd library record-glass --type <name> --num <count>`

**Purpose**: clean, dry, high-SNR recordings of each glass type the venue
serves. These define the spectral signature the model learns to recognise.
They're stored under `data_library/glasses_clean/<type>/` and reused on
every project that uses the same glassware.

**Setup**:
- quiet room (≤30 dBA noise floor); a closet with blankets is fine
- mic at 30–60 cm from the glasses (closer than venue distance — we want
  the cleanest possible source; venue acoustics get added at training time
  via domain mixing with the empty-venue tone and library ambience)
- non-resonant surface — a folded towel under the glass

**What to record per glass type (aim 60–100 clips each)**:
- Vary the **clink force**: from "barely touching" through "casual toast"
  to "enthusiastic". Include the full dynamic range you'd expect at the
  venue. **Especially capture quiet clinks** — these are the ones the
  augmenter struggles to keep audible after mixing, and the masking-check
  will skip them if you don't have enough quiet variants.
- Vary the **contact angle**: lip-to-lip, lip-to-bowl, bowl-to-bowl.
- Two-glass and three-glass clinks if the venue serves toasts.
- Use 1–2 actual venue glasses if you can, plus equivalents from your
  test set — small variations between physical glasses help the model
  generalise.

**Bad practice**:
- A single perfect "studio shot" clink replicated 50 times. Real clinks
  vary; your training set must too.
- Recording on a hard table without padding — gives a thuddy "hit table"
  attack alongside the clink.
- Only one glass per type. Two physical instances of "wine_thin" sound
  different enough to be worth capturing both.

### B. Empty-venue acoustics — the venue-specific ingredient

`clinkd record continuous --project <id> --class ambience --duration 30m --chunk 30`

**Purpose**: capture the venue's acoustic signature when no one is there.
This is the room tone, HVAC, refrigeration hum, distant traffic, the
reverb response — everything the venue contributes to a recording even
when nothing's happening. It's used at calibration time to *simulate*
busy conditions (room tone × library busy ambience) when you can't record
real busy hours, and at training time as one component of the ambience
pool.

**Setup**:
- Mics in their **final permanent positions** (so the model learns the
  venue → mic transfer function and the deployed mic placement).
- Whatever the venue normally has running: HVAC, fridges, ice machines,
  background music if it's always on. The point is to capture the
  *deployed silent state*, which is "no people, but everything else."

**Recipe**:
- 20–40 minutes total, broken into 30-second chunks.
- Capture both "completely empty" and "staff prep" if the venue allows —
  the latter has cutlery, dishwasher, voices in the back, which gives the
  model harder negatives.

**Also**: walk around clinking a few glasses yourself at ~5 different
table positions. Tag this as positive recording (`clinkd record auto` with
`--class clink_X`). Even a few hundred such on-site positives make a
huge difference: they anchor the model in **this venue's reverb and mic
positions**, which the library glass clips alone can't capture.

**Aim**: 100–300 on-site positives per glass class total, captured in
3–5 short sessions at different table positions. Quality > quantity —
30 well-placed positives at 3 different tables beats 500 from one spot.

**Bad practice**:
- Skipping empty-venue recording entirely. The simulation calibration
  needs it; the model needs the venue's reverb signature.
- Only standing at one table location. The session_id-based splitter
  needs ≥3 distinct sessions to give you meaningful val/test splits.

### C. Library busy ambience — the off-site pool

`clinkd library record-ambience --label <source> --density busy --duration 30m`

**Purpose**: realistic crowd noise, clattering cutlery, music, conversation
hubbub. This is the most important pool for FP rate, and it's the one most
people get wrong. **The trick is that you record it OFF-SITE**, anywhere
you have permission — your local bar, a friend's restaurant, a café — and
it ends up in the shared library where every future project benefits from
it. Build it deeply once.

**Where to record**:
- A genuinely busy bar, restaurant, or café you can sit in for an hour.
- Friend's venue at peak hours (most owners say yes if you explain).
- Any large food court, brewery, hotel lobby.
- **Not** open-air parks, libraries, or your kitchen — these have a
  different noise spectrum than indoor hospitality venues.

**What to record**:
- 30–60 minutes per source.
- **Vary density**: tag clips with `--density quiet` (early evening),
  `--density normal` (dinner service), `--density busy` (full house),
  `--density peak` (Friday 9pm with music up). The augmenter samples
  across these uniformly, so all four bands need representation.
- Continuous recording, NOT auto-trigger. Let the messiness in.

**How much is enough**:
- **Minimum viable**: 20 min total across 3 distinct sources, with at
  least one tagged `busy` or `peak`. The doctor will fail below this.
- **Comfortable**: 60 min across 5 sources, with full density coverage.
- **Excellent**: 3+ hours across 8+ sources, including unusual ones
  (jazz club, sports bar with TVs on, hotel lobby, busy café, etc.).

**Spectral diversity matters more than total minutes**. A model trained
on 20 minutes from 5 different bars generalises better than one trained
on 5 hours from one bar.

**Bad practice**:
- Recording only your favourite café (over-fits to that specific space).
- Recording with your phone in your pocket (muffled, low SNR — useless).
- Forgetting to tag density. The doctor warns when no busy/peak clips
  exist, because the model will never see realistic peak conditions.

**Where to put your recorder**: on the table next to your glass, mic side
up, not pointed at any single conversation. You're trying to record the
*aggregate* of the room, not any specific source.

### D. Hard negatives — the FP killers

`clinkd library record-hardneg --label <name> --duration 15m`

**Purpose**: tell the model that *this thing that sounds clinky is NOT
your target*. After domain mixing, this is the single biggest lever for
low FP rate. Hard negatives also live in the **library**, not on a
per-project basis — cutlery sounds the same at one venue as another, and
you should record once and benefit forever.

**Strategy**: think about what could trigger a false positive in the
deployed venue, then deliberately record those things off-site. Record
each in its natural setting (kitchen, café, hotel) for realism.

**Generic bar/restaurant hardneg list (record 100–300 each)**:

| Sound source           | Why it matters                            |
|------------------------|--------------------------------------------|
| Cutlery on plate       | Sharp metal-on-ceramic transient          |
| Cutlery in dishwasher tray | Multiple metal-on-metal events         |
| Ice cubes in glass     | Metallic-ish transients with reverb       |
| Bottle clank (glass-glass) | Thicker than wine; common at bar          |
| Bottle on counter      | Low-thud + high-tap                       |
| Wine glass set DOWN on bar | Often confused with a clink            |
| Espresso machine knock-out | Loud single transient                  |
| Coffee cup / saucer    | Ceramic on ceramic                        |
| Beer mug clinks        | If venue sells them, train as a hardneg   |
| Phone notification     | Spiky high-freq                           |
| Cash register / POS click | Mechanical click                       |
| Music transients (snare, hihat, claps) | Ride the same band as clinks |
| Talking laughter peaks | Plosives and shrieks                      |
| Background applause    | Many small claps overlap                  |
| Dropped object         | Random impulse                            |

**On-site hard negatives (the venue's own distractors)**: while you're at
the venue capturing empty acoustics, also record 5–10 minutes specifically
of any unusual venue-specific distractors — espresso machine if it's near
the mics, ice well agitation patterns, register click frequency, etc.
Tag these as positives kind=hardneg in the project (not the library) to
mark them as venue-specific.

---

## 3. Per-venue strategy

### High-end fine-glass restaurant (quiet, soft clinks)

- **Hardest case**: signal is quiet, FP penalty is high.
- Use `per_class_min_snr_db` in `[augmentation]` to clamp the lower SNR
  bound for fragile classes (e.g. `{ clink_wine_thin = 5.0 }`). This stops
  domain mixing from burying soft clinks under heavy ambience.
- Place mics CLOSER to typical table positions (≤2 m) to compensate for
  signal weakness.
- Record more on-site positives (300+ from multiple tables) — recall is
  harder when SNR is low.
- Calibrate to ≤1 FP/h; use `--ambience-source simulated_busy` if you
  can't capture real peak hours.

### Busy noisy bar (loud clinks, lots of distractors)

- Glass clinks are louder; FP rate from cutlery + bottle clanks dominates.
- **Library hard-negatives are the priority** — invest in the bottle
  clank, ice well, and music transient pools.
- Per-class SNR floor can be lower (`{ clink_pint = -2.0 }` is fine) since
  the actual SNR at the venue is closer to that.
- Calibrate to 2–3 FP/h — venue is forgiving and recall matters more.

### Multi-glass discrimination (e.g. wine vs champagne)

- The two glasses must sound spectrally different. Record both at the
  studio with the SAME mic distance and gain so the model learns the
  shape, not the level.
- Ensure both are well-represented in venue on-site recordings — same
  number of sessions, same mic setup, same time.
- Bump `margin_to_second_best` to 0.12 if you keep seeing wrong-class
  triggers; recall drops slightly but inter-class confusion drops more.

### Outdoor / patio venue

- Wind, traffic, and overhead aircraft change the noise profile entirely.
- Library ambience pool needs outdoor recordings or the model will over-fit
  to indoor noise. Add 2–3 sources from outdoor venues.
- Wind protection on mics is essential (foam + furry cover for outdoors).

---

## 4. The pre-flight before each step

The `clinkd check` commands surface common issues before you waste time.

```bash
# Before recording a new session
clinkd audio devices
clinkd audio meter --project <id>
clinkd check record --project <id>

# Before training (validates dataset, library coverage)
clinkd check train --project <id>

# Before deploying (model latency, ArtNet, systemd)
clinkd check deploy --project <id>

# All three at once
clinkd check all --project <id>
```

`clinkd check train` specifically checks the empty-venue workflow:
- Ambience pool **duration** (FAIL if empty, WARN if <20 min)
- Ambience **diversity** (WARN if <3 distinct sources)
- Ambience **density coverage** (WARN if no `busy`/`peak` clips)
- Per-class SNR floors are reasonable for the configured classes

`clinkd library ambience-coverage` shows the same metrics in detail:

```bash
clinkd library ambience-coverage
```

---

## 5. Verification — do this BEFORE training

After recording sessions, **always**:

```bash
clinkd inspect --project <id> --open
```

The HTML report shows:
- per-class clip counts and total duration
- spectrograms + audio playback for sample clips
- a **UMAP scatter** of all clips coloured by class — outliers stand out
- **Augmentation Preview**: side-by-side spectrograms of clean target vs
  domain-mixed target, with a masking score (red/yellow/green) so you can
  see which mixes preserve the target
- per-class outlier candidates flagged → `reports/outliers.csv`

**Listen to flagged outliers.** If they're genuinely bad (mic hit, wrong
glass, mid-sentence speech), quarantine them:

```bash
clinkd quarantine --project <id> reports/outliers.csv
```

If the augmentation preview shows >30 % red samples for a class, you have
a masking problem:
- Record more positives, especially **quieter** ones — the masking check
  fails when the target's transient is weaker than the band-energy of the
  mixed ambience at the same time.
- Raise the per-class SNR floor: `per_class_min_snr_db.<class> = 5.0`.
- Record more diverse ambience — the masking score depends on which
  ambience is randomly drawn; a thin pool gives consistently bad luck.

---

## 6. Common mistakes & their symptoms

| Symptom in production | Most likely cause |
|---|---|
| Triggers fire on cutlery | No cutlery hardneg in library; record `clinkd library record-hardneg --label cutlery` |
| Triggers fire on music transients | No music in ambience pool; library lacks venues with PA |
| Silent at peak hours, fine when quiet | Too few quiet on-site positives; SNR floor too high; ambience pool dominates the augmented mix |
| Triggers fire 10× per hour | Threshold not calibrated, or calibrated against empty-venue ambience instead of `simulated_busy` |
| Wrong glass class triggers | Glass classes too similar spectrally; raise `margin_to_second_best`; consider merging |
| One mic always silent in live table | Cable / channel routing — `clinkd check record` catches this |
| Model only fires for 1-2 specific staff | Too few sessions / one staff member dominated training; record more sessions with different people |
| Live performance very different from val | Session leakage in splits — re-run `clinkd dataset compose`, check `clinkd check train` |
| F1 plateau ≤0.5 in training | Insufficient positive variance OR ambience pool too thin → augmenter sees same mixes every epoch |

---

## 7. How much data is enough?

| Tier | Per-class positives | Library ambience | Library hardnegs | Calibration |
|---|---|---|---|---|
| **Minimum viable** | 60 studio + 100 on-site | 20 min, 3 sources | 200 clips total | `simulated_busy` against empty venue |
| **Comfortable** | 80 studio + 250 on-site | 60 min, 5 sources, all densities | 600 clips, 6+ labels | Real venue ambience after deployment |
| **Excellent** | 100 studio + 500 on-site | 3 hr, 8 sources | 1500 clips | Multiple calibration passes over weeks |

The library investment **pays compound interest**. Hour 1 you record library
ambience and have it; project 5 you reach into it for free. The first
project takes ~6 hours of recording across studio + venue + off-site
library; subsequent projects take ~2 hours (just venue empty + on-site
positives) because the library is already there.

---

## 8. After deployment: the second visit

Once the system is running at the venue, **come back during a real busy
night** with the same audio interface and capture an hour of actual
ambience:

```bash
clinkd record continuous --project <id> --class ambience \
    --duration 1h --chunk 60
```

Then re-calibrate against THAT real ambience:

```bash
clinkd calibrate --project <id> --ambience-source venue \
    --target-fp-per-hour 1.0
```

This refines the thresholds you initially set with `simulated_busy`,
typically dropping FP rate further with no recall loss.

A second visit is worth 10× more than another hour of training data — the
model is fine; it's the thresholds that benefit from real ambience.
