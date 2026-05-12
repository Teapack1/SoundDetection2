# SED — Sound Event Detection for Interactive Lighting

Detect specific sounds (glass clinks, etc.) in noisy venues and trigger
ArtNet/DMX lighting cues. Runs 24/7 on a fanless Intel edge PC.

```
   mic(s) ──▶ edge computer ──▶ ArtNet ──▶ DMX fixtures
                  ▲
                  └── new models trained on a workstation
```

## Two computers

| | Training workstation | Edge device at venue |
|---|---|---|
| Hardware | Laptop / desktop with GPU | Onlogic CL210 / CL260 |
| OS | anything | Ubuntu Server 24.04 LTS, headless |
| Job | record library, compose datasets, train, export, calibrate | record on-site, run inference 24/7 |
| Install | `uv sync --extra train --extra dev` | `deploy/install.sh` |

## Repo layout

```
data_library/                shared sound assets (reusable across all projects)
  glasses_clean/<type>/      clean glass clips, recorded once per glass
  ambience/<label>/          busy ambience from any venue
  negatives_external/<...>   hard negatives (cutlery, ice, bottles, etc.)
  manifest.jsonl

projects/<venue>/            one folder per venue
  project.toml               classes, ArtNet config, audio device
  recordings/
    positives/<class>/       on-site target clips (triggered)
    hardneg/<class>/         venue-specific distractors (triggered)
    ambience/                empty-venue room tone (continuous)
  dataset/                   train.csv val.csv test.csv classes.json
  models/                    *.ckpt, model_int8.xml/.bin, manifest.json
  reports/                   HTML inspection, training curves

src/sed/                     Python package
deploy/                      systemd units + install.sh for the edge device
```

The library is built up over time; projects reuse it. A new venue typically
needs only ~2 h of new recording because most ingredients come from the
library.

## Commands

The CLI binary is **`clinkd`** (clink-detector). All commands print
`-h`/`--help`.

```
clinkd audio     list devices, live input meter, probe USB audio
clinkd library   record into / manage the shared sound library
clinkd project   create / list / inspect venue projects
clinkd record    record positives / hardnegs / ambience for a project
clinkd dataset   compose train/val/test from recordings + library
clinkd inspect   HTML report with spectrograms, UMAP, augmentation preview
clinkd train     fine-tune the model (workstation only)
clinkd export    export to OpenVINO INT8 for the edge device
clinkd calibrate sweep per-class thresholds against ambience
clinkd run       24/7 inference loop (uses systemd on the edge)
clinkd check     pre-flight test suites: record / train / deploy / all
clinkd quarantine flag bad clips so they're excluded from the next compose
```

### Key concepts

**Why two computers** — torch + lightning is ~3 GB and needs a GPU. The
edge device only needs OpenVINO + numpy (~150 MB). Training happens on the
workstation; you copy a ~5 MB INT8 model to the edge.

**Synthesis-first training** — install crews show up before the venue
opens. You can't record at peak hours. So:
1. Record clean glass clips in a quiet room (library, reusable).
2. Record busy ambience separately, off-site, anywhere with permission
   (library, reusable).
3. Record empty-venue acoustics + a few hundred on-site positives at the
   target venue (one short visit).
4. At training time, **domain mixing** combines clean positives with
   random ambience at random SNR — synthesising realistic busy-venue
   audio without ever recording it. A masking-aware safety net drops mixes
   where the clink would be inaudible.

**Calibration** is choosing the right detection threshold per target class,
not adjusting the microphone. `clinkd calibrate` runs the model offline
over recorded ambience and sweeps thresholds, picking the lowest one that
gives ≤ N false positives per hour. Three modes for the ambience source:
`venue` (real recordings), `simulated_busy` (empty-venue + library mixed —
the default for first installs), `library` (library only).

**The `check` command** runs pre-flight tests: `check record` before
recording (mic gains, dead channels), `check train` before training
(dataset stats, ambience pool depth), `check deploy` before deployment
(model loads, latency budget, ArtNet reachable). Each test prints
PASS/WARN/FAIL with a fix hint.

### How `clinkd` is wired

`pyproject.toml [project.scripts]` declares `clinkd = "sed.cli:app"`.
`uv sync` (or `pip install -e .`) creates `.venv/bin/clinkd` as a shim
that imports `sed.cli` and invokes the Typer `app`. All command groups
are decorated functions in `src/sed/cli.py`. `clinkd ... --help` shows
options auto-generated from each function's signature.

You can also run modules directly: `python -m sed.cli run --project x`.

### Negative-class strategy

Default: a single `_negative` class that lumps all hard-negatives,
ambience, and irrelevant sound together. The arbiter gates on its
smoothed probability — if `_negative > negative_max`, no trigger fires
even when a target class also has high probability.

If a specific distractor proves problematic (e.g. cutlery is causing 80 %
of your FPs), you can add an explicit class like `neg_cutlery` in
`project.toml` and the arbiter will gate on the max over all negative
classes. Cost: each new negative class needs ~100+ clips. Most users
keep the single `_negative`.

## Next

- [WORKFLOW.md](WORKFLOW.md) — step-by-step setup for both machines.
- [RECORDING_GUIDE.md](RECORDING_GUIDE.md) — what to record, where, how
  much, what makes a clip good or bad.
- [DESIGN.md](DESIGN.md) — design rationale (focal loss, smoothing chain,
  multi-mic, masking-aware mixing, ...).

## 5-minute install (workstation)

```bash
git clone <repo> sed && cd sed
git init && git submodule add https://github.com/fschmid56/EfficientAT vendor/EfficientAT
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv BEFORE clinkd is on PATH
uv sync --extra train --extra dev                 # ~3 GB on first run
uv run clinkd --help
uv run pytest                                     # ≈ 24 tests
```

If `pytest` errors with "No such file or directory", re-run with
`--extra dev`. If `uv` install errors come up referencing strange
substitution patterns, an old `sed` (our binary) is shadowing the Unix
stream editor — remove it from PATH, or just upgrade: the binary is now
`clinkd`.

MIT license.
