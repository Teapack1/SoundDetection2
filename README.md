# SoundDetection2

A **Sound Event Detection** system for reliable, venue-specific, sound-reactive lighting.

## Overview

**SoundDetection2** listens to **microphones**, detects real-time sound events such as
**cheering** or **glass clinks**, and sends **ArtNet/DMX** triggers to lighting.
It is built for **interactive installations** in real venues, where **false positives**
matter more than demo accuracy.

Clean target sounds are recorded, mixed offline with **ambience** and **negative examples**,
trained **per venue**, and calibrated to a target false-positive rate.
Work is organized into **projects**, usually one per venue.

## Flow

```text
Workstation: record -> dataset -> train -> calibrate -> inference smoke test
                         |
                         v
                    model bundle
                         |
                         v
Edge PC:     live inference -> ArtNet/DMX -> lighting
```

The **workstation** handles recording, dataset composition, training, calibration,
and a no-ArtNet live inference smoke test before deployment.
The **edge PC** runs continuous CPU inference with **OpenVINO** and talks to the
lighting controller.

## CLI

The CLI binary is **`sed`**, managed with **`uv`**. Its main surface includes
`audio`, `project`, `library`, `dataset`, `models`, `evaluate`, and `check`, plus
top-level commands for **recording**, **training**, **calibration**, and **live runtime**.

Two of them open a browser UI instead of a terminal one:

```bash
uv run sed record --serve --project <id>            # recording console  (port 8790)
uv run sed dataset inspect --project <id> --serve   # dataset / training (port 8765)
```

## Documentation

| Guide | Use this when... |
| --- | --- |
| [SETUP.md](SETUP.md) | **Install and provision** a machine: workstation, edge PC, or network. Start here once per machine. |
| [WORKFLOW.md](WORKFLOW.md) | **Run the system** end to end: create a project, record, train, calibrate, deploy, and run. Day-to-day usage. |
| [DESIGN.md](DESIGN.md) | **Understand the architecture** and dataset-recording best practices. Read this to understand why and to capture good data. |
| [docs/pretraining_guide.md](docs/pretraining_guide.md) | **Design stage one**: the class space, audio feature, normalization and architecture the shared-corpus encoder is built with — and that every venue run inherits. |
| [docs/finetuning_guide.md](docs/finetuning_guide.md) | **Design stage two**: adapting that encoder to one venue's glassware, and diagnosing a fine-tune that underperforms. |

**Requirements:** Python 3.11 or 3.12, **`uv`**, a Linux or Windows workstation, and a Linux edge PC.
