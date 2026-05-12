# SoundDetection2 - Agent Guidelines & Project Overview

## 1. Project Objective & Venue Deployment
**SoundDetection2** is a real-time **Sound Event Detection (SED)** system deployed in high-end venues (luxury bars, clubs, karaoke lounges) for sound-reactive dynamic lighting installations.
- **Hardware Setup:** 1 or more microphones in lighting fixtures connected to an audio interface; an Edge PC runs continuous CPU inference to trigger ArtNet/DMX lighting effects.
- **Target Sounds:** Distinct glass clinging sounds (up to 3 distinct glass classes, including thin/fine/quiet glasses with transient duration ~0.2s).
- **Venue Noise & Extreme Class Imbalance:** Venues are loud (music, speech, cutlery, background glass clinks). Inference runs continuously, so **>99.9% of streaming windows are negative background audio**. Target events occur rarely.

## 2. Core Duality & Agent Research Role
1. **Venue Production Deployment:** Training venue-tuned models, performing frequency-selective noise overlays, calibrating decision thresholds to guarantee ultra-low False Positive Rate (FPR), and exporting OpenVINO model bundles.
2. **ML Research & Feature Experimentation:** Researching custom model backbones, classification heads, higher sample rates (44.1kHz / 48kHz), and detailed audio features. Agents should evaluate the best possible approach for this task—even if it breaks standard 32kHz pretrained contracts (e.g. EfficientAT).
*(See [.agents/skills/sed-ml-research/SKILL.md](.agents/skills/sed-ml-research/SKILL.md) for detailed ML and augmentation rules).*

## 3. Non-Negotiable Constraints
- **Ultra-Low False Positive Rate (FPR) > Demo Accuracy:** Triggering a lighting effect when no real clink occurred ruins the luxury installation. Strict FPR threshold calibration against venue negative audio is mandatory.
- **Clip Context Window (~0.4s):** Primary model window context is ~0.4s to prevent short ~0.2s transients from being diluted in longer background windows.
- **Frequency-Selective Noise Augmentation:** Never bury quiet, high-frequency glass transients under dense noise during augmentation.
- **No Swallowed Exceptions:** Never swallow audio processing errors silently or use dummy fallback data.

## 4. Tooling & Commands
- **Environment Manager:** `uv`
- **Main CLI Entrypoint:** `sed` (e.g., `uv run sed dataset inspect`, `uv run sed train`, `uv run sed record`).
- **Tests:** `uv run pytest`
- **Linting & Formatting:** `uv run ruff check .` / `uv run ruff format .`
