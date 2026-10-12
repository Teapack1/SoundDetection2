"""Advisory checks for audio features and model choices.

This is a research tool: every feature and every backbone stays available and
supported. Nothing here blocks or removes a configuration — it explains what a
choice costs for *this* task so an unusual one is deliberate rather than
accidental.

The task shapes the advice. A glass clink is a set of damped modal resonances:
a fundamental around 400 Hz-3 kHz, a partial comb reaching to 20 kHz, and a
decay of tens to hundreds of milliseconds. Two facts follow, and most of the
findings below are consequences of them:

  * The **upper partials carry the glass identity** and sit where venue noise is
    weakest (speech, music and HVAC are strongly low-frequency dominated). Any
    front-end that spends its resolution low and smears high is spending it
    where the noise is.
  * The event is **short**, so time resolution competes with frequency
    resolution in the usual way, and the window has to be long enough to contain
    the decay that distinguishes one glass from another.
"""

from __future__ import annotations

import math

# Where the discriminative partial structure lives for glassware.
GLASS_BAND_LO_HZ = 2000.0
GLASS_BAND_HI_HZ = 16000.0

# Above this, a mel band is wide enough to merge partials that separate one
# glass from another (they sit tens to a couple of hundred Hz apart up here).
MEL_SMEAR_WARN_HZ = 250.0

# Low-frequency energy that is almost pure venue noise for this task.
RUMBLE_HZ = 150.0

SEVERITY_ORDER = ("error", "warn", "info")


def _mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_band_width_hz(freq_hz: float, *, n_mels: int, fmin: float, fmax: float) -> float | None:
    """Width of the mel band covering ``freq_hz`` — how far apart two partials
    must be before this front-end can tell them apart."""
    if n_mels <= 0 or fmax <= fmin or not (fmin <= freq_hz <= fmax):
        return None
    step = (_mel(fmax) - _mel(fmin)) / n_mels
    return _hz(_mel(freq_hz) + step) - freq_hz


def feature_advice(features, sample_rate: int) -> list[dict]:
    """Advisory findings for a feature configuration. Never blocks.

    ``features`` is a `FeaturesCfg` (or anything with the same attributes).
    """
    out: list[dict] = []
    ftype = str(getattr(features, "feature_type", "mel") or "mel")
    n_mels = int(getattr(features, "n_mels", 0) or 0)
    n_fft = int(getattr(features, "n_fft", 0) or 0)
    hop = int(getattr(features, "hop_length", 0) or 0)
    win = int(getattr(features, "win_length", 0) or 0)
    fmin = float(getattr(features, "fmin", 0.0) or 0.0)
    fmax = float(getattr(features, "fmax", 0.0) or 0.0)
    window_s = float(getattr(features, "window_seconds", 0.0) or 0.0)
    nyquist = sample_rate / 2.0
    fmax_eff = fmax or nyquist

    # --- coverage of the band that carries glass identity -------------------
    if fmax_eff < GLASS_BAND_HI_HZ:
        out.append({
            "severity": "warn", "code": "fmax_below_glass_band",
            "message": (
                f"fmax is {fmax_eff:.0f} Hz, so everything above it is discarded. The "
                f"partial comb that separates one glass from another runs to about "
                f"{GLASS_BAND_HI_HZ:.0f} Hz, and that band is also where venue noise is "
                "weakest. Raising fmax (and the sample rate if it caps you) is the most "
                "direct way to give the model the evidence it needs."
            ),
            "suggest": {"fmax": min(nyquist, 20000.0)},
        })
    if nyquist < GLASS_BAND_HI_HZ:
        out.append({
            "severity": "warn", "code": "sample_rate_caps_glass_band",
            "message": (
                f"A {sample_rate} Hz sample rate has a {nyquist:.0f} Hz ceiling, so the "
                "upper partials cannot be represented at any fmax. 32 kHz keeps most of "
                "the band; 44.1 kHz keeps all of it."
            ),
        })

    # --- fmin: bands spent where only noise lives ---------------------------
    if fmin < RUMBLE_HZ:
        out.append({
            "severity": "info", "code": "fmin_includes_rumble",
            "message": (
                f"fmin is {fmin:.0f} Hz. Below roughly {RUMBLE_HZ:.0f} Hz there is almost "
                "no clink energy but plenty of HVAC, footfall and kick drum. Raising fmin "
                "to 150-300 Hz spends those bands on signal instead, and with absolute "
                "(not per-clip) normalisation it also stops loud low-frequency noise from "
                "shifting the whole input."
            ),
            "suggest": {"fmin": 200},
        })

    # --- mel resolution where the identity information is -------------------
    if ftype == "mel" and n_mels > 0 and fmax_eff > GLASS_BAND_LO_HZ:
        probe = min(8000.0, fmax_eff)
        width = mel_band_width_hz(probe, n_mels=n_mels, fmin=fmin, fmax=fmax_eff)
        if width is not None and width > MEL_SMEAR_WARN_HZ:
            out.append({
                "severity": "info", "code": "mel_smears_upper_partials",
                "message": (
                    f"At {probe:.0f} Hz each of the {n_mels} mel bands spans about "
                    f"{width:.0f} Hz, so partials closer together than that are merged. "
                    "Mel resolves the fundamental well and smears the upper comb — which "
                    "is why linear spectrograms often look better here. More bands "
                    "(192-256) is the cheaper half of that benefit; a linear front-end "
                    "buys the rest at 3-7x the input rows."
                ),
                "suggest": {"n_mels": 256 if n_mels < 256 else n_mels},
                "detail": {"probe_hz": probe, "band_width_hz": width},
            })

    # --- empty filters: bands that carry nothing at all ---------------------
    if ftype == "mel" and n_mels > 0 and n_fft > 0:
        bin_hz = sample_rate / n_fft
        lowest = mel_band_width_hz(max(fmin, 1.0), n_mels=n_mels, fmin=fmin, fmax=fmax_eff)
        if lowest is not None and lowest < bin_hz:
            # n_fft that would make the narrowest band at least one bin wide.
            needed = 1 << max(0, math.ceil(math.log2(sample_rate / max(lowest, 1e-6))))
            out.append({
                "severity": "warn", "code": "empty_mel_filters",
                "message": (
                    f"The lowest mel bands are narrower ({lowest:.0f} Hz) than one FFT bin "
                    f"({bin_hz:.0f} Hz), so some carry no energy and render as empty rows. "
                    "This is the real tension in a dense mel bank: more bands need more FFT "
                    "resolution, and more FFT resolution costs time resolution on a short "
                    f"transient. Options are n_fft {needed} (sharper in frequency, blunter "
                    "in time), fewer bands, or a higher fmin."
                ),
                "suggest": {"n_fft": needed},
                "detail": {"narrowest_band_hz": lowest, "fft_bin_hz": bin_hz},
            })

    # --- time resolution vs the transient -----------------------------------
    if hop > 0:
        hop_ms = hop / sample_rate * 1000.0
        if hop_ms > 12.0:
            out.append({
                "severity": "info", "code": "hop_coarse_for_transients",
                "message": (
                    f"A {hop_ms:.0f} ms hop gives few frames across a clink's attack. "
                    "5-10 ms resolves the onset and the start of the decay, which is "
                    "where the model gets its evidence for a quiet event."
                ),
                "suggest": {"hop_length": max(1, int(sample_rate * 0.005))},
            })
    if win > 0 and hop > 0:
        win_ms = win / sample_rate * 1000.0
        if win_ms > 25.0:
            out.append({
                "severity": "info", "code": "window_smears_onset",
                "message": (
                    f"A {win_ms:.0f} ms analysis window smears a sharp onset across "
                    "frames. A shorter win_length with the same (or larger) n_fft keeps "
                    "the frequency grid while sharpening the attack."
                ),
            })

    # --- clip window vs the decay -------------------------------------------
    if 0 < window_s < 0.25:
        out.append({
            "severity": "warn", "code": "window_shorter_than_decay",
            "message": (
                f"A {window_s * 1000:.0f} ms window may cut the ring before it decays. "
                "The decay rate is part of what separates a thin crystal glass from a "
                "thick tumbler, so truncating it removes discriminative evidence."
            ),
        })

    # --- what actually normalizes the values ---------------------------------
    # With no dataset scaling the mel path still applies an affine — AudioSet's
    # fixed (x + 4.5) / 5. That constant was fitted at 32 kHz / 128 mel / 15 kHz
    # fmax, so on any other geometry it is an arbitrary offset and gain rather
    # than a normalization.
    scaling_method = str(getattr(features, "scaling_method", "none") or "none").lower()
    scaling_source = str(getattr(features, "scaling_source", "dataset") or "dataset").lower()
    fitted = scaling_method != "none" and scaling_source == "dataset"
    on_contract = bool(
        callable(getattr(features, "matches_pretrained_normalization", None))
        and features.matches_pretrained_normalization()
        and sample_rate == 32000
        and n_mels == 128
        and fmax_eff == 15000
    )
    if not fitted and not on_contract and not bool(
        getattr(features, "pcen", False)
    ) and not bool(getattr(features, "db_norm", False)):
        out.append({
            "severity": "warn", "code": "scaling_none_off_contract",
            "message": (
                "Nothing about this feature's normalization was measured on your data. "
                "The values reach the model through AudioSet's fixed (x + 4.5) / 5 affine "
                "— a constant measured at 32 kHz / 128 mel / 15 kHz fmax, which this "
                "feature is not. Fitting per-band statistics on your own training split is "
                "the standard practice and gives every band comparable scale; the constants "
                "are frozen into the bundle and reused in evaluation, calibration and "
                "deployment."
            ),
            "suggest": {
                "scaling_method": "zscore", "scaling_scope": "per_band",
                "scaling_source": "dataset",
            },
        })
    elif fitted and str(
        getattr(features, "scaling_scope", "global") or "global"
    ) == "global" and ftype == "mel" and n_mels >= 64:
        out.append({
            "severity": "info", "code": "scaling_global_on_dense_bands",
            "message": (
                f"Scaling is fitted as one value pair for all {n_mels} bands. A mel "
                "spectrogram's band energies differ by orders of magnitude between the "
                "low and high end, so a single scale leaves the quiet upper partials — "
                "the ones carrying glass identity — near the bottom of the input range. "
                "Per-band is the usual choice at this resolution."
            ),
            "suggest": {"scaling_scope": "per_band"},
        })

    out.sort(key=lambda f: SEVERITY_ORDER.index(f["severity"]))
    return out


def model_advice(model, features, sample_rate: int) -> list[dict]:
    """Advisory findings for a backbone choice. Never blocks or removes.

    Every architecture in the catalog stays trainable; this only explains what a
    given pairing costs on this task.
    """
    out: list[dict] = []
    backbone = str(getattr(model, "backbone", "") or "")
    pretrained = bool(getattr(model, "pretrained", False))
    init = str(getattr(model, "initialization", "") or "")
    fmax = float(getattr(features, "fmax", 0.0) or 0.0) or sample_rate / 2.0

    # A frontend pinned below the glass band cannot represent the evidence,
    # whatever the architecture does with it.
    if backbone.startswith("ced") and fmax < GLASS_BAND_HI_HZ:
        out.append({
            "severity": "warn", "code": "backbone_frontend_caps_band",
            "message": (
                f"{backbone} is pinned to a 16 kHz / 8 kHz-fmax front-end, so the upper "
                "partials that distinguish glassware are not in its input at all. It "
                "remains fully supported and is a reasonable general audio-tagging "
                "encoder — just expect it to separate glass types poorly here."
            ),
        })

    # ImageNet weights on absolutely-normalised spectrograms transfer badly, and
    # this project has measured it (0.42 F1 upstream vs 0.967 from scratch on the
    # same backbone).
    timm_like = any(backbone.startswith(p) for p in ("resnet", "efficientnet", "convnext", "mobilenet"))
    if timm_like and (pretrained or init == "upstream"):
        out.append({
            "severity": "info", "code": "imagenet_init_transfers_poorly",
            "message": (
                "ImageNet weights are trained on natural images, not on absolutely "
                "normalised spectrograms. Runs in this project have collapsed with "
                "upstream init on the same backbone that trained cleanly from scratch. "
                "If this run underperforms, try random initialisation before changing "
                "architecture."
            ),
            "suggest": {"initialization": "random"},
        })

    # A non-native geometry on the task-specific encoder is legal and buildable
    # — say what it costs rather than refusing to run it.
    if backbone == "clink-cnn-v2":
        try:
            from types import SimpleNamespace

            from sed.config import clink_cnn_v2_native_deviations

            deviations = clink_cnn_v2_native_deviations(
                SimpleNamespace(model=model, features=features)
            )
        except Exception:
            deviations = []
        if deviations:
            out.append({
                "severity": "info", "code": "clink_cnn_v2_off_native_geometry",
                "message": (
                    "This is not ClinkCNN-v2's native geometry (" + "; ".join(deviations)
                    + "). The trunk is fully convolutional with an adaptive frequency "
                    "pool, so it builds and trains here — but the receptive fields and "
                    "the frequency-coordinate planes were sized for 44.1 kHz over 0.4 s, "
                    "so treat this as an experiment rather than the tuned configuration."
                ),
            })

    # AudioSet MobileNets are strong, but their weights were fitted at a
    # specific geometry; far from it, the prior is worth less than it looks.
    if backbone.startswith(("mn", "dymn")) and (pretrained or init == "upstream"):
        drift = []
        if sample_rate != 32000:
            drift.append(f"{sample_rate} Hz vs 32 kHz")
        if fmax > 15000:
            drift.append(f"fmax {fmax:.0f} vs 15 kHz")
        if drift:
            out.append({
                "severity": "info", "code": "upstream_contract_drift",
                "message": (
                    "AudioSet weights were fitted at 32 kHz / 15 kHz fmax / 10 ms hop; "
                    "this run differs (" + ", ".join(drift) + "). The encoder is fully "
                    "convolutional so it still trains, but the pretrained prior is worth "
                    "less the further the input geometry drifts. Domain-pretraining on "
                    "your own corpus and initialising from that run is usually stronger."
                ),
            })

    out.sort(key=lambda f: SEVERITY_ORDER.index(f["severity"]))
    return out


# A fine-tune is "small" once the val split is thin enough that its metrics move
# by more than the differences being measured.
SMALL_TRANSFER_ROWS = 2000

# Masking this share of the feature rows starts removing the narrow partials
# that separate one glass from another rather than just adding invariance.
SPECAUG_FREQ_MASK_WARN_SHARE = 0.10


# Origin buckets whose names say "this is a near-miss", not "this is room
# tone". Kept deliberately small and English/German-mixed to match the library's
# actual folder names; it only powers a suggestion, never a rule.
_NEAR_MISS_HINTS: tuple[str, ...] = (
    "metal", "cutlery", "cuttlery", "besteck", "fork", "knife", "spoon",
    "glass", "glas", "bottle", "flasche", "klopf", "schnalz", "tap", "clink",
)


def pipeline_stage(cfg) -> dict:
    """Which stage of the two-stage workflow this run is.

    Stage one shapes a general encoder on a large corpus; stage two adapts it to
    one venue's glassware. They want different depths, different regularization
    and a different class space, so almost every other finding reads differently
    depending on which one this is.
    """
    initialization = str(getattr(cfg.model, "initialization", "upstream") or "upstream")
    if initialization == "parent":
        return {
            "stage": 2,
            "label": "venue fine-tune",
            "parent_project": str(getattr(cfg.model, "parent_project", "") or ""),
            "parent_run": str(getattr(cfg.model, "parent_run", "") or ""),
            "frontend_source": "inherited from the parent run",
        }
    return {
        "stage": 1,
        "label": "domain pretrain",
        "parent_project": "",
        "parent_run": "",
        "frontend_source": (
            "this project"
            if str(getattr(cfg.features, "scaling_source", "dataset")) == "dataset"
            and str(getattr(cfg.features, "scaling_method", "none")) != "none"
            else "frontend default"
        ),
    }


def _target_classes(cfg) -> list[str]:
    return [str(c) for c in getattr(cfg.model, "classes", []) if str(c) != "_negative"]


def _cold_head_classes(cfg) -> list[str]:
    """Classes whose classifier row would start random on a parent transfer.

    Reads the parent's real class list, so the answer is the same one
    ``initialize_encoder_from_parent`` will produce at train time. An
    unreadable parent yields no finding rather than a guess.
    """
    try:
        from sed.train.transfer import parent_classes, resolve_head_sources
    except Exception:  # torch-free callers still get the rest
        return []
    parent_project = str(getattr(cfg.model, "parent_project", "") or "")
    parent_run = str(getattr(cfg.model, "parent_run", "") or "")
    if not parent_project or not parent_run:
        return []
    try:
        names = parent_classes(parent_project, parent_run)
    except Exception:
        return []
    children = [str(c) for c in getattr(cfg.model, "classes", [])]
    if not names or not children:
        return []
    sources = resolve_head_sources(
        names, children, dict(getattr(cfg.training, "transfer_head_map", {}) or {}),
    )
    return [name for name, source in sources.items() if source == "random"]


def training_advice(cfg, dataset: dict | None = None) -> list[dict]:
    """Advisory findings for the training/fine-tuning stage. Never blocks.

    Feature and model advice answer "can the evidence reach the model". This
    answers the next question — "will this schedule actually learn the
    distinction being asked for" — which is where a chained
    pretrain → fine-tune run usually goes wrong: the run is configured as a
    generic transfer while the task it is being asked to solve is separating
    variants of ONE sound, and those two need different settings.
    """
    from sed.train.fine_tuning import (
        fine_tune_policy_stage_count,
        is_transfer_initialization,
        resolve_fine_tune_plan,
    )

    out: list[dict] = []
    data = dict(dataset or {})
    rows = int(data.get("rows") or 0)
    drift = data.get("scaling_drift") if isinstance(data.get("scaling_drift"), dict) else None
    training = cfg.training
    features = cfg.features
    targets = _target_classes(cfg)
    transfer = is_transfer_initialization(cfg)
    from_parent = str(getattr(cfg.model, "initialization", "") or "") == "parent"
    plan = resolve_fine_tune_plan(cfg)
    trainable_stages = fine_tune_policy_stage_count(plan.target_depth)
    frozen_encoder = plan.target_depth in ("head", "head_only")
    # Two or more target classes that are variants of the same event: telling
    # them apart is a different problem from detecting the event at all.
    fine_grained = len(targets) >= 2

    # --- is there anything to absorb the near-misses? -----------------------
    # One `_negative` row has to model room tone, music, speech and cutlery at
    # once. It handles the first three easily and the last one badly, because
    # cutlery shares the target's attack and its bright partials — and cutlery
    # is what actually fires in a venue. A separate class gives the model a
    # place to put that decision, and (since the decision layer now lets a
    # non-firing class win a window) an effect at inference.
    roles = {}
    try:
        roles = cfg.class_roles([str(c) for c in getattr(cfg.model, "classes", [])])
    except Exception:
        roles = {}
    has_suppressor = any(role == "suppressor" for role in roles.values())
    near_miss_origins = sorted({
        str(name) for name in (data.get("negative_origins") or [])
        if any(hint in str(name).lower() for hint in _NEAR_MISS_HINTS)
    })
    if not has_suppressor and near_miss_origins:
        out.append({
            "severity": "info", "code": "near_miss_negatives_unsplit",
            "message": (
                "The `_negative` class is drawing from source buckets that sound like "
                "near-misses rather than background: "
                + ", ".join(near_miss_origins[:6])
                + (" and others" if len(near_miss_origins) > 6 else "")
                + ". Everything else — room tone, music, speech, mixed clips — can stay "
                "in `_negative`; moving just these into a `hardnegs` class gives the "
                "model a place to put the decision it actually gets wrong. Leave it out "
                "of `[targets]` (or set enabled = false) and it competes for each window "
                "without ever firing, so winning it suppresses the trigger."
            ),
            "detail": {"origins": near_miss_origins},
        })

    # --- separating sibling classes ----------------------------------------
    if fine_grained:
        margin = float(getattr(cfg.inference, "margin_to_second_best", 0.0) or 0.0)
        if margin < 0.15:
            out.append({
                "severity": "info", "code": "margin_too_small_for_siblings",
                "message": (
                    f"{len(targets)} target classes are variants of the same event, but the "
                    f"decision layer only asks the winner to beat the runner-up by "
                    f"{margin:.2f}. Checkpoint selection uses that same margin, so a model "
                    "that is confidently undecided between two glasses scores as well as one "
                    "that actually separates them — which is exactly the model you do not "
                    "want. Raising the margin makes the selected checkpoint the one that "
                    "commits."
                ),
                "suggest": {"margin_to_second_best": 0.20},
            })
        if frozen_encoder:
            out.append({
                "severity": "warn", "code": "frozen_encoder_cannot_separate_siblings",
                "message": (
                    "The encoder is frozen (head only), so this run can only draw a linear "
                    "boundary in the embedding the parent already learned. That is enough to "
                    "detect a clink, but the parent was never asked to keep two glass types "
                    "apart, so the distinction may simply not be present in its embedding. "
                    "Give the last stages room to adapt."
                ),
                "suggest": {"fine_tune_policy": "last_two"},
            })
        elif trainable_stages is not None and trainable_stages <= 1:
            out.append({
                "severity": "info", "code": "shallow_depth_for_siblings",
                "message": (
                    "Only the final encoder stage adapts. The last stage carries the most "
                    "abstract features, and separating variants of one sound needs the "
                    "mid-level filters that resolve partial structure to move too. If the "
                    "run plateaus, widen the depth before touching the learning rate."
                ),
            })

    # --- what the head starts from -----------------------------------------
    if from_parent:
        head_init = str(getattr(training, "transfer_head_init", "matching_classes"))
        cold = _cold_head_classes(cfg) if head_init != "random" else []
        if cold:
            out.append({
                "severity": "warn", "code": "finetune_positive_rows_cold",
                "message": (
                    f"{len(cold)} classifier row(s) start random on a transfer run: "
                    + ", ".join(cold) + ". The encoder transfers whole, but a random row "
                    "has to be re-derived from this dataset at the fine-tune learning rate — "
                    "a budget sized for a warm head. When the parent already models the same "
                    "event under a different name (a general clink class versus your named "
                    "glasses), map those rows onto it in the Model tab; otherwise expect to "
                    "need a higher head LR or a deeper fine-tune."
                ),
                "detail": {"classes": cold},
            })
        if head_init == "random":
            out.append({
                "severity": "info", "code": "head_discarded_on_transfer",
                "message": (
                    "transfer_head_init = random throws away the parent's classifier even "
                    "for classes both runs share. Those rows were fitted against the exact "
                    "embedding this run starts from, so re-deriving them costs epochs the "
                    "budget was not sized for. Keep 'matching_classes' unless you are "
                    "deliberately measuring a cold head."
                ),
                "suggest": {"transfer_head_init": "matching_classes"},
            })

    # --- does this data still fit the frozen frontend scale? ----------------
    # A parent fine-tune inherits its parent's fitted constants on purpose: they
    # are part of the encoder's numeric contract. That makes the useful question
    # the other one — are these clips still inside that scale?
    if drift and str(drift.get("status")) in ("warn", "error"):
        if str(drift.get("method")) == "zscore":
            detail = (
                f"the worst band's mean sits {drift.get('max_mean_shift_sigma')} standard "
                f"deviations from the fitted one (band {drift.get('worst_band')}, spread ratio "
                f"{drift.get('worst_band_std_ratio')})"
            )
        else:
            detail = (
                f"{float(drift.get('clipped_fraction') or 0.0):.1%} of values fall outside "
                "the fitted range and are clipped"
            )
        out.append({
            "severity": "warn" if str(drift.get("status")) == "warn" else "error",
            "code": "scaling_drifted_from_this_dataset",
            "message": (
                f"These clips do not sit inside the active feature scaling — {detail}. "
                + (
                    "The scaling is inherited from the parent run and must stay that way "
                    "(refitting it would change the input distribution the transferred "
                    "encoder expects), so the fix is on the audio side: check gain "
                    "staging, microphone and distance against the material the parent was "
                    "fitted on."
                    if from_parent else
                    "Re-estimate the statistics from the current training split in the "
                    "Audio Features tab."
                )
            ),
            "detail": dict(drift),
        })

    # --- BatchNorm: what the exported model will normalize with -------------
    freeze_bn = bool(getattr(training, "freeze_batchnorm", False))
    bn_recal = bool(getattr(training, "bn_recalibrate", False))
    mixup = float(getattr(training, "mixup_alpha", 0.0) or 0.0)
    if freeze_bn and not bn_recal:
        out.append({
            "severity": "warn", "code": "frozen_bn_without_recalibration",
            "message": (
                "BatchNorm is frozen, so the exported model will normalize with running "
                "statistics measured on the parent's data while the weights above them "
                "moved to this venue. Enable the post-fit BatchNorm refresh so the "
                "deployed model normalizes with constants measured on your clips — it "
                "changes saved statistics only, never learned weights."
            ),
            "suggest": {"bn_recalibrate": True},
        })
    elif mixup > 0 and not bn_recal:
        out.append({
            "severity": "info", "code": "mixup_without_bn_recalibration",
            "message": (
                f"mixup_alpha = {mixup:g} trains BatchNorm on convex blends of two clips, "
                "which never occur at inference. Refresh the running statistics on real "
                "clips after the fit so the deployed thresholds mean what they measured."
            ),
            "suggest": {"bn_recalibrate": True},
        })
    if not freeze_bn and transfer and rows and rows < SMALL_TRANSFER_ROWS:
        out.append({
            "severity": "info", "code": "bn_relearned_from_a_thin_sample",
            "message": (
                f"BatchNorm will re-estimate its running statistics from {rows} clips. On a "
                "set this small — usually one room, one session — those estimates describe "
                "the recording condition more than the sound, and they move every epoch, "
                "which is a common reason validation loss looks noisy and slow. Freezing "
                "them keeps the parent's estimate and makes the curve readable."
            ),
            "suggest": {"freeze_batchnorm": True},
        })

    # --- masking vs. the evidence ------------------------------------------
    if bool(getattr(training, "specaugment_enabled", False)):
        bands = int(getattr(features, "n_mels", 0) or 0)
        freq_mask = int(getattr(training, "specaugment_freq_mask", 0) or 0)
        if bands > 0 and freq_mask > bands * SPECAUG_FREQ_MASK_WARN_SHARE:
            out.append({
                "severity": "warn", "code": "specaugment_masks_the_partials",
                "message": (
                    f"SpecAugment can blank {freq_mask} of {bands} feature rows "
                    f"({freq_mask / bands:.0%}) in one mask. Frequency masking teaches "
                    "invariance to the band that is missing — which is the right trade for "
                    "general tagging and the wrong one here, because the narrow partial "
                    "comb IS the glass identity. Keep the mask under a tenth of the rows "
                    "when target classes differ only in timbre."
                ),
                "suggest": {"specaugment_freq_mask": max(1, int(bands * 0.06))},
            })
    if fine_grained and mixup > 0:
        out.append({
            "severity": "info", "code": "mixup_blurs_sibling_classes",
            "message": (
                "mixup blends two clips and their labels. When the classes are variants of "
                "one sound, a 50/50 blend of two glasses is not a meaningful training "
                "example — it teaches the model that the boundary between them is soft. "
                "Mixup earns its place on a large pretraining corpus; turn it off for the "
                "fine-grained fine-tune."
            ),
            "suggest": {"mixup_alpha": 0.0},
        })

    # --- schedule coherence -------------------------------------------------
    epochs = int(getattr(training, "epochs", 0) or 0)
    if plan.warmup_epochs and epochs and plan.warmup_epochs >= epochs * 0.4:
        out.append({
            "severity": "warn", "code": "head_warmup_eats_the_budget",
            "message": (
                f"{plan.warmup_epochs} of {epochs} epochs run with the encoder frozen, so "
                "less than two thirds of the budget actually adapts it. Shorten the "
                "head warm-up or raise the epoch budget."
            ),
            "suggest": {"fine_tune_warmup_epochs": 1},
        })
    lr_warmup = int(getattr(training, "warmup_epochs", 0) or 0)
    patience = int(getattr(training, "patience", 0) or 0)
    if lr_warmup and patience and lr_warmup >= patience:
        out.append({
            "severity": "warn", "code": "lr_warmup_outlasts_patience",
            "message": (
                f"The LR ramp takes {lr_warmup} epochs while early stopping gives up after "
                f"{patience} without improvement. The run can be stopped before it has "
                "reached full learning rate. Keep the ramp well under the patience."
            ),
            "suggest": {"warmup_epochs": max(1, patience // 3)},
        })
    if transfer and float(getattr(training, "lr_head", 0.0) or 0.0) > 0:
        ratio = float(training.lr_head) / max(float(training.lr_backbone), 1e-12)
        if ratio >= 20:
            out.append({
                "severity": "info", "code": "head_outruns_the_encoder",
                "message": (
                    f"The head learns {ratio:.0f}x faster than the encoder. A fresh "
                    "classifier that fast fits the training set on the parent's unchanged "
                    "features within a few epochs, early stopping then fires, and the "
                    "encoder never adapts — the run looks converged while nothing "
                    "transferred. Keep the head within about 5x the encoder LR."
                ),
                "suggest": {"lr_head": float(training.lr_backbone) * 4},
            })

    # --- regularization for a small, frozen-encoder fit ---------------------
    head_dropout = float(getattr(training, "head_dropout", 0.0) or 0.0)
    if (
        frozen_encoder
        and rows
        and rows < SMALL_TRANSFER_ROWS
        and head_dropout < 0.3
        and not bool(getattr(training, "head_dropout_auto", False))
    ):
        out.append({
            "severity": "info", "code": "thin_probe_is_underregularized",
            "message": (
                f"A frozen encoder on {rows} clips means one Linear layer over a wide "
                "embedding is the entire fitted model, and classifier dropout is its only "
                "regularizer. 0.3-0.4 is the usual range for a probe this thin."
            ),
            "suggest": {"head_dropout": 0.3},
        })

    out.sort(key=lambda f: SEVERITY_ORDER.index(f["severity"]))
    return out


def config_advice(cfg, dataset: dict | None = None) -> dict:
    """All advisory findings for a project config, grouped by area."""
    sample_rate = int(getattr(cfg.model, "sample_rate", 0) or 0)
    return {
        "stage": pipeline_stage(cfg),
        "features": feature_advice(cfg.features, sample_rate),
        "model": model_advice(cfg.model, cfg.features, sample_rate),
        "training": training_advice(cfg, dataset),
    }
