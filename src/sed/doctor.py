"""Pre-flight checklists. Each `doctor_*` function returns a list of
`Check` results; the CLI prints them as a colored table.

Intent: catch problems BEFORE the expensive step. A failed pre-record check
saves an hour of bad recordings; a failed pre-train check saves an hour of
GPU time; a failed pre-deploy check saves a service that crashes at 11 PM.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sed.config import Config, ProjectLayout

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str               # "PASS" | "WARN" | "FAIL"
    detail: str = ""
    fix: str = ""             # actionable hint when status != PASS

    @property
    def emoji(self) -> str:
        return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(self.status, "?")


def _ok(name: str, detail: str = "") -> Check:
    return Check(name, "PASS", detail)


def _warn(name: str, detail: str, fix: str = "") -> Check:
    return Check(name, "WARN", detail, fix)


def _fail(name: str, detail: str, fix: str = "") -> Check:
    return Check(name, "FAIL", detail, fix)


# ---------------------------------------------------------------------------
# 1. Pre-record
# ---------------------------------------------------------------------------


def doctor_record(cfg: Config, seconds: float = 5.0) -> list[Check]:
    """Verify the audio interface BEFORE you spend an hour recording.

    Tests:
      - the configured device exists and reports the expected channel count
      - sample-rate is supported
      - all channels actually carry signal (catches dead/unplugged mic)
      - per-channel RMS / peak in healthy ranges
      - no clipping
      - latency / xrun count over a short capture window
    """
    out: list[Check] = []

    # --- (a) device discovery ---
    try:
        from sed.audio.capture import list_input_devices, resolve_device
        devs = list_input_devices()
        if not devs:
            out.append(_fail("audio devices",
                             "no input devices found",
                             "plug the audio interface in; "
                             "check `arecord -l` (Linux) or your OS sound settings"))
            return out
        out.append(_ok("audio devices", f"{len(devs)} input device(s) available"))

        try:
            idx = resolve_device(cfg.audio.record_device)
        except ValueError as e:
            out.append(_fail("device resolve",
                             str(e),
                             "check `[audio] record_device` in your project.toml — "
                             f"available devices: {', '.join(d.name for d in devs[:5])}"))
            return out

        chosen = next((d for d in devs if d.index == idx), None)
        if chosen is None:
            out.append(_fail("device resolve", f"index {idx} not in input list"))
            return out
        out.append(_ok("device selected", f"[{chosen.index}] {chosen.name} "
                                          f"({chosen.max_input_channels}ch max)"))
        if cfg.audio.record_device == "default":
            out.append(_warn(
                "device pinned",
                "`record_device` is `default`; OS default inputs can change",
                "set `[audio] record_device` to a stable device name substring "
                "from `clinkd audio devices`",
            ))

        # Channel count sanity
        if cfg.audio.record_channels > chosen.max_input_channels:
            out.append(_fail(
                "channel count",
                f"asked for {cfg.audio.record_channels} but device has "
                f"{chosen.max_input_channels}",
                f"set [audio] record_channels ≤ {chosen.max_input_channels}",
            ))
            return out
        else:
            out.append(_ok("channel count",
                           f"requested={cfg.audio.record_channels}, "
                           f"available={chosen.max_input_channels}"))
        out.append(_ok(
            "sample rates",
            f"input={cfg.audio.input_sample_rate} Hz -> features={cfg.audio.sample_rate} Hz",
        ))
    except ImportError as e:
        out.append(_fail("portaudio",
                         f"sounddevice import failed: {e}",
                         "install libasound2 + portaudio: "
                         "`apt install libportaudio2 libasound2`"))
        return out

    # --- (b) capture a few seconds and analyse ---
    import sounddevice as sd

    try:
        rec = sd.rec(
            int(seconds * cfg.audio.input_sample_rate),
            samplerate=cfg.audio.input_sample_rate,
            channels=cfg.audio.record_channels,
            dtype="float32",
            device=idx,
        )
        sd.wait()
    except Exception as e:  # noqa: BLE001
        out.append(_fail("capture",
                         f"sd.rec failed: {e}",
                         "check sample rate support: try 44100 or 48000 instead"))
        return out

    out.append(_ok("capture", f"{seconds:.0f}s recorded successfully"))

    # --- (c) per-channel signal analysis ---
    for c in range(cfg.audio.record_channels):
        x = rec[:, c]
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        peak = float(np.max(np.abs(x)))
        rms_db = 20 * np.log10(rms + 1e-12)
        peak_db = 20 * np.log10(peak + 1e-12)

        # Heuristics tuned for room-mic recording
        if peak < 1e-3:
            out.append(_fail(
                f"mic ch{c}",
                f"essentially silent (peak {peak_db:.0f} dBFS)",
                "mic unplugged / muted / phantom power off / wrong input channel",
            ))
        elif peak > 0.99:
            out.append(_fail(
                f"mic ch{c}",
                f"clipping (peak {peak_db:.1f} dBFS)",
                "lower preamp gain on the interface; aim for peaks at -10 to -3 dBFS",
            ))
        elif peak > 0.7:
            out.append(_warn(
                f"mic ch{c}",
                f"peaks high (peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS)",
                "consider lowering preamp gain by 3-6 dB to leave headroom",
            ))
        elif peak < 0.02:
            out.append(_warn(
                f"mic ch{c}",
                f"very quiet (peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS)",
                "raise preamp gain by 6-12 dB; aim for typical-source peaks at -20 to -10 dBFS",
            ))
        else:
            out.append(_ok(
                f"mic ch{c}",
                f"peak {peak_db:.1f} dBFS · RMS {rms_db:.1f} dBFS — healthy",
            ))

    # --- (d) DC offset ---
    for c in range(cfg.audio.record_channels):
        dc = float(rec[:, c].mean())
        if abs(dc) > 0.005:
            out.append(_warn(
                f"DC offset ch{c}",
                f"DC = {dc:+.4f} (≈{20*np.log10(abs(dc)+1e-12):.0f} dBFS)",
                "interface or DAC issue; usually harmless, but if persistent, "
                "enable a high-pass filter on your interface (typ. 80 Hz)",
            ))

    # --- (e) inter-channel correlation: catches "all 4 channels are actually 1 mic" ---
    if cfg.audio.record_channels >= 2:
        for c2 in range(1, cfg.audio.record_channels):
            corr = float(np.corrcoef(rec[:, 0], rec[:, c2])[0, 1])
            if abs(corr) > 0.98:
                out.append(_warn(
                    f"corr ch0↔ch{c2}",
                    f"channels appear identical (r={corr:.3f})",
                    "either the same mic is wired to multiple inputs, "
                    "or the device is summing channels — check interface routing",
                ))
            else:
                out.append(_ok(
                    f"corr ch0↔ch{c2}",
                    f"r={corr:+.3f} — independent signals",
                ))

    return out


# ---------------------------------------------------------------------------
# 2. Pre-train
# ---------------------------------------------------------------------------


def doctor_train(cfg: Config, project_id: str) -> list[Check]:
    """Sanity-check the dataset and training config BEFORE training."""
    out: list[Check] = []
    layout = ProjectLayout(project_id)

    if not layout.dataset_dir.exists():
        out.append(_fail("dataset", f"no dataset/ at {layout.dataset_dir}",
                         "run `clinkd dataset compose --project " + project_id + "`"))
        return out

    # --- (a) split files exist and parse ---
    import csv as csvlib
    splits: dict[str, list[dict]] = {}
    for name in ("train", "val", "test"):
        p = layout.dataset_dir / f"{name}.csv"
        if not p.exists():
            out.append(_fail(f"{name}.csv", f"missing {p}",
                             "run `clinkd dataset compose`"))
            return out
        with open(p, "r", encoding="utf-8") as f:
            splits[name] = list(csvlib.DictReader(f))
    out.append(_ok("split files", f"train={len(splits['train'])} "
                                  f"val={len(splits['val'])} "
                                  f"test={len(splits['test'])}"))

    # --- (b) classes.json + agreement with config ---
    classes_path = layout.dataset_dir / "classes.json"
    if not classes_path.exists():
        out.append(_fail("classes.json", "missing", "run `clinkd dataset compose`"))
        return out
    with open(classes_path, "r", encoding="utf-8") as f:
        ds_classes = json.load(f)["classes"]

    if ds_classes != cfg.model.classes:
        out.append(_warn(
            "class agreement",
            f"dataset has {ds_classes} but project.toml says {cfg.model.classes}",
            "re-run `clinkd dataset compose` after editing project.toml",
        ))
    else:
        out.append(_ok("class agreement", f"{len(ds_classes)} classes"))

    # --- (c) per-class counts (warn if any class has < 50 in train) ---
    train_counts: dict[str, int] = {c: 0 for c in ds_classes}
    for r in splits["train"]:
        train_counts[r["class_name"]] = train_counts.get(r["class_name"], 0) + 1
    for cls, n in train_counts.items():
        if n == 0:
            # Declared in [model.classes] but no recordings yet → training will
            # fail because the classifier head expects N classes. Either record
            # some clips for this class, or drop it from project.toml until you do.
            out.append(_fail(
                f"count {cls}",
                "no training samples (class is declared but empty)",
                f"record with `clinkd record positives --class {cls}` "
                f"OR remove '{cls}' from [model.classes] in project.toml "
                "until you're ready",
            ))
        elif n < 50:
            out.append(_warn(f"count {cls}", f"{n} train samples",
                             "aim for ≥200; recall will likely suffer"))
        else:
            out.append(_ok(f"count {cls}", f"{n} train samples"))

    # --- (d) val/test must contain every class for meaningful metrics ---
    for split_name in ("val", "test"):
        present = {r["class_name"] for r in splits[split_name]}
        missing = set(ds_classes) - present
        if missing:
            out.append(_warn(
                f"{split_name} coverage",
                f"missing classes: {sorted(missing)}",
                "record more sessions per class so the splitter can distribute them",
            ))
        else:
            out.append(_ok(f"{split_name} coverage", "all classes present"))

    # --- (e) negative ratio sanity ---
    neg = train_counts.get("_negative", 0)
    pos = sum(n for c, n in train_counts.items() if c != "_negative")
    if pos == 0:
        out.append(_warn(
            "pos:neg ratio",
            "no positives in train",
            "early-state OK while you bootstrap; record positives with "
            "`clinkd record positives` before training",
        ))
    else:
        ratio = neg / max(pos, 1)
        if ratio < 1.0:
            out.append(_warn(
                "pos:neg ratio",
                f"only {ratio:.1f}x more negatives than positives",
                "aim for 3-5x; record more ambience and hard-negatives "
                "to cut FP rate in production",
            ))
        else:
            out.append(_ok("pos:neg ratio",
                           f"{ratio:.1f}x more negatives than positives"))

    # --- (f) ambience pool for domain mixing ---
    # An empty/thin pool is NOT fatal — training succeeds; the augmenter just
    # skips the domain-mix step ([augmentation] domain_mix_prob is also tunable
    # in the project config). It does reduce noise robustness, which is why we
    # still surface this as a WARN.
    from sed.data.augment import AudioPool, resolve_pool_path
    from sed.data.library import Library
    amb_path = resolve_pool_path(cfg.augmentation.ambience_pool)
    pool = AudioPool(amb_path, cfg.audio.sample_rate)
    if len(pool) == 0:
        out.append(_warn(
            "ambience pool",
            f"empty: {amb_path}",
            "training will work, but with no domain mixing the model is less "
            "robust to real venue noise. Record some with "
            "`clinkd library record-ambience --label <source>` when convenient.",
        ))
    else:
        # Library-level metrics (more reliable than counting raw files because
        # the library tracks duration in its manifest)
        try:
            lib = Library()
            amb_clips = lib.query(category="ambience")
            total_min = sum(c.duration_s for c in amb_clips) / 60
            distinct_labels = len({c.label for c in amb_clips})
            densities = {
                t.split(":", 1)[1] for c in amb_clips for t in c.tags
                if t.startswith("density:")
            }
            has_busy = bool(densities & {"busy", "peak"})
        except Exception:  # noqa: BLE001
            total_min, distinct_labels, has_busy = 0.0, 0, False

        # Soft warning only — not a blocker. Domain mixing simply reuses
        # the same ambience often when the pool is thin.
        if total_min and total_min < 5:
            out.append(_warn(
                "ambience pool duration",
                f"only {total_min:.1f} min — thin pool, augmentation reuses clips",
                "record at least 30 min across 3+ sources with "
                "`clinkd library record-ambience` for stronger noise robustness",
            ))
        elif total_min < 20:
            out.append(_warn(
                "ambience pool duration",
                f"{total_min:.1f} min (recommend ≥ 20)",
                "augmenter will reuse the same ambience often, reducing "
                "effective variety; record more or `clinkd library import`",
            ))
        elif total_min > 0:
            out.append(_ok("ambience pool duration", f"{total_min:.1f} min"))

        if total_min > 0 and distinct_labels < 3:
            out.append(_warn(
                "ambience pool diversity",
                f"only {distinct_labels} distinct source label(s)",
                "aim for 3+ different sources (different bars/restaurants/"
                "rooms). Diversity matters more than total minutes.",
            ))
        elif distinct_labels >= 3:
            out.append(_ok("ambience pool diversity",
                           f"{distinct_labels} distinct sources"))

        if total_min > 0 and not has_busy:
            out.append(_warn(
                "ambience pool density",
                "no clips tagged density:busy or density:peak",
                "the model won't see realistic peak-hour conditions during "
                "training. Record library ambience during a busy Friday "
                "night with --density busy",
            ))

        # Always show the file count as info even when fine
        if len(pool) < 10:
            out.append(_warn(
                "ambience pool clip count",
                f"only {len(pool)} files",
                "more variety = better generalisation; aim for 30+ clips",
            ))
        else:
            out.append(_ok("ambience pool clip count", f"{len(pool)} files"))

    # --- (g) hyperparameter sanity ---
    if cfg.training.balanced_sampler:
        out.append(_warn(
            "balanced_sampler",
            "enabled — this will inflate val F1 but hurt deployed FP rate",
            "set [training] balanced_sampler = false",
        ))
    else:
        out.append(_ok("balanced_sampler",
                       "disabled (preserves natural class distribution)"))

    if cfg.training.focal_gamma_neg < cfg.training.focal_gamma_pos:
        out.append(_warn(
            "focal asymmetry",
            f"γ_neg={cfg.training.focal_gamma_neg} < γ_pos={cfg.training.focal_gamma_pos}",
            "for FP minimisation γ_neg should be ≥ γ_pos (typical 2-4 vs 1)",
        ))
    else:
        out.append(_ok("focal asymmetry",
                       f"γ_neg={cfg.training.focal_gamma_neg} γ_pos={cfg.training.focal_gamma_pos}"))

    # --- (h) session leakage spot-check: any session_id in train AND val? ---
    train_sess = {r["session_id"] for r in splits["train"]}
    val_sess   = {r["session_id"] for r in splits["val"]}
    test_sess  = {r["session_id"] for r in splits["test"]}
    overlap    = (train_sess & val_sess) | (train_sess & test_sess) | (val_sess & test_sess)
    if overlap:
        out.append(_fail(
            "session leakage",
            f"sessions appear in multiple splits: {sorted(overlap)[:5]}...",
            "this inflates val metrics — fix the splitter or recompose",
        ))
    else:
        out.append(_ok("session leakage", "no session in two splits"))

    return out


# ---------------------------------------------------------------------------
# 3. Pre-deploy
# ---------------------------------------------------------------------------


def doctor_deploy(cfg: Config, project_id: str) -> list[Check]:
    """Final checks before leaving the system running 24/7."""
    out: list[Check] = []
    layout = ProjectLayout(project_id)

    # --- (a) model artifacts ---
    manifest_path = layout.models_dir / "manifest.json"
    if not manifest_path.exists():
        out.append(_fail("manifest", f"no manifest.json at {manifest_path}",
                         "run `clinkd export --project " + project_id + "`"))
        return out
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out.append(_ok("manifest", f"classes={manifest['classes']}"))

    int8 = layout.models_dir / manifest.get("model_int8", "")
    fp32 = layout.models_dir / manifest.get("model_fp32", "")
    if not int8.exists() and not fp32.exists():
        out.append(_fail("model files", "no .xml found", "run `sed export`"))
        return out
    out.append(_ok("model files",
                   f"INT8={'✓' if int8.exists() else '✗'} "
                   f"FP32={'✓' if fp32.exists() else '✗'}"))

    # --- (b) thresholds calibrated? ---
    calibrated = [c for c in manifest["classes"] if c != "_negative" and c in cfg.thresholds]
    needed = [c for c in manifest["classes"] if c != "_negative"]
    if not calibrated:
        out.append(_warn(
            "thresholds",
            "no per-class thresholds set; using defaults",
            "run `clinkd calibrate --project " + project_id + "` against ambience",
        ))
    elif set(calibrated) < set(needed):
        out.append(_warn(
            "thresholds",
            f"some classes uncalibrated: {set(needed) - set(calibrated)}",
            "run `sed calibrate` again",
        ))
    else:
        out.append(_ok("thresholds",
                       " ".join(f"{c}={cfg.thresholds[c]:.2f}" for c in calibrated)))

    # --- (c) load model & measure latency ---
    try:
        import openvino as ov
        from sed.runtime.infer import resolve_model_path
        core = ov.Core()
        cm = core.compile_model(
            core.read_model(str(resolve_model_path(layout, manifest))),
            "CPU",
            config={"PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": str(cfg.inference.inference_threads)},
        )
        req = cm.create_infer_request()
        n_mels = manifest["n_mels"]
        n_frames = int(round(
            manifest["window_seconds"] * manifest["sample_rate"] / manifest["hop_length"]
        )) + 1
        # batch=record_channels for the realistic multi-mic case
        b = max(1, cfg.audio.record_channels) if cfg.inference.multi_mic_mode != "single" else 1
        x = np.random.randn(b, 1, n_mels, n_frames).astype(np.float32)
        # warm
        for _ in range(10):
            req.infer({0: x})
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            req.infer({0: x})
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        p50, p99 = times[25], times[49]
        budget_ms = cfg.inference.hop_ms
        if p99 > budget_ms:
            out.append(_warn(
                "inference latency",
                f"p50={p50:.1f}ms p99={p99:.1f}ms (batch={b}) — exceeds "
                f"hop {budget_ms}ms",
                f"reduce inference_threads, switch to mn01_as backbone, "
                f"or increase [inference] hop_ms above {p99:.0f}",
            ))
        else:
            out.append(_ok(
                "inference latency",
                f"p50={p50:.1f}ms p99={p99:.1f}ms (batch={b}, budget {budget_ms}ms)",
            ))
    except Exception as e:  # noqa: BLE001
        out.append(_fail("model load", str(e),
                         "check OpenVINO install and that the model exported cleanly"))

    # --- (d) ArtNet target reachable? ---
    if cfg.artnet.enabled:
        target = cfg.artnet.target_ip
        try:
            socket.inet_aton(target)
            if target.endswith(".255") or target == "255.255.255.255":
                out.append(_ok("artnet target", f"{target} (broadcast)"))
            else:
                # quick UDP probe to ArtNet port (we don't expect a reply)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.2)
                try:
                    s.sendto(b"\x41\x72\x74\x2d\x4e\x65\x74\x00", (target, 6454))
                    out.append(_ok("artnet target", f"{target}:6454 reachable"))
                except OSError as e:
                    out.append(_warn("artnet target",
                                     f"{target}:6454 unreachable: {e}",
                                     "check the network cable and switch; "
                                     "broadcast (255.255.255.255) avoids this"))
                finally:
                    s.close()
        except OSError:
            out.append(_fail("artnet target",
                             f"invalid IP: {target}",
                             "set [artnet] target_ip to a valid IPv4 address"))
    else:
        out.append(_ok("artnet", "disabled in config"))

    # --- (e) systemd unit ready? ---
    unit = Path("/etc/systemd/system/sed.service")
    if unit.exists():
        out.append(_ok("systemd unit", f"{unit} present"))
    else:
        out.append(_warn("systemd unit", "/etc/systemd/system/sed.service missing",
                         "run `sudo PROJECT_ID=" + project_id + " ./deploy/install.sh`"))

    # --- (f) writable output paths ---
    for p in (layout.reports_dir, layout.models_dir):
        try:
            test_file = p / ".doctor_write_test"
            test_file.touch()
            test_file.unlink()
            out.append(_ok(f"writable {p.name}", str(p)))
        except OSError as e:
            out.append(_fail(f"writable {p.name}", str(e),
                             f"chown -R sed:sed {p}"))

    return out


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def render(checks: list[Check], title: str) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title=title, header_style="bold")
    table.add_column("", width=2)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail")
    for c in checks:
        style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[c.status]
        table.add_row(c.emoji, c.name, f"[{style}]{c.status}[/{style}]", c.detail)
    console.print(table)
    # Show fix advice for any non-PASS
    bad = [c for c in checks if c.status != "PASS"]
    if bad:
        console.print("\n[bold]Suggested fixes:[/bold]")
        for c in bad:
            if c.fix:
                console.print(f"  [{('red' if c.status == 'FAIL' else 'yellow')}]"
                              f"• {c.name}[/]: {c.fix}")
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    if fail_count:
        console.print(f"\n[red bold]{fail_count} FAIL[/red bold] · "
                      f"[yellow]{warn_count} WARN[/yellow] — "
                      f"address FAILs before proceeding.")
    elif warn_count:
        console.print(f"\n[yellow bold]{warn_count} WARN[/yellow bold] — "
                      f"safe to proceed but check the warnings.")
    else:
        console.print(f"\n[green bold]all {len(checks)} checks passed[/green bold]")


def all_failed(checks: list[Check]) -> bool:
    return any(c.status == "FAIL" for c in checks)
