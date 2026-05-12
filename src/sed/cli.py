"""Top-level Typer CLI.

    sed --help

Subcommand groups:

    audio       device discovery, simple capture probe
    library     manage the shared sound library (across projects)
    project     create / list / inspect projects
    record      record clips into a project (auto-trigger or continuous)
    inspect     generate the dataset HTML report
    quarantine  flag bad samples so they're skipped at composition time
    dataset     compose train/val/test splits
    train       fine-tune the model
    export      export to ONNX + OpenVINO INT8
    calibrate   sweep thresholds against ambience to hit a target FP/h
    run         start the 24/7 inference loop
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Sound event detection for interactive lighting.",
)
audio_app = typer.Typer(no_args_is_help=True, help="Audio device + probe utilities.")
library_app = typer.Typer(no_args_is_help=True, help="Manage the shared sound library.")
project_app = typer.Typer(no_args_is_help=True, help="Project / venue management.")
record_app = typer.Typer(no_args_is_help=True, help="Record audio for a project.")
dataset_app = typer.Typer(no_args_is_help=True, help="Compose / preview a dataset.")
app.add_typer(audio_app, name="audio")
app.add_typer(library_app, name="library")
app.add_typer(project_app, name="project")
app.add_typer(record_app, name="record")
app.add_typer(dataset_app, name="dataset")

check_app = typer.Typer(no_args_is_help=True,
                        help="Pre-flight check suites for each phase: check record, "
                             "check train, check deploy, check all.")
app.add_typer(check_app, name="check")

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


@audio_app.command("devices")
def audio_devices() -> None:
    """List available input devices."""
    import sounddevice as sd

    from sed.audio.capture import list_input_devices

    default_idx = sd.default.device[0]
    table = Table(title="Audio input devices")
    table.add_column("idx")
    table.add_column("default")
    table.add_column("name")
    table.add_column("ch", justify="right")
    table.add_column("sr", justify="right")
    table.add_column("api")
    for d in list_input_devices():
        table.add_row(
            str(d.index),
            "yes" if d.index == default_idx else "",
            d.name,
            str(d.max_input_channels),
            f"{d.default_samplerate:.0f}",
            d.hostapi_name,
        )
    console.print(table)


@audio_app.command("probe")
def audio_probe(
    device: str = typer.Option("default", help="Device name substring or index."),
    seconds: float = typer.Option(3.0, help="How long to record."),
    channels: int = typer.Option(1, help="Channels to capture."),
    sample_rate: int = typer.Option(48000),
) -> None:
    """Record a few seconds and print RMS / peak per channel — sanity check."""
    import numpy as np
    import sounddevice as sd

    from sed.audio.capture import resolve_device

    _setup_logging()
    idx = resolve_device(device)
    console.print(f"recording {seconds:.1f}s on device {idx} ({channels}ch @ {sample_rate} Hz)")
    data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                  channels=channels, dtype="float32", device=idx)
    sd.wait()
    table = Table(title="probe results")
    table.add_column("ch")
    table.add_column("rms", justify="right")
    table.add_column("peak", justify="right")
    for c in range(channels):
        rms = float(np.sqrt(np.mean(data[:, c] ** 2) + 1e-12))
        peak = float(np.max(np.abs(data[:, c])))
        table.add_row(str(c), f"{rms:.4f}", f"{peak:.4f}")
    console.print(table)
    if np.max(np.abs(data)) < 1e-3:
        console.print("[red]Signal is essentially silent — wrong device or muted mic.[/red]")


def _audio_meter_table(levels: list[dict[str, float]], title: str) -> Table:
    table = Table(title=title)
    table.add_column("ch", justify="right")
    table.add_column("peak", justify="right")
    table.add_column("rms", justify="right")
    table.add_column("level")
    for i, lvl in enumerate(levels):
        peak_db = lvl["peak_db"]
        rms_db = lvl["rms_db"]
        filled = max(0, min(32, int((peak_db + 72.0) / 72.0 * 32)))
        bar = "█" * filled + "░" * (32 - filled)
        if lvl["peak"] >= 0.99:
            style = "red"
        elif lvl["peak"] < 1e-3:
            style = "dim red"
        elif lvl["peak"] < 0.02:
            style = "yellow"
        else:
            style = "green"
        table.add_row(
            str(i),
            f"{peak_db:6.1f} dBFS",
            f"{rms_db:6.1f} dBFS",
            f"[{style}]{bar}[/{style}]",
        )
    return table


@audio_app.command("meter")
def audio_meter(
    project: str | None = typer.Option(None, help="Use this project's audio config."),
    device: str | None = typer.Option(None, help="Override device name substring or index."),
    channels: int | None = typer.Option(None, help="Override channel count."),
    sample_rate: int | None = typer.Option(None, help="Override sample rate."),
    seconds: float = typer.Option(0.25, help="Meter refresh window."),
) -> None:
    """Live per-channel input meter using the same config as recording/inference."""
    import time

    import numpy as np
    import sounddevice as sd

    from sed.audio.capture import resolve_device
    from sed.config import load_config

    _setup_logging("WARNING")
    cfg = load_config(project) if project else load_config()
    dev = device if device is not None else cfg.audio.record_device
    ch = channels if channels is not None else cfg.audio.record_channels
    sr = sample_rate if sample_rate is not None else cfg.audio.input_sample_rate
    idx = resolve_device(dev)
    info = sd.query_devices(idx)
    title = f"Audio meter — [{idx}] {info['name']} · {ch}ch @ {sr} Hz · Ctrl+C to stop"

    if ch > int(info["max_input_channels"]):
        raise typer.BadParameter(
            f"requested {ch} channel(s), but device has {info['max_input_channels']}"
        )

    n = max(256, int(seconds * sr))
    levels = [
        {"peak": 0.0, "rms": 0.0, "peak_db": -120.0, "rms_db": -120.0}
        for _ in range(ch)
    ]

    def callback(indata, frames, time_info, status):
        nonlocal levels
        if status:
            console.print(f"[yellow]{status}[/yellow]")
        block = indata.astype("float32", copy=False)
        next_levels = []
        for c in range(ch):
            x = block[:, c]
            peak = float(np.max(np.abs(x)))
            rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
            next_levels.append(
                {
                    "peak": peak,
                    "rms": rms,
                    "peak_db": 20 * np.log10(peak + 1e-12),
                    "rms_db": 20 * np.log10(rms + 1e-12),
                }
            )
        levels = next_levels

    try:
        with sd.InputStream(
            device=idx,
            channels=ch,
            samplerate=sr,
            blocksize=n,
            dtype="float32",
            callback=callback,
        ), Live(
            _audio_meter_table(levels, title),
            console=console,
            refresh_per_second=8,
        ) as live:
            while True:
                live.update(_audio_meter_table(levels, title))
                time.sleep(0.125)
    except KeyboardInterrupt:
        console.print()


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


@project_app.command("new")
def project_new(
    project_id: str = typer.Argument(..., help="lowercase, no spaces, e.g. 'the_hawthorn'"),
    name: str = typer.Option("", help="Human-readable name."),
) -> None:
    """Scaffold a new project under projects/<id>/."""
    from datetime import datetime, timezone
    from sed.config import ProjectLayout
    layout = ProjectLayout(project_id)
    if layout.root.exists():
        console.print(f"[yellow]exists: {layout.root}[/yellow]")
        return
    layout.create()
    layout.project_toml.write_text(
        f"""# {project_id} — project-specific overrides
# Everything not listed here inherits from configs/default.toml

[project]
id         = "{project_id}"
name       = "{name or project_id}"
created_at = "{datetime.now(timezone.utc).date().isoformat()}"
notes      = ""

[model]
# First entry must stay "_negative". Add/remove target classes here.
classes = ["_negative", "clink_wine_thin", "clink_wine_thick"]

[targets]
# Keys must match entries in [model.classes] above.
clink_wine_thin  = {{ enabled = true, channel = 1, value = 85,  effect = "default" }}
clink_wine_thick = {{ enabled = true, channel = 1, value = 170, effect = "default" }}

[thresholds]
# Rewritten by `clinkd calibrate`. These are starting values only.
clink_wine_thin  = 0.600
clink_wine_thick = 0.600

[audio]
# Run `clinkd audio devices` to find your interface name or index.
record_device = "default"

# [onset]
# Override sensitivity here if this venue is unusually quiet or loud.
# See configs/default.toml [onset] for full documentation and tuning guide.
#
# mad_k    = 4.0    # raise (6-8) if too trigger-happy; lower (3) if missing clinks
# min_flux = 0.005  # raise if ambient noise fires it; lower if clinks aren't caught

# [augmentation]
# Override training augmentations here. See configs/default.toml [augmentation]
# for the full list of parameters and tuning guide. Set any prob/amount to 0 to
# disable that augmentation individually. Anything you do not set is inherited.
#
# domain_mix_prob       = 0.95       # 0 disables domain mixing entirely
# snr_db_range          = [-2.0, 25.0]
# per_class_min_snr_db  = {{ clink_wine_thin = 5.0 }}  # protect fragile classes
# external_negative_mix_prob = 0.20  # 0 disables library-negative mixing
# pitch_shift_prob      = 0.4        # 0 disables pitch shift
# time_stretch_prob     = 0.2        # 0 disables time stretch
# time_shift_ms         = 200        # 0 disables time shift
# gaussian_noise_amp    = [0.001, 0.01]   # [0, 0] disables gaussian noise
# specaug_freq_masks    = 2          # 0 disables SpecAugment freq masks
# specaug_time_masks    = 2          # 0 disables SpecAugment time masks

[artnet]
target_ip = "192.168.50.40"

# [dataset.library]
# Uncomment to pin specific library clips to this project instead of auto-discovery.
# clink_wine_thin = [{{ category = "glasses_clean", label = "wine_thin" }}]
# "_negative"     = [{{ category = "negatives_external", label = "cutlery" }}]
""",
        encoding="utf-8",
    )
    console.print(f"[green]created {layout.root}[/green]")
    console.print(f"  edit {layout.project_toml} to define your classes and targets")


@project_app.command("list")
def project_list() -> None:
    """List all projects."""
    from sed.config import list_projects
    for p in list_projects():
        console.print(f"- {p}")


@project_app.command("info")
def project_info(project_id: str) -> None:
    """Show project status: clip counts, model status."""
    from sed.config import ProjectLayout, load_config
    layout = ProjectLayout(project_id)
    cfg = load_config(project_id)
    console.print(f"[bold]{project_id}[/bold] @ {layout.root}")
    console.print(f"classes: {cfg.model.classes}")
    if not layout.root.exists():
        console.print("[red]project does not exist[/red]")
        return
    table = Table(title="recordings")
    table.add_column("kind"); table.add_column("class"); table.add_column("count", justify="right")
    pos_root = layout.recordings_dir / "positives"
    if pos_root.exists():
        for d in sorted(pos_root.iterdir()):
            if d.is_dir():
                table.add_row("positive", d.name, str(len(list(d.glob("*.wav")))))
    hn_root = layout.recordings_dir / "hardneg"
    if hn_root.exists():
        for d in sorted(hn_root.iterdir()):
            if d.is_dir():
                table.add_row("hardneg", d.name, str(len(list(d.glob("*.wav")))))
    if layout.ambience_dir.exists():
        table.add_row("ambience", "-", str(len(list(layout.ambience_dir.glob("*.wav")))))
    console.print(table)
    ckpt = layout.models_dir / "current.ckpt"
    ov_link = layout.models_dir / "current.openvino"
    console.print(f"checkpoint: {'✅' if ckpt.exists() else '❌'} {ckpt}")
    console.print(f"openvino:   {'✅' if ov_link.exists() or ov_link.is_symlink() else '❌'} {ov_link}")


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def _parse_duration(s: str) -> float:
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


@record_app.command("positives")
def record_positives_cmd(
    project: str = typer.Option(..., help="Project ID."),
    class_name: str = typer.Option(..., "--class",
                  help="Target class — must match one defined in project.toml [model] classes."),
    num: int = typer.Option(0, help="Stop after this many clips. 0 = use duration."),
    duration: str = typer.Option("15m", help="e.g. '30m', '2h', '1200s'"),
    mic_position: str = typer.Option(""),
) -> None:
    """Record on-site target positives (onset-triggered, 1 s clips).

    Saves to projects/<id>/recordings/positives/<class>/. Used by dataset
    compose. Ctrl+C is safe.
    """
    _setup_logging()
    from sed.config import load_config, ProjectLayout
    from sed.data.recorder import record_auto
    cfg = load_config(project)
    layout = ProjectLayout(project); layout.create()
    if class_name not in cfg.model.classes:
        console.print(
            f"[yellow]warning: '{class_name}' is not in project.toml [model] classes "
            f"({cfg.model.classes}). The clips will be saved but won't be picked up by "
            f"`dataset compose` until you add the class to the project config.[/yellow]"
        )
    out = layout.positives_dir(class_name); out.mkdir(parents=True, exist_ok=True)
    n = record_auto(cfg, out, class_name=class_name, project_id=project,
                    duration_s=_parse_duration(duration), mic_position=mic_position,
                    max_clips=(num if num > 0 else None))
    console.print(f"[green]saved {n} clips → {out}[/green]")


@record_app.command("hardnegs")
def record_hardnegs_cmd(
    project: str = typer.Option(..., help="Project ID."),
    class_name: str = typer.Option(..., "--class",
                  help="Hard-negative label (any free-form name: espresso, register, etc.)."),
    num: int = typer.Option(0, help="Stop after this many clips. 0 = use duration."),
    duration: str = typer.Option("15m"),
    mic_position: str = typer.Option(""),
) -> None:
    """Record venue-specific hard-negatives (onset-triggered, 1 s clips).

    Saves to projects/<id>/recordings/hardneg/<class>/. All hardneg clips
    (regardless of class name) are folded into the _negative class during
    dataset composition — the class_name is for your bookkeeping only.

    Most hard-negatives belong in the shared library
    (`clinkd library record-hardneg`), not the project. Use this command
    only for distractors specific to THIS venue's setup (e.g. the espresso
    machine right next to mic 2).
    """
    _setup_logging()
    from sed.config import load_config, ProjectLayout
    from sed.data.recorder import record_auto
    cfg = load_config(project)
    layout = ProjectLayout(project); layout.create()
    out = layout.hardneg_dir(class_name); out.mkdir(parents=True, exist_ok=True)
    n = record_auto(cfg, out, class_name=class_name, project_id=project,
                    duration_s=_parse_duration(duration), mic_position=mic_position,
                    max_clips=(num if num > 0 else None))
    console.print(f"[green]saved {n} clips → {out}[/green]")


@record_app.command("ambience")
def record_ambience_cmd(
    project: str = typer.Option(..., help="Project ID."),
    duration: str = typer.Option("30m", help="Total wall-clock duration."),
    chunk: float = typer.Option(30.0, help="Seconds per output WAV file."),
    density: str = typer.Option("empty", help="empty | quiet | normal | busy | peak. "
                                              "Tag for what you're capturing. 'empty' is the "
                                              "default for an install before the venue opens."),
    mic_position: str = typer.Option(""),
) -> None:
    """Record venue ambience (continuous, untriggered).

    Saves to projects/<id>/recordings/ambience/. Ambience is NOT a model
    class — it's the room tone / crowd noise of THIS venue, used at two
    points downstream:
      • Augmentation: project ambience is added to the library ambience
        pool so domain mixing during training uses both sources.
      • Calibration: `clinkd calibrate --ambience-source venue` (or
        simulated_busy) reads from here to sweep thresholds.

    Tag the density truthfully — empty/quiet for a pre-opening install,
    busy/peak when you return on a live night.

    Ctrl+C is safe: the in-flight partial chunk is also saved.
    """
    _setup_logging()
    from sed.config import load_config, ProjectLayout
    from sed.data.recorder import record_continuous
    cfg = load_config(project)
    layout = ProjectLayout(project); layout.create()
    out = layout.ambience_dir; out.mkdir(parents=True, exist_ok=True)

    # Write a density sidecar on each clip so we know what was captured
    # without re-listening. The on_save callback patches the per-clip JSON
    # metadata file the recorder already writes.
    import json as _json

    def _tag_density(wav_path, _meta):
        sidecar = wav_path.with_suffix(".json")
        try:
            data = _json.loads(sidecar.read_text())
        except Exception:  # noqa: BLE001
            data = {}
        data["density"] = density
        sidecar.write_text(_json.dumps(data, indent=2))

    n = record_continuous(cfg, out, class_name="ambience", project_id=project,
                          duration_s=_parse_duration(duration), chunk_s=chunk,
                          mic_position=mic_position, on_save=_tag_density)
    console.print(f"[green]saved {n} chunks (density={density}) → {out}[/green]")


# Legacy aliases — kept for back-compat with earlier docs/scripts. New code
# should use the explicit subcommands above.
@record_app.command("auto", hidden=True)
def record_auto_cmd(
    project: str = typer.Option(...),
    class_name: str = typer.Option(..., "--class"),
    kind: str = typer.Option("positive", help="positive | hardneg"),
    duration: str = typer.Option("10m"),
    mic_position: str = typer.Option(""),
) -> None:
    """[deprecated] Use `record positives` or `record hardnegs` instead."""
    console.print("[yellow]`record auto` is deprecated — use `record positives` "
                  "or `record hardnegs`[/yellow]")
    if kind == "positive":
        record_positives_cmd(project=project, class_name=class_name, num=0,
                             duration=duration, mic_position=mic_position)
    elif kind == "hardneg":
        record_hardnegs_cmd(project=project, class_name=class_name, num=0,
                            duration=duration, mic_position=mic_position)
    else:
        raise typer.BadParameter("kind must be 'positive' or 'hardneg'")


@record_app.command("continuous", hidden=True)
def record_continuous_cmd(
    project: str = typer.Option(...),
    class_name: str = typer.Option("ambience", "--class"),
    duration: str = typer.Option("30m"),
    chunk: float = typer.Option(30.0),
    mic_position: str = typer.Option(""),
) -> None:
    """[deprecated] Use `record ambience` instead."""
    console.print("[yellow]`record continuous` is deprecated — use "
                  "`record ambience`[/yellow]")
    if class_name != "ambience":
        raise typer.BadParameter(
            "non-ambience continuous recording is no longer supported; "
            "use `record positives` / `record hardnegs` (onset-triggered) or "
            "`record ambience` (continuous)."
        )
    record_ambience_cmd(project=project, duration=duration, chunk=chunk,
                       mic_position=mic_position)


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


@library_app.command("stats")
def library_stats() -> None:
    from sed.data.library import Library
    lib = Library()
    s = lib.stats()
    console.print(s)


@library_app.command("import")
def library_import(
    src_dir: Path = typer.Argument(..., exists=True),
    category: str = typer.Option(..., help="glasses_clean | ambience | negatives_external"),
    label: str = typer.Option(..., help="e.g. wine_thin / bar_busy / cutlery"),
    tags: list[str] = typer.Option([], "--tag"),
    notes: str = typer.Option(""),
) -> None:
    """Import an existing folder of WAV files into the library."""
    _setup_logging()
    import soundfile as sf
    from sed.data.library import Library
    lib = Library()
    n = 0
    for wav in sorted(Path(src_dir).rglob("*.wav")):
        try:
            info = sf.info(str(wav))
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]skip {wav}: {e}[/yellow]")
            continue
        lib.add_clip(
            source_path=wav, category=category, label=label,
            sample_rate=info.samplerate, channels=info.channels,
            duration_s=info.frames / info.samplerate if info.samplerate else 0.0,
            tags=list(tags), notes=notes, move=False,
        )
        n += 1
    console.print(f"[green]imported {n} clips[/green]")


def _make_library_on_save(lib, category: str, label: str, tags: list[str], notes: str):
    """Return an on_save callback that registers each saved clip into the
    library manifest the moment it lands on disk. The wav file is ALREADY
    in the right library subdirectory, so we only need to write the manifest
    entry — no file copy or move."""
    from dataclasses import asdict
    from datetime import datetime, timezone
    import json as _json

    def _on_save(wav_path, meta) -> None:
        # Register in library manifest directly; the clip is already inside
        # data_library/<category>/<label>/ so no file IO beyond appending.
        entry = {
            "uuid": meta.uuid,
            "relpath": str(wav_path.relative_to(lib.root)),
            "category": category,
            "label": label,
            "duration_s": meta.duration_s,
            "sample_rate": meta.sample_rate,
            "channels": meta.channels,
            "recorded_at": meta.recorded_at,
            "session_id": meta.session_id,
            "device": meta.device,
            "mic_position": meta.mic_position,
            "tags": tags,
            "notes": notes,
        }
        with open(lib.manifest, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry) + "\n")
    return _on_save


@library_app.command("record-glass")
def library_record_glass(
    glass_type: str = typer.Option(..., "--type", help="e.g. wine_thin"),
    num: int = typer.Option(50, help="Stop after this many onset-triggered clips. "
                                     "Set 0 to record for the full duration."),
    duration: str = typer.Option("15m", help="Wall-clock duration cap; e.g. 10m, 1h."),
    notes: str = typer.Option(""),
) -> None:
    """Record studio-clean glass clinks DIRECTLY into the library.

    Onset-triggered: each clink produces one 1-second clip. Ctrl+C stops
    early; clips already recorded stay in the library and the manifest
    entry is written for each one the moment it's saved (so nothing is
    lost on interrupt).
    """
    _setup_logging()
    from sed.config import load_config
    from sed.data.library import Library
    from sed.data.recorder import record_auto
    cfg = load_config()
    lib = Library()
    out_dir = lib.root / "glasses_clean" / glass_type
    out_dir.mkdir(parents=True, exist_ok=True)
    on_save = _make_library_on_save(lib, "glasses_clean", glass_type,
                                    tags=["studio"], notes=notes)
    n = record_auto(
        cfg=cfg, out_dir=out_dir, class_name=glass_type,
        project_id="_library", duration_s=_parse_duration(duration),
        max_clips=(num if num > 0 else None),
        on_save=on_save,
    )
    console.print(f"[green]added {n} clips to library/glasses_clean/{glass_type}[/green]")


@library_app.command("record-ambience")
def library_record_ambience(
    label: str = typer.Option(..., "--label", help="e.g. bar_busy / restaurant_dinner / cafe_morning"),
    duration: str = typer.Option("30m"),
    chunk: float = typer.Option(30.0, help="Seconds per output WAV."),
    density: str = typer.Option("normal", help="quiet | normal | busy | peak"),
    location: str = typer.Option("", help="Free-text location note (city, venue type)"),
    notes: str = typer.Option(""),
) -> None:
    """Record ambience into the shared library (off-site, reusable across projects).

    Use this from any bar/restaurant/cafe with permission, NOT just the
    target venue. The library ambience pool powers training-time domain
    mixing — the more variety here, the better every model trained on
    it generalises. Aim for at least 3-5 distinct sources at varied
    densities.

    Ctrl+C stops early; the in-flight partial chunk is also saved.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.data.library import Library
    from sed.data.recorder import record_continuous
    cfg = load_config()
    lib = Library()
    tags = [f"density:{density}"]
    if location:
        tags.append(f"loc:{location.replace(' ', '_').lower()}")
    out_dir = lib.root / "ambience" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    on_save = _make_library_on_save(lib, "ambience", label, tags=tags, notes=notes)
    n = record_continuous(
        cfg=cfg, out_dir=out_dir, class_name=label,
        project_id="_library", duration_s=_parse_duration(duration),
        chunk_s=chunk, on_save=on_save,
    )
    console.print(f"[green]added {n} ambience chunks to library/ambience/{label}[/green]")


@library_app.command("record-hardneg")
def library_record_hardneg(
    label: str = typer.Option(..., "--label", help="e.g. cutlery / ice_in_glass / bottle_clank"),
    mode: str = typer.Option("auto", help="auto = onset-triggered (one clip per impact); "
                                          "continuous = stream chunks like ambience."),
    num: int = typer.Option(0, help="For auto mode, stop after this many clips. "
                                    "0 = full duration."),
    duration: str = typer.Option("15m"),
    chunk: float = typer.Option(30.0, help="For continuous mode, seconds per chunk."),
    notes: str = typer.Option(""),
) -> None:
    """Record hard-negatives into the shared library.

    Two modes:
      auto       — onset-triggered (default). One 1-second clip per impact.
                   Best for short, repeatable sounds: bashing cutlery, ice
                   cubes hitting a glass, bottle clanks. You make the sound
                   N times, you get N clips.
      continuous — stream chunks like ambience. Best for ongoing distractor
                   sounds: dishwasher running, espresso machine cycle, busy
                   kitchen.

    Library hard-negatives are reusable across every project: cutlery sounds
    at one restaurant are the same kind of distractor at another. Build the
    library deeply once; future venues benefit automatically.

    Ctrl+C stops early; clips already captured stay.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.data.library import Library
    from sed.data.recorder import record_auto, record_continuous
    if mode not in ("auto", "continuous"):
        raise typer.BadParameter("mode must be 'auto' or 'continuous'")
    cfg = load_config()
    lib = Library()
    out_dir = lib.root / "negatives_external" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    on_save = _make_library_on_save(lib, "negatives_external", label,
                                    tags=[], notes=notes)
    if mode == "auto":
        n = record_auto(
            cfg=cfg, out_dir=out_dir, class_name=label,
            project_id="_library", duration_s=_parse_duration(duration),
            max_clips=(num if num > 0 else None),
            on_save=on_save,
        )
    else:
        n = record_continuous(
            cfg=cfg, out_dir=out_dir, class_name=label,
            project_id="_library", duration_s=_parse_duration(duration),
            chunk_s=chunk, on_save=on_save,
        )
    console.print(f"[green]added {n} hardneg clips to library/negatives_external/{label}[/green]")


@library_app.command("ambience-coverage")
def library_ambience_coverage() -> None:
    """Show ambience pool coverage: total minutes, per-label breakdown,
    density distribution. Use this to know whether your library is rich
    enough to train against."""
    from sed.data.library import Library
    lib = Library()
    clips = lib.query(category="ambience")
    if not clips:
        console.print("[yellow]library has no ambience clips yet[/yellow]")
        console.print("Record some with: clinkd library record-ambience --label bar_busy")
        return
    by_label: dict[str, list] = {}
    by_density: dict[str, float] = {}
    total_s = 0.0
    for c in clips:
        by_label.setdefault(c.label, []).append(c)
        total_s += c.duration_s
        density = next(
            (t.split(":", 1)[1] for t in c.tags if t.startswith("density:")),
            "unknown",
        )
        by_density[density] = by_density.get(density, 0.0) + c.duration_s

    console.print(f"[bold]ambience pool: {total_s/60:.1f} min, {len(clips)} clips, "
                  f"{len(by_label)} distinct labels[/bold]")
    table = Table(title="by label")
    table.add_column("label"); table.add_column("clips", justify="right")
    table.add_column("minutes", justify="right")
    for lbl, cs in sorted(by_label.items()):
        mins = sum(c.duration_s for c in cs) / 60
        table.add_row(lbl, str(len(cs)), f"{mins:.1f}")
    console.print(table)

    table2 = Table(title="by density")
    table2.add_column("density"); table2.add_column("minutes", justify="right")
    for d in ["quiet", "normal", "busy", "peak", "unknown"]:
        if d in by_density:
            table2.add_row(d, f"{by_density[d]/60:.1f}")
    console.print(table2)

    # Heuristic warnings
    if total_s / 60 < 20:
        console.print("[yellow]⚠ pool is thin (<20 min). Aim for 60+ min across "
                      "3-5 distinct sources.[/yellow]")
    if len(by_label) < 3:
        console.print("[yellow]⚠ low variety (<3 labels). Domain mixing is "
                      "more effective with diverse sources.[/yellow]")
    if "busy" not in by_density and "peak" not in by_density:
        console.print("[yellow]⚠ no busy/peak ambience. The model won't see "
                      "realistic peak-hour conditions during training.[/yellow]")


# ---------------------------------------------------------------------------
# inspect / quarantine
# ---------------------------------------------------------------------------


@app.command("inspect")
def inspect_cmd(
    project: str = typer.Option(...),
    open_browser: bool = typer.Option(False, "--open",
                  help="Open the report in your default browser (static mode)."),
    serve: bool = typer.Option(False, "--serve",
                  help="Run an interactive local server: tick a checkbox on any "
                       "clip to quarantine it (writes the sidecar flag live, no "
                       "CSV copy/paste). Auto-opens the browser."),
    port: int = typer.Option(8765, help="Port for --serve mode."),
    max_thumbs: int = typer.Option(30, help="Max thumbnails per class in the report."),
    no_augmentation_preview: bool = typer.Option(
        False, "--no-aug",
        help="Skip the augmentation preview section (faster).",
    ),
) -> None:
    """Generate an HTML inspection report (spectrograms + UMAP + outlier list).

    Two modes:
      static  (default): writes a self-contained HTML you open from disk.
                         No interactivity beyond what the browser provides.
      --serve:           launches an interactive local server. Each clip gets
                         a checkbox; ticking it writes the sidecar `flagged`
                         attribute live, so the next `clinkd dataset compose`
                         skips it. No CSV copy/paste needed.
    """
    _setup_logging()
    from sed.config import load_config
    cfg = load_config(project)
    if serve:
        # The server's initial render also writes the static HTML so the
        # file-on-disk reflects the current state too. After that it serves
        # a re-rendered, interactive copy on each GET.
        from sed.data.inspect_server import serve as run_server
        run_server(cfg, project, port=port,
                   max_thumbs_per_class=max(max_thumbs, 200),  # bigger panels for curation
                   augmentation_preview=not no_augmentation_preview)
        return
    # Static mode
    from sed.data.inspect import inspect_project
    out = inspect_project(
        cfg, project, max_thumbs_per_class=max_thumbs,
        augmentation_preview=not no_augmentation_preview,
        interactive=False,
    )
    console.print(f"[green]report → {out}[/green]")
    if open_browser:
        import webbrowser
        webbrowser.open(out.as_uri())


@app.command("quarantine")
def quarantine_cmd(
    project: str = typer.Option(...),
    csv_or_uuids: list[str] = typer.Argument(...,
                  help="Path to outliers.csv OR space-separated uuids/paths."),
) -> None:
    """Flag clips so they're excluded from the next dataset compose."""
    _setup_logging()
    from sed.config import load_config
    from sed.data.inspect import quarantine
    cfg = load_config(project)
    targets: list[str] = []
    for arg in csv_or_uuids:
        p = Path(arg)
        if p.exists() and p.suffix == ".csv":
            import csv as csvlib
            with open(p, "r", encoding="utf-8") as f:
                for row in csvlib.DictReader(f):
                    targets.append(row.get("path") or row.get("uuid"))
        else:
            targets.append(arg)
    n = quarantine(cfg, project, [t for t in targets if t])
    console.print(f"[green]flagged {n} clips[/green]")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


@dataset_app.command("compose")
def dataset_compose_cmd(project: str = typer.Option(...)) -> None:
    """Build train/val/test from project recordings + library refs."""
    _setup_logging()
    from sed.config import load_config
    from sed.data.compose import compose_dataset
    cfg = load_config(project)
    summary = compose_dataset(cfg, project)
    console.print(summary)


# ---------------------------------------------------------------------------
# train / export / calibrate / run
# ---------------------------------------------------------------------------


@app.command("train")
def train_cmd(
    project: str = typer.Option(...),
    resume: Optional[str] = typer.Option(None, help="Path to a Lightning checkpoint."),
) -> None:
    _setup_logging()
    from sed.config import load_config
    from sed.train.train import train_project
    cfg = load_config(project)
    ckpt = train_project(cfg, project, resume=resume)
    console.print(f"[green]best checkpoint → {ckpt}[/green]")


@app.command("export")
def export_cmd(
    project: str = typer.Option(...),
    calibration_n: int = typer.Option(300),
) -> None:
    _setup_logging()
    from sed.config import load_config
    from sed.export.openvino import export_project
    cfg = load_config(project)
    manifest = export_project(cfg, project, calibration_n=calibration_n)
    console.print(manifest)


@app.command("calibrate")
def calibrate_cmd(
    project: str = typer.Option(...),
    target_fp_per_hour: float = typer.Option(1.0, help="Tolerated false-positive rate."),
    ambience_dir: Optional[Path] = typer.Option(None,
                  help="Override path; defaults to projects/<id>/recordings/ambience."),
    ambience_source: str = typer.Option("venue",
                  help="venue | library | simulated_busy. Use 'simulated_busy' "
                       "when you only have empty-venue ambience: it mixes that "
                       "with library busy-ambience to estimate peak-hour FP."),
    simulated_minutes: float = typer.Option(30.0,
                  help="For simulated_busy, how many minutes of synthetic stream."),
    simulated_snr_low: float = typer.Option(-3.0,
                  help="For simulated_busy, lowest crowd-vs-room SNR (dB)."),
    simulated_snr_high: float = typer.Option(12.0,
                  help="For simulated_busy, highest crowd-vs-room SNR (dB)."),
) -> None:
    """Calibrate per-class detection thresholds against ambience.

    Three sources of ambience:
      venue (default)   — recorded at the project's site
      library           — shared library ambience pool only
      simulated_busy    — empty-venue + library busy-ambience mixed (the
                          realistic mode when setup happens before opening)
    """
    _setup_logging()
    from sed.config import load_config
    from sed.runtime.calibrate import calibrate
    if ambience_source not in ("venue", "library", "simulated_busy"):
        raise typer.BadParameter(
            "ambience_source must be one of: venue, library, simulated_busy"
        )
    cfg = load_config(project)
    chosen = calibrate(
        cfg, project,
        ambience_dir=ambience_dir,
        target_fp_per_hour=target_fp_per_hour,
        ambience_source=ambience_source,
        simulated_minutes=simulated_minutes,
        simulated_snr_db_range=(simulated_snr_low, simulated_snr_high),
    )
    console.print(chosen)


@app.command("run")
def run_cmd(
    project: str = typer.Option(...),
    no_artnet: bool = typer.Option(False, "--no-artnet"),
    no_live: bool = typer.Option(False, "--no-live"),
    targets: Optional[list[str]] = typer.Option(None, "--target", help=
        "Restrict triggering to these target classes (repeatable). "
        "Omit to use targets enabled in project.toml."),
) -> None:
    """Start the 24/7 inference loop."""
    from sed.runtime.run import run as runtime_run
    runtime_run(project, with_artnet=not no_artnet, live=not no_live,
                targets_override=list(targets) if targets else None)


# ---------------------------------------------------------------------------
# check — pre-flight checks for each phase
# ---------------------------------------------------------------------------


@check_app.command("record")
def check_record_cmd(
    project: Optional[str] = typer.Option(None, help="Use this project's audio config."),
    seconds: float = typer.Option(5.0, help="Capture window for the test."),
) -> None:
    """Pre-record check: audio device, gains, all mics functional, formats OK."""
    _setup_logging("WARNING")          # quiet — check prints its own table
    from sed.config import load_config
    from sed import doctor as checks_mod
    cfg = load_config(project) if project else load_config()
    checks = checks_mod.doctor_record(cfg, seconds=seconds)
    checks_mod.render(checks, "Check — pre-record")
    if checks_mod.all_failed(checks):
        raise typer.Exit(code=2)


@check_app.command("train")
def check_train_cmd(project: str = typer.Option(...)) -> None:
    """Pre-train check: dataset stats, leakage, ambience pool, hyperparam sanity."""
    _setup_logging("WARNING")
    from sed.config import load_config
    from sed import doctor as checks_mod
    cfg = load_config(project)
    checks = checks_mod.doctor_train(cfg, project)
    checks_mod.render(checks, f"Check — pre-train · {project}")
    if checks_mod.all_failed(checks):
        raise typer.Exit(code=2)


@check_app.command("deploy")
def check_deploy_cmd(project: str = typer.Option(...)) -> None:
    """Pre-deploy check: model loads, inference latency OK, ArtNet reachable."""
    _setup_logging("WARNING")
    from sed.config import load_config
    from sed import doctor as checks_mod
    cfg = load_config(project)
    checks = checks_mod.doctor_deploy(cfg, project)
    checks_mod.render(checks, f"Check — pre-deploy · {project}")
    if checks_mod.all_failed(checks):
        raise typer.Exit(code=2)


@check_app.command("all")
def check_all_cmd(project: str = typer.Option(...)) -> None:
    """Run all three check suites in sequence."""
    _setup_logging("WARNING")
    from sed.config import load_config
    from sed import doctor as checks_mod
    cfg = load_config(project)
    for fn, label in [
        (lambda: checks_mod.doctor_record(cfg, seconds=3.0), "pre-record"),
        (lambda: checks_mod.doctor_train(cfg, project), "pre-train"),
        (lambda: checks_mod.doctor_deploy(cfg, project), "pre-deploy"),
    ]:
        try:
            checks = fn()
            checks_mod.render(checks, f"Check — {label} · {project}")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]check {label} failed: {e}[/red]")


# ---------------------------------------------------------------------------
# augmentation preview (also runs as part of `inspect` by default)
# ---------------------------------------------------------------------------


@dataset_app.command("preview-augmentation")
def preview_augmentation_cmd(
    project: str = typer.Option(...),
    n_per_class: int = typer.Option(20, help="Generate N augmented variants per target class."),
    open_browser: bool = typer.Option(False, "--open"),
) -> None:
    """Generate augmented samples and add them to the inspection report.

    This is what the model will actually train on — listen to flagged samples
    and tune `[augmentation]` in project.toml if too many are masked.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.data.inspect import inspect_project
    cfg = load_config(project)
    out = inspect_project(cfg, project, augmentation_preview=True,
                          aug_per_class=n_per_class)
    console.print(f"[green]report → {out}[/green]")
    if open_browser:
        import webbrowser
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    app()
