"""Top-level Typer CLI.

    sed --help

Subcommand groups:

    audio       device discovery, simple capture probe
    library     inspect / rebuild the shared sound library
    project     create / list / inspect projects
    record      record onset-triggered clips into the shared data library
    dataset     compose / inspect / augment train/val/test splits
    train       fine-tune the model (auto-tests + exports OpenVINO)
    models      list / compare / select / show / delete trained models
    calibrate   sweep thresholds against negative/background clips to hit a target FP/h
    evaluate    score a model against the frozen test set; add/list test clips
    inference   start the 24/7 inference loop
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
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
dataset_app = typer.Typer(no_args_is_help=True, help="Compose / preview a dataset.")
pretraining_app = typer.Typer(
    no_args_is_help=True,
    help="Import / index a large pretraining corpus (data_library/pretraining/).",
)
app.add_typer(audio_app, name="audio")
app.add_typer(library_app, name="library")
app.add_typer(project_app, name="project")
app.add_typer(dataset_app, name="dataset")
app.add_typer(pretraining_app, name="pretraining")

check_app = typer.Typer(no_args_is_help=True,
                        help="Pre-flight check suites for each phase: sed check record, "
                             "sed check train, sed check deploy, sed check all.")
app.add_typer(check_app, name="check")

models_app = typer.Typer(no_args_is_help=True,
                         help="Compare, select, show, or delete trained models.")
app.add_typer(models_app, name="models")

evaluate_app = typer.Typer(no_args_is_help=True,
                           help="Score a trained model against the project's "
                                "frozen test set. Run bare to regenerate the "
                                "frozen-test report; `augmented` audits derived "
                                "clips; `list` shows the test set.")
app.add_typer(evaluate_app, name="evaluate")

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    from sed.runtime.run import quiet_noisy_loggers

    quiet_noisy_loggers()


def _print_devices_table(project_id: str | None = None) -> None:
    import sounddevice as sd

    from sed.audio.capture import list_input_devices, resolve_device

    active_idx = None
    configured_device = None
    try:
        cfg = _load_capture_config(project_id)
        configured_device = cfg.deploy.record_device
        active_idx = resolve_device(configured_device)
    except Exception:
        try:
            active_idx = resolve_device("default")
        except Exception:
            pass

    default_idx = sd.default.device[0]

    title = "Audio input devices"
    if project_id and configured_device:
        title = f"Audio input devices (project: {project_id}, configured: {configured_device})"

    table = Table(title=title)
    table.add_column("idx")
    table.add_column("default")
    table.add_column("name")
    table.add_column("ch", justify="right")
    table.add_column("sr", justify="right")
    table.add_column("api")

    for d in list_input_devices():
        is_default = (d.index == default_idx)
        is_active = (active_idx is not None and d.index == active_idx)

        # Build default column string
        val_default = ""
        if is_default:
            if is_active and project_id:
                val_default = "yes (config)"
            else:
                val_default = "yes"
        elif is_active and project_id:
            val_default = "config"

        # Row styling
        row_style = None
        if project_id:
            if is_active:
                row_style = "bold green"
        else:
            if is_default:
                row_style = "bold green"

        table.add_row(
            str(d.index),
            val_default,
            d.name,
            str(d.max_input_channels),
            f"{d.default_samplerate:.0f}",
            d.hostapi_name,
            style=row_style,
        )
    console.print(table)


def _device_callback(value: Optional[str]):
    if value == "LIST":
        import sys
        project_id = None
        for i, arg in enumerate(sys.argv):
            if arg in ("--project", "-p") and i + 1 < len(sys.argv):
                project_id = sys.argv[i + 1]
                break

        _print_devices_table(project_id)
        raise typer.Exit()
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt_number(value: object, digits: int = 3, missing: str = "-") -> str:
    if not _is_number(value):
        return missing
    return f"{float(value):.{digits}f}"


def _fmt_loss(value: object) -> str:
    return _fmt_number(value, digits=4)


def _fmt_percent(value: object, missing: str = "-") -> str:
    if not _is_number(value):
        return missing
    return f"{float(value) * 100:.1f}%"


def _model_created_label(manifest: dict) -> str:
    created = str(manifest.get("created_at") or "")
    return created[:16].replace("T", " ") if created else "-"


def _model_epoch_label(manifest: dict) -> str:
    training = manifest.get("training") or {}
    ran = training.get("epochs_ran", "?")
    max_epochs = training.get("epochs_max", "?")
    return f"{ran}/{max_epochs}"


def _model_ir_label(layout, model_id: str) -> str:
    model_dir = layout.model_dir(model_id)
    return "int8" if (model_dir / "model_int8.xml").exists() else (
        "fp16" if (model_dir / "model_fp16.xml").exists() else "-"
    )


def _model_calibrated_label(layout, model_id: str) -> str:
    return "yes" if (layout.model_dir(model_id) / "runtime_settings.json").exists() else "no"


def _print_model_id_key(items: list[dict], active: str | None = None) -> list[str]:
    labels: list[str] = []
    console.print("[bold]Model IDs[/bold]  [dim](copy the full id from these lines)[/dim]")
    for idx, manifest in enumerate(items, start=1):
        model_id = str(manifest.get("model_id") or "")
        handle = f"#{idx}"
        labels.append(handle)
        active_label = "  [green]active[/green]" if model_id == active else ""
        console.print(
            f"  [bold]{handle}[/bold]  [cyan]{escape(model_id)}[/cyan]  "
            f"[dim]{_model_created_label(manifest)}[/dim]{active_label}",
            soft_wrap=True,
        )
    return labels


def _resolve_model_id(project: str, value: str) -> str:
    """Map a ``#N`` handle or a full model id to a full model id.

    ``#N`` indexes ``registry.load_manifests`` (newest-first — the same order the
    Model IDs header numbers). A full id is returned as-is once verified. Anything
    else raises ``typer.BadParameter`` listing the available ids.
    """
    from sed.train import registry
    raw = (value or "").strip()
    ids = [str(m.get("model_id") or "") for m in registry.load_manifests(project)]
    if raw.startswith("#"):
        try:
            idx = int(raw[1:])
        except ValueError:
            idx = 0
        if 1 <= idx <= len(ids):
            return ids[idx - 1]
        raise typer.BadParameter(
            f"no model {raw}; project '{project}' has {len(ids)} model(s). "
            "See `sed models compare`."
        )
    if raw in ids:
        return raw
    avail = ", ".join(ids) if ids else "none"
    raise typer.BadParameter(
        f"'{raw}' is not a trained model for '{project}'. Available: {avail}."
    )


def _evaluation_result_line(model_id: str, metrics: dict) -> str:
    return (
        f"[green]{escape(model_id)}[/green]  "
        f"n={metrics['n_samples']}  "
        f"acc={metrics.get('accuracy', 0.0):.3f}  "
        f"prob={escape(str(metrics.get('probability_mode', 'unknown')))}  "
        f"F1(targets)={metrics['f1_macro_targets']:.3f}  "
        f"AP(targets)={metrics['ap_macro_targets']:.3f}"
    )


_VERBOSE_LEVELS = ("none", "info", "live")


def _verbose_callback(value: Optional[str]) -> Optional[str]:
    """Normalize the --verbose level. Omitted → config; bare --verbose → live."""
    if value is None:
        return None
    level = value.lower()
    if level not in _VERBOSE_LEVELS:
        raise typer.BadParameter(
            f"must be one of: {', '.join(_VERBOSE_LEVELS)}"
        )
    return level


class OptionalValueCommand(typer.core.TyperCommand):
    """Command supporting Click "optional value" options.

    Typer cannot express options whose value is optional, so we patch the
    relevant parameters after construction. Subclasses set
    ``optional_value_params`` mapping a parameter name to the value used when
    the flag is given bare (``--flag`` with no following value).
    """

    optional_value_params: dict[str, str] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if param.name in self.optional_value_params:
                param.is_flag = False
                param.flag_value = self.optional_value_params[param.name]
                param._flag_needs_value = True


class InferenceCommand(OptionalValueCommand):
    """Inference command: bare ``--device`` lists devices, bare ``--verbose`` -> live."""

    optional_value_params = {"device": "LIST", "verbose": "live"}

    def format_help(self, ctx: typer.Context, formatter) -> None:
        defaults, epilog = _resolved_inference_help(ctx.params if ctx else None)
        originals = {
            param.name: (param.default, getattr(param, "show_default", None))
            for param in self.params
        }
        original_help = self.help
        try:
            for param in self.params:
                if param.name in defaults:
                    param.default = defaults[param.name]
                    param.show_default = True
            if epilog:
                self.help = f"{self.help or ''}\n\n{epilog}"
            return super().format_help(ctx, formatter)
        finally:
            self.help = original_help
            for param in self.params:
                if param.name in originals:
                    param.default, param.show_default = originals[param.name]


def _argv_option_values(flags: tuple[str, ...], *, optional_value: bool = False) -> list[str | None]:
    import sys

    out: list[str | None] = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        matched = False
        for flag in flags:
            if arg == flag:
                if i + 1 >= len(args) or args[i + 1] in {"--help", "-h"}:
                    out.append(None)
                elif optional_value and args[i + 1].startswith("-"):
                    out.append(None)
                else:
                    out.append(args[i + 1])
                    i += 1
                matched = True
                break
            prefix = flag + "="
            if arg.startswith(prefix):
                out.append(arg[len(prefix):])
                matched = True
                break
        i += 1
        if matched:
            continue
    return out


def _argv_flag_present(flag: str) -> bool:
    import sys

    return any(arg == flag for arg in sys.argv[1:])


def _last_option_value(flags: tuple[str, ...], *, optional_value: bool = False) -> str | None:
    values = _argv_option_values(flags, optional_value=optional_value)
    return values[-1] if values else None


def _fmt_help_value(value: object) -> str:
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value) or "none"
    return str(value)


def _resolved_inference_help(ctx_params: dict | None = None) -> tuple[dict[str, str], str]:
    ctx_params = ctx_params or {}
    project_id = ctx_params.get("project") or _last_option_value(("--project",))
    target_values = ctx_params.get("target")
    if target_values is None:
        target_values = _argv_option_values(("--target",))
    elif isinstance(target_values, str):
        target_values = [target_values]
    target_override = None
    if target_values:
        from sed.runtime.run import _parse_target_csv

        target_override = _parse_target_csv([v for v in target_values if v])
    device_override = ctx_params.get("device")
    if device_override is None:
        device_override = _last_option_value(("--device",), optional_value=True)
    verbose_override = ctx_params.get("verbose")
    if verbose_override is None:
        verbose_override = _last_option_value(("--verbose",), optional_value=True)
    if verbose_override is None and not ctx_params and _argv_option_values(("--verbose",), optional_value=True):
        verbose_override = "live"
    min_dbfs_override = ctx_params.get("min_dbfs")
    if min_dbfs_override is None:
        min_dbfs_override = _last_option_value(("--min-dbfs",))
    no_artnet = bool(ctx_params.get("no_artnet", False)) or _argv_flag_present("--no-artnet")
    mine_detections = (
        bool(ctx_params.get("mine_detections", False))
        or _argv_flag_present("--mine")
    )

    from sed.config import Config, ProjectLayout, load_config, load_config_for_inference, resolve_active_model_dir
    from sed.runtime.manifest import resolve_model_path
    from sed.runtime.run import _active_requested_classes, _resolve_verbose_level
    from sed.train.objective import probability_mode

    note = ""
    cfg = Config()
    if project_id:
        try:
            cfg = load_config_for_inference(project_id)
        except Exception as exc:  # noqa: BLE001
            note = f"runtime config unavailable: {exc}"
            try:
                cfg = load_config(project_id)
                note += " (showing editable project config where possible)"
            except Exception:  # noqa: BLE001
                cfg = Config()
    else:
        note = "no --project supplied; showing built-in defaults"

    if device_override and device_override != "LIST":
        cfg.deploy.record_device = device_override
        device_source = "CLI"
    else:
        device_source = "config" if project_id and not note.startswith("no --project") else "default"

    if min_dbfs_override is not None:
        try:
            cfg.inference.min_dbfs = float(min_dbfs_override)
            min_dbfs_source = "CLI"
        except ValueError:
            min_dbfs_source = "CLI invalid"
    else:
        min_dbfs_source = "config" if project_id and not note.startswith("no --project") else "default"

    try:
        effective_verbose = _resolve_verbose_level(cfg, verbose_override)
    except ValueError:
        effective_verbose = str(verbose_override or cfg.deploy.verbose)
    verbose_source = "CLI" if verbose_override is not None else (
        "config" if project_id and not note.startswith("no --project") else "default"
    )

    active_targets = sorted(_active_requested_classes(cfg, target_override))
    target_source = "CLI" if target_override is not None else (
        "targets config" if cfg.targets else "model classes"
    )
    mining_enabled = bool(cfg.mining.enabled or mine_detections)
    artnet_enabled = bool((not no_artnet) and cfg.artnet.enabled)
    live_monitor = effective_verbose == "live"

    active_model = cfg.deploy.active_model or "auto"
    model_path = "unresolved"
    if project_id:
        try:
            model_dir = resolve_active_model_dir(ProjectLayout(project_id), cfg.deploy.active_model)
            active_model = cfg.deploy.active_model or model_dir.name
            model_path = str(resolve_model_path(model_dir))
        except Exception:  # noqa: BLE001
            pass

    min_dbfs = _fmt_help_value(cfg.inference.min_dbfs)
    defaults = {
        "device": f"effective={cfg.deploy.record_device} ({device_source})",
        "target": f"active={_fmt_help_value(active_targets)} ({target_source})",
        "verbose": f"effective={effective_verbose} ({verbose_source})",
        "min_dbfs": f"effective={min_dbfs} ({min_dbfs_source})",
        "no_artnet": (
            "effective=off (CLI)" if no_artnet
            else f"effective={_fmt_help_value(artnet_enabled)} (artnet.enabled={_fmt_help_value(cfg.artnet.enabled)})"
        ),
        "mine_detections": (
            "effective=on (CLI)" if mine_detections
            else f"effective={_fmt_help_value(mining_enabled)} (mining.enabled={_fmt_help_value(cfg.mining.enabled)})"
        ),
    }

    artnet_reason = "disabled by --no-artnet" if no_artnet else f"artnet.enabled={_fmt_help_value(cfg.artnet.enabled)}"
    epilog_lines = [
        f"Resolved runtime defaults ({'project ' + project_id if project_id else 'no project'}):",
        f"  active_model: {active_model}",
        f"  model: {model_path}",
        f"  classes: {len(cfg.model.classes)} ({_fmt_help_value(cfg.model.classes)})",
        f"  active_targets: {_fmt_help_value(active_targets)} ({target_source})",
        (
            "  audio: "
            f"device={cfg.deploy.record_device!r} inference_channels={_fmt_help_value(cfg.inference.inference_channels)} "
            f"record_sr={cfg.audio.record_sample_rate} feature_sr={cfg.model.sample_rate}"
        ),
        (
            "  inference: "
            f"window={cfg.features.window_seconds:g}s hop={cfg.inference.hop_ms}ms "
            f"prob={probability_mode(cfg)} smoothing={_fmt_help_value(cfg.inference.smoothing_enabled)} "
            f"threshold={cfg.inference.default_threshold:g} negative_max={cfg.inference.negative_max:g} "
            f"margin={cfg.inference.margin_to_second_best:g} min_dbfs={min_dbfs}"
        ),
        (
            "  output: "
            f"artnet={_fmt_help_value(artnet_enabled)} ({artnet_reason}) "
            f"mining={_fmt_help_value(mining_enabled)} verbose={effective_verbose} "
            f"live_monitor={_fmt_help_value(live_monitor)}"
        ),
    ]
    if note:
        epilog_lines.append(f"  note: {note}")
    return defaults, "\n".join(epilog_lines)


def _load_capture_config(project_id: str | None = None):
    """Load the config that capture-side commands should use.

    Recording and pre-record checks read the editable project config.toml
    because those commands are part of the next training iteration.
    """
    from sed.config import (
        load_config,
        validate_pretrained_feature_contract,
    )

    if project_id:
        cfg = load_config(project_id)
        validate_pretrained_feature_contract(cfg, phase="recording")
        return cfg
    return load_config()


class ConfigHelpCommand(OptionalValueCommand):
    optional_value_params = {"device": "LIST"}

    def format_help(self, ctx: typer.Context, formatter) -> None:
        import sys

        # 1. Peek into sys.argv for project ID
        project_id = None
        for i, arg in enumerate(sys.argv):
            if arg in ("--project", "-p") and i + 1 < len(sys.argv):
                project_id = sys.argv[i + 1]
                break

        # 2. Load config
        try:
            cfg = _load_capture_config(project_id)
        except Exception:
            try:
                cfg = _load_capture_config()
            except Exception:
                cfg = None

        # 3. Update parameter defaults so Typer's HelpFormatter shows them natively
        if cfg:
            overrides = {}
            if hasattr(cfg, "audio"):
                overrides["device"] = ("record_device", str(cfg.deploy.record_device))
                overrides["channels"] = ("record_channels", cfg.audio.record_channels)
            for param in self.params:
                if param.name in overrides:
                    key, val = overrides[param.name]
                    if isinstance(val, float):
                        val = f"{val:g}"
                    elif isinstance(val, bool):
                        val = "True" if val else "False"

                    # Show exactly which config key is driving this default
                    param.default = f"{key}={val} (config)"
                    param.show_default = True

        return super().format_help(ctx, formatter)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


@audio_app.command("devices")
def audio_devices(
    project: Optional[str] = typer.Option(
        None, "--project", "-p",
        help="Optional project association to highlight its configured device.",
    ),
) -> None:
    """List available input devices."""
    _print_devices_table(project)


@audio_app.command("probe")
def audio_probe(
    device: str = typer.Option("default", help="Device name substring or index."),
    seconds: float = typer.Option(3.0, help="How long to record."),
    channels: str = typer.Option(
        "0", "--channels", "--channel", "-ch",
        help="Ordered channel selection, e.g. '3,0,1' (default '0').",
    ),
    sample_rate: int = typer.Option(48000),
) -> None:
    """Record a few seconds and print RMS / peak per channel — sanity check."""
    import numpy as np
    import sounddevice as sd

    from sed.audio.capture import resolve_device

    _setup_logging()
    sel = _parse_channels(channels)
    n_open = max(sel) + 1
    idx = resolve_device(device)
    console.print(f"recording {seconds:.1f}s on device {idx} (channels {sel} @ {sample_rate} Hz)")
    data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                  channels=n_open, dtype="float32", device=idx)
    sd.wait()
    table = Table(title="probe results")
    table.add_column("ch")
    table.add_column("rms", justify="right")
    table.add_column("peak", justify="right")
    for c in sel:
        rms = float(np.sqrt(np.mean(data[:, c] ** 2) + 1e-12))
        peak = float(np.max(np.abs(data[:, c])))
        table.add_row(str(c), f"{rms:.4f}", f"{peak:.4f}")
    console.print(table)
    if np.max(np.abs(data)) < 1e-3:
        console.print("[red]Signal is essentially silent — wrong device or muted mic.[/red]")


def _audio_meter_table(
    levels: list[dict[str, float]], title: str, channel_ids: list[int] | None = None
) -> Table:
    table = Table(title=title)
    table.add_column("ch", justify="right")
    table.add_column("peak", justify="right")
    table.add_column("rms", justify="right")
    table.add_column("level")
    for i, lvl in enumerate(levels):
        ch_label = str(channel_ids[i]) if channel_ids is not None else str(i)
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
            ch_label,
            f"{peak_db:6.1f} dBFS",
            f"{rms_db:6.1f} dBFS",
            f"[{style}]{bar}[/{style}]",
        )
    return table


@audio_app.command("meter")
def audio_meter(
    project: str | None = typer.Option(
        None,
        help="Use this project's editable audio config.",
    ),
    device: str | None = typer.Option(None, help="Override device name substring or index."),
    channels: str | None = typer.Option(
        None, "--channels", "--channel", "-ch",
        help="Ordered channel selection to meter, e.g. '3,0,1'. Default: config [audio] record_channels.",
    ),
    sample_rate: int | None = typer.Option(None, help="Override sample rate."),
    seconds: float = typer.Option(0.25, help="Meter refresh window."),
) -> None:
    """Live per-channel input meter for the editable project audio config."""
    import time

    import numpy as np
    import sounddevice as sd

    from sed.audio.capture import resolve_device

    _setup_logging("WARNING")
    cfg = _load_capture_config(project)
    dev = device if device is not None else cfg.deploy.record_device
    # Ordered physical channels to meter. Open enough channels to cover the
    # highest requested index, then read the selected channels in that order.
    sel = _parse_channels(channels) if channels else list(cfg.audio.record_channels)
    n_open = max(sel) + 1
    sr = sample_rate if sample_rate is not None else cfg.audio.record_sample_rate
    idx = resolve_device(dev)
    info = sd.query_devices(idx)
    title = f"Audio meter — [{idx}] {info['name']} · ch {sel} @ {sr} Hz · Ctrl+C to stop"

    if n_open > int(info["max_input_channels"]):
        raise typer.BadParameter(
            f"channels {sel} need {n_open} input channel(s), but device has "
            f"{info['max_input_channels']}"
        )

    n = max(256, int(seconds * sr))
    levels = [
        {"peak": 0.0, "rms": 0.0, "peak_db": -120.0, "rms_db": -120.0}
        for _ in sel
    ]
    xruns = 0
    last_status = ""

    def callback(indata, frames, time_info, status):
        nonlocal levels, xruns, last_status
        _ = time_info
        if status:
            xruns += 1
            last_status = str(status)
        block = indata.astype("float32", copy=False)
        next_levels = []
        for c in sel:
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
            channels=n_open,
            samplerate=sr,
            blocksize=n,
            dtype="float32",
            callback=callback,
        ), Live(
            _audio_meter_table(levels, title, sel),
            console=console,
            refresh_per_second=8,
        ) as live:
            while True:
                meter_title = title
                if xruns:
                    meter_title = f"{title} · drops {xruns} ({last_status})"
                live.update(_audio_meter_table(levels, meter_title, sel))
                time.sleep(0.125)
    except KeyboardInterrupt:
        console.print()


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


@project_app.command("new")
def project_new(
    project_id: str = typer.Argument(..., help="lowercase, no spaces, e.g. 'test_project'"),
    name: str = typer.Option("", help="Human-readable name."),
    pretraining: bool = typer.Option(False, "--pretraining", help="Mark as a pretraining/BYOL project."),
) -> None:
    """Scaffold a new project under projects/<id>/."""
    from datetime import datetime, timezone
    from sed.config import ProjectLayout, write_project_config_template
    layout = ProjectLayout(project_id)
    created_at = datetime.now(timezone.utc).isoformat()
    kind = "pretraining" if pretraining else ""
    if layout.root.exists():
        layout.create()
        if not layout.config_toml.exists():
            write_project_config_template(layout, name=name or project_id, created_at=created_at, kind=kind)
            console.print(f"[green]created missing {layout.config_toml}[/green]")
        else:
            console.print(f"[yellow]exists: {layout.root}[/yellow]")
        return
    layout.create()
    write_project_config_template(layout, name=name or project_id, created_at=created_at, kind=kind)
    console.print(f"[green]created {layout.root}[/green]")
    console.print(f"  edit {layout.config_toml} for classes/audio/runtime settings")
    console.print(f"  then open `sed dataset inspect --project {project_id} --serve` for dataset and augmentation design")


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
    console.print(f"[bold]{project_id}[/bold] @ {layout.root}")
    if not layout.root.exists():
        console.print("[red]project does not exist[/red]")
        return
    cfg = load_config(project_id)
    console.print(f"config: {layout.config_toml}")
    console.print(f"classes: {cfg.model.classes}")
    table = Table(title="recordings")
    table.add_column("kind"); table.add_column("class"); table.add_column("count", justify="right")
    pos_root = layout.recordings_dir / "positives"
    if pos_root.exists():
        for d in sorted(pos_root.iterdir()):
            if d.is_dir():
                table.add_row("positive", d.name, str(len(list(d.glob("*.wav")))))
    nroot = layout.recordings_dir / "negatives"
    if nroot.exists():
        for d in sorted(nroot.iterdir()):
            if d.is_dir():
                table.add_row("negative", d.name, str(len(list(d.glob("*.wav")))))
    console.print(table)

    from sed.config import resolve_active_model_dir
    models = layout.list_models()
    console.print(f"models:            {', '.join(models) if models else '—'}")
    console.print(f"active_model:      {cfg.deploy.active_model or '(auto)'}")
    try:
        model_dir = resolve_active_model_dir(layout, cfg.deploy.active_model)
    except FileNotFoundError as e:
        console.print(f"active model:      ❌ {e}")
        return
    ov_link = model_dir / "active_openvino.xml"
    train_cfg = model_dir / "training_config.json"
    runtime_cfg = model_dir / "runtime_settings.json"
    int8 = model_dir / "model_int8.xml"
    fp16 = model_dir / "model_fp16.xml"
    has_ir = ov_link.exists() or ov_link.is_symlink() or int8.exists() or fp16.exists()
    console.print(f"active model dir:  {model_dir}")
    console.print(f"openvino:          {'✅' if has_ir else '❌'} {model_dir.name}")
    console.print(f"training config:   {'✅' if train_cfg.exists() else '❌'} {train_cfg}")
    console.print(f"runtime settings:  {'✅' if runtime_cfg.exists() else '❌'} {runtime_cfg}")


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


def _parse_tags(value: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        tag = raw.strip().lstrip("#").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _parse_channels(raw: str) -> list[int]:
    """Parse an ordered channel selection like '0,3,1' into [0, 3, 1]."""
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return [0]
    try:
        idx = [int(p) for p in parts]
    except ValueError:
        raise typer.BadParameter(f"--channels must be integer channel indexes, got {raw!r}")
    if any(c < 0 for c in idx):
        raise typer.BadParameter("--channels indexes must be >= 0")
    if len(set(idx)) != len(idx):
        raise typer.BadParameter("--channels must not contain duplicate indexes")
    return idx


def _apply_audio_overrides(
    cfg,
    *,
    device: str | None = None,
    channels: list[int] | None = None,
) -> None:
    """Apply terminal-level recording overrides to the loaded config."""
    if device is not None:
        cfg.deploy.record_device = device
    if channels is not None:
        cfg.audio.record_channels = channels


def _apply_onset_threshold_override(cfg, onset_threshold_db: float | None = None) -> None:
    """Apply a terminal-level onset floor override in dB."""
    if onset_threshold_db is None:
        return
    cfg.onset.min_flux_db = float(onset_threshold_db)
    cfg.onset.min_flux = float(10.0 ** (cfg.onset.min_flux_db / 20.0))


def _finalize_onset_config(cfg) -> None:
    """Validate derived onset timing, including the randomized preroll range.

    Re-runs the whole ``[onset]`` section through its own validators rather than
    checking ``preroll_ms`` alone. That matters for ``--record-seconds``: a
    shorter clip than the project's ``preroll_ms_range`` used to be accepted
    silently, and then each clip drew a preroll longer than the clip itself, so
    the transient landed outside the audio that was saved.
    """
    from sed.data.record_engine import finalize_onset

    try:
        cfg.onset = finalize_onset(cfg.onset)
    except Exception as exc:  # pydantic ValidationError, reworded for the CLI
        message = str(exc).split("Value error, ", 1)[-1].split(" [type=", 1)[0]
        raise typer.BadParameter(f"[onset] {message.strip()}") from exc


_DENSITIES = ("empty", "quiet", "normal", "busy", "peak")


def record_server_default_port() -> int:
    """Default port for `sed record --serve`.

    Read from the server module rather than repeated here, and deliberately
    clear of the inspect server: that one defaults to 8765 and scans 20 ports
    upward when busy, so the two never collide.
    """
    from sed.data.record_server import DEFAULT_PORT

    return DEFAULT_PORT


@app.command("record", cls=ConfigHelpCommand)
def record_cmd(
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Sound label to save under data_library/sounds/<label>/."),
    class_name: Optional[str] = typer.Option(None, "--class", "-c", help="Alias for --label / -l."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p",
        help="Optional project association. Must already exist.",
    ),
    num: int = typer.Option(50, "--num", "-n", help="Stop after this many onset-triggered clips. Set 0 to rely on duration/Ctrl+C."),
    duration: str = typer.Option("", "--duration", "-du", help="Optional wall-clock cap, e.g. '15m'. Empty = no time limit."),
    density: str = typer.Option("normal", "--density", "-de", help="empty | quiet | normal | busy | peak"),
    description: str = typer.Option("", "--description", "-d", help="Free-form note saved in each sidecar."),
    location: str = typer.Option("", "--location", "-loc", help="Free-form venue or site note saved in each sidecar (e.g. 'rnd-office', 'bar-zone-1')."),
    mic_position: str = typer.Option("", "--mic-position", "-mp", help="Free-form mic placement note saved in each sidecar."),
    record_seconds: Optional[float] = typer.Option(
        None, "--record-seconds", "-rs",
        help="Override clip duration in seconds (e.g. 0.4). Default: config [onset] record_seconds.",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-dev", callback=_device_callback, is_eager=True,
        help='Override recording device. A bare "--device" lists available devices.',
    ),
    channels: Optional[str] = typer.Option(None, "--channels", "--channel", "-ch", help="Ordered list of 0-based channel INDEXES to record (not a channel count): interface input N is index N-1, so inputs 3 & 4 are '-ch 2,3'. E.g. '3,0,1' or '0' or '1,2'. Default: config [audio] record_channels."),
    onset_threshold: Optional[float] = typer.Option(
        None, "--onset-threshold", "-ot",
        help="Override onset floor in dB for this recording session. Lower catches quieter onsets.",
    ),
    tags: str = typer.Option(
        "", "--tags", "-tg",
        help='Comma-separated tags saved in each sidecar, e.g. "wine-glass,full-50".',
    ),
    split: bool = typer.Option(
        False, "--split",
        help="Also split each multichannel clip into per-channel mono clips as it is recorded.",
    ),
    test: bool = typer.Option(
        False, "--test", "-t",
        help=(
            "Tag clips 'holdout' so a frozen-test source rule can select them "
            "(requires --project). A recording carries no split by itself — "
            "membership is a composition decision."
        ),
    ),
    serve: bool = typer.Option(
        False, "--serve",
        help=(
            "Open the recording console in a browser instead of recording in "
            "the terminal: live per-mic meters, onset tuning while the stream "
            "runs, and manual/continuous capture. Every other flag prefills "
            "the form, and --label becomes optional."
        ),
    ),
    port: int = typer.Option(
        record_server_default_port(), help="Port for --serve mode.",
    ),
) -> None:
    """Record onset-triggered clips into the unified shared data library."""
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.data.library import HOLDOUT_TAG, Library, RAW_CATEGORY
    from sed.data.recorder import record_auto

    target_label = label or class_name
    if not target_label and not serve:
        raise typer.BadParameter("Must specify --label (-l) or --class (-c).")

    if test and not project:
        raise typer.BadParameter("--test requires --project.")

    if density not in _DENSITIES:
        raise typer.BadParameter(
            f"--density must be one of: {', '.join(_DENSITIES)}"
        )
    parsed_tags = _parse_tags(tags)
    # Recording intent becomes a tag, not clip-level membership: a frozen-test
    # source rule filtering on this tag decides what actually lands in test.csv.
    if test and HOLDOUT_TAG not in parsed_tags:
        parsed_tags.append(HOLDOUT_TAG)
    if project:
        layout = ProjectLayout(project)
        if not layout.root.exists():
            raise typer.BadParameter(
                f"Project '{project}' does not exist. Run 'sed project new {project}' first."
            )
    from sed.config import LIBRARY_PROJECT_ID

    project_id = project or LIBRARY_PROJECT_ID
    cfg = _load_capture_config(project)
    if record_seconds is not None:
        if record_seconds <= 0:
            raise typer.BadParameter("--record-seconds must be > 0.")
        cfg.onset.record_seconds = float(record_seconds)

    channels_list = _parse_channels(channels) if channels else None
    _apply_audio_overrides(cfg, device=device, channels=channels_list)
    _apply_onset_threshold_override(cfg, onset_threshold)
    _finalize_onset_config(cfg)

    if serve:
        from sed.data.record_server import serve as run_record_server

        run_record_server(
            cfg,
            project or "",
            port=port,
            prefill={
                "label": target_label,
                "density": density,
                "tags": parsed_tags,
                "location": location,
                "mic_position": mic_position,
                "description": description,
                "holdout": test,
                "split_channels": split,
                "num": num,
                "duration": duration,
            },
        )
        return

    lib = Library()
    out = lib.raw_dir(target_label)
    console.print(
        f"[bold]recording[/bold] [cyan]{target_label}[/cyan] "
        f"[dim]project={project_id} density={density} -> {out}[/dim]"
    )
    duration_s = _parse_duration(duration) if duration else 24 * 3600.0
    split_active = split and len(cfg.audio.record_channels) > 1

    def _on_save(wav_path, _meta) -> None:
        lib.append_sidecar(wav_path)
        if split_active:
            lib.split_clip(wav_path)

    n = record_auto(
        cfg,
        out,
        class_name=target_label,
        label=target_label,
        category=RAW_CATEGORY,
        project_id=project_id,
        density=density,
        review_status="accepted",
        duration_s=duration_s,
        mic_position=mic_position,
        location=location,
        max_clips=(num if num > 0 else None),
        description=description,
        tags=parsed_tags,
        on_save=_on_save,
    )
    console.print(f"[green]saved {n} clip(s) -> {out}[/green]")


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
    label: str = typer.Option(..., "--class", help="Library sound label."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Optional project association."),
    density: str = typer.Option("normal", "--density", help="empty | quiet | normal | busy | peak"),
    move: bool = typer.Option(False, "--move", help="Move files instead of copying them."),
    tags: str = typer.Option(
        "", "--tags", "-tg",
        help='Comma-separated tags saved in each sidecar, e.g. "legacy,old-rig".',
    ),
) -> None:
    """Import existing WAV files into data_library/sounds/<label>/."""
    _setup_logging()
    from sed.data.library import Library
    if density not in _DENSITIES:
        raise typer.BadParameter(
            f"--density must be one of: {', '.join(_DENSITIES)}"
        )
    parsed_tags = _parse_tags(tags)
    lib = Library()
    n = 0
    for wav in sorted(Path(src_dir).rglob("*.wav")):
        lib.import_clip(
            wav,
            label=label,
            project_id=project or "_library",
            density=density,
            move=move,
            tags=parsed_tags,
        )
        n += 1
    console.print(f"[green]imported {n} clips[/green]")


@library_app.command("slice")
def library_slice(
    src: Path = typer.Argument(..., exists=True, help="Long WAV file, or a directory of WAVs."),
    label: str = typer.Option(..., "--class", help="Library sound label."),
    seconds: float = typer.Option(1.0, "--seconds", "-s", help="Slice length; match features.window_seconds."),
    hop: Optional[float] = typer.Option(None, "--hop", help="Step between slice starts. Default: --seconds (no overlap)."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Optional project association."),
    density: str = typer.Option("normal", "--density", help="empty | quiet | normal | busy | peak"),
    tags: str = typer.Option(
        "", "--tags", "-tg",
        help='Comma-separated tags saved in each sidecar, e.g. "ambience,bar-night".',
    ),
) -> None:
    """Slice long recordings (e.g. continuous ambience beds) into window-length clips.

    Every slice is a full model window, so training never zero-pads them. Keep
    the source file as well — threshold calibration consumes long files directly.
    """
    _setup_logging()
    from sed.data.library import Library
    if density not in _DENSITIES:
        raise typer.BadParameter(
            f"--density must be one of: {', '.join(_DENSITIES)}"
        )
    parsed_tags = _parse_tags(tags)
    sources = sorted(src.rglob("*.wav")) if src.is_dir() else [src]
    if not sources:
        raise typer.BadParameter(f"No WAV files found under {src}")
    lib = Library()
    total = 0
    for wav in sources:
        clips = lib.slice_file(
            wav,
            label=label,
            seconds=seconds,
            hop_seconds=hop,
            project_id=project or "_library",
            density=density,
            tags=parsed_tags,
        )
        total += len(clips)
        console.print(f"  {wav.name}: [cyan]{len(clips)}[/cyan] slice(s)")
    console.print(f"[green]sliced {total} clip(s) -> {lib.raw_dir(label)}[/green]")


@library_app.command("rebuild-manifest")
def library_rebuild_manifest() -> None:
    """Rebuild data_library/manifest.jsonl from WAV sidecars."""
    _setup_logging()
    from sed.data.library import Library
    clips = Library().rebuild_manifest()
    console.print(f"[green]indexed {len(clips)} clip(s)[/green]")


@library_app.command("migrate-legacy")
def library_migrate_legacy(
    mode: str = typer.Option(
        "copy",
        "--mode",
        help="copy or move legacy WAVs into sounds/.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would migrate without writing.",
    ),
) -> None:
    """Copy or move data_library/positives|negatives into data_library/sounds."""
    _setup_logging()
    from sed.data.library import Library

    if mode not in {"copy", "move"}:
        raise typer.BadParameter("--mode must be copy or move")
    result = Library().migrate_legacy(mode=mode, dry_run=dry_run)
    action = "would migrate" if dry_run else "migrated"
    console.print(f"[green]{action} {result['count']} legacy clip(s)[/green]")
    if result["items"]:
        table = Table(title="Legacy library migration", header_style="bold")
        table.add_column("label")
        table.add_column("source")
        table.add_column("dest")
        for item in result["items"][:20]:
            table.add_row(item["label"], item["source"], item["dest"])
        console.print(table)
        if len(result["items"]) > 20:
            console.print(f"[dim]... {len(result['items']) - 20} more[/dim]")


@library_app.command("split-channels")
def library_split_channels(
    uuid: str | None = typer.Option(
        None, "--uuid", help="Split a single clip by its library uuid.",
    ),
    label: str | None = typer.Option(
        None, "--class", "-c", help="Split every multichannel clip under this label.",
    ),
    all_clips: bool = typer.Option(
        False, "--all", help="Split every multichannel clip in the library.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-split even if per-channel clips already exist.",
    ),
) -> None:
    """Split multichannel recordings into per-channel mono library clips.

    The multichannel originals are kept; each channel is written as an
    independent mono clip under sounds/<label>/<label>_ch<c>/, grouped back to
    its original via the sidecar parent_uuid so you can review, listen to, and
    select each microphone separately.
    """
    _setup_logging()
    from sed.data.library import RAW_CATEGORY, Library

    if sum(bool(x) for x in (uuid, label, all_clips)) != 1:
        raise typer.BadParameter("Pass exactly one of --uuid, --class, or --all.")

    lib = Library()
    clips = lib.all_clips(rebuild_if_empty=True)

    def _is_multichannel_original(c) -> bool:
        return (
            c.category == RAW_CATEGORY
            and c.clip_kind != "channel_split"
            and int(c.channels or 1) > 1
        )

    if uuid:
        sources = [c for c in clips if c.uuid == uuid]
        if not sources:
            raise typer.BadParameter(f"No library clip with uuid {uuid!r}.")
    elif label:
        sources = [c for c in clips if c.label == label and _is_multichannel_original(c)]
    else:
        sources = [c for c in clips if _is_multichannel_original(c)]

    if not sources:
        console.print("[yellow]No multichannel clips to split.[/yellow]")
        return

    console.print(f"[cyan]Splitting {len(sources)} multichannel recording(s)...[/cyan]")

    total_files = 0
    split_count = 0
    skipped = 0
    missing = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Splitting channels[/cyan]", total=len(sources))
        for clip in sources:
            wav = clip.absolute_path(lib.root)
            display = f"{clip.label}/{Path(clip.relpath).name}"
            progress.update(
                task_id,
                description=f"[cyan]Splitting[/cyan] {escape(display)}",
            )
            if not wav.exists():
                missing += 1
                progress.advance(task_id)
                continue
            created = lib.split_clip(wav, force=force)
            if created:
                split_count += 1
                total_files += len(created)
            else:
                skipped += 1
            progress.advance(task_id)

    console.print("[dim]Rebuilding library manifest...[/dim]")
    lib.rebuild_manifest()
    summary = f"[green]split {split_count} recording(s) into {total_files} mono clip(s)[/green]"
    if skipped:
        summary += f" [dim]({skipped} already split - use --force to redo)[/dim]"
    if missing:
        summary += f" [yellow]({missing} source file(s) missing)[/yellow]"
    console.print(summary)


@pretraining_app.command("import")
def pretraining_import(
    source: Path = typer.Option(
        ..., "--source", "-s", exists=True, file_okay=False, dir_okay=True,
        help="Folder of pretraining WAVs, organised as <class>/.../*.wav.",
    ),
    move: bool = typer.Option(
        False, "--move", help="Move files instead of copying them.",
    ),
) -> None:
    """Copy a pretraining corpus into data_library/pretraining/, preserving the
    class folder structure, then (re)build the lightweight index.

    Only ``*.wav`` files are taken — any sidecar images/other files are ignored.
    The corpus is kept out of the main library manifest, so its sidecar-less
    clips are never touched by the library rebuild/prune logic.
    """
    _setup_logging()
    import shutil

    from sed.data.pretraining import PretrainIndex, pretraining_root

    source = Path(source)
    dest_root = pretraining_root()
    wavs = sorted(source.rglob("*.wav"))
    if not wavs:
        raise typer.BadParameter(f"No WAV files found under {source}")

    copied = 0
    skipped = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        verb = "Moving" if move else "Copying"
        task_id = progress.add_task(f"[cyan]{verb} clips[/cyan]", total=len(wavs))
        for wav in wavs:
            try:
                rel = wav.relative_to(source)
            except ValueError:
                rel = Path(wav.name)
            dest = dest_root / rel
            progress.update(task_id, description=f"[cyan]{verb}[/cyan] {escape(str(rel))}")
            if dest.exists():
                skipped += 1
                progress.advance(task_id)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(wav), str(dest))
            else:
                shutil.copy2(str(wav), str(dest))
            copied += 1
            progress.advance(task_id)

    console.print("[dim]Building pretraining index...[/dim]")
    rows = PretrainIndex().rebuild()
    summary = f"[green]imported {copied} clip(s)[/green]"
    if skipped:
        summary += f" [dim]({skipped} already present)[/dim]"
    summary += f" -> indexed {len(rows)} clip(s) in {dest_root}"
    console.print(summary)


@pretraining_app.command("scan")
def pretraining_scan() -> None:
    """Rebuild data_library/pretraining/_index.jsonl from disk."""
    _setup_logging()
    from sed.data.pretraining import PretrainIndex

    rows = PretrainIndex().rebuild()
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    console.print(f"[green]indexed {len(rows)} clip(s) across {len(counts)} class(es)[/green]")
    table = Table(title="Pretraining classes", header_style="bold")
    table.add_column("class")
    table.add_column("clips", justify="right")
    for label in sorted(counts):
        table.add_row(label, str(counts[label]))
    console.print(table)


@pretraining_app.command("previews")
def pretraining_previews(
    label: Optional[str] = typer.Option(
        None, "--class", "-c", help="Only generate previews for this class folder.",
    ),
) -> None:
    """Pre-generate shared spectrogram-cache previews for the pretraining corpus.

    Uses the same bounded, signature-aware cache engine as ``inspect --serve``.
    The command remains useful for preparing a corpus before interactive work.
    """
    _setup_logging()
    from concurrent.futures import FIRST_COMPLETED, wait
    from sed.config import load_config
    from sed.data.preview_cache import PreviewCache, PreviewWorkerPool, preview_features_from_config
    from sed.data.pretraining import PretrainIndex

    index = PretrainIndex()
    rows = (
        index.rows_for_label(label) if label else index.all_rows()
    )
    if not rows:
        console.print("[yellow]No pretraining clips found.[/yellow]")
        return

    cache = PreviewCache()
    workers = PreviewWorkerPool(cache, workers=2)
    features = preview_features_from_config(load_config())
    made = 0
    skipped = 0
    failed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Generating previews[/cyan]", total=len(rows))
        pending: dict[object, Path] = {}
        next_index = 0
        while next_index < len(rows) or pending:
            while next_index < len(rows) and len(pending) < 12:
                wav = index.abs_path(rows[next_index])
                next_index += 1
                migrated = cache.migrate_legacy(wav, features, label="pretraining")
                if migrated is not None:
                    skipped += 1
                    progress.advance(task_id)
                    continue
                pending[workers.submit(wav, features, label="pretraining", priority=10)] = wav
            if not pending:
                continue
            done, _waiting = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                wav = pending.pop(future)
                progress.update(task_id, description=f"[cyan]Preview[/cyan] {escape(wav.name)}")
                try:
                    result = future.result()
                    if result.created:
                        made += 1
                    else:
                        skipped += 1
                except Exception:  # noqa: BLE001
                    failed += 1
                progress.advance(task_id)

    summary = f"[green]generated {made} preview(s)[/green]"
    if skipped:
        summary += f" [dim]({skipped} already cached)[/dim]"
    if failed:
        summary += f" [yellow]({failed} failed)[/yellow]"
    console.print(summary)


@pretraining_app.command("clean")
def pretraining_clean(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d", help="Print what would be deleted without actually deleting it."
    ),
) -> None:
    """Remove non-WAV clutter (images, sidecars, etc.) from data_library/pretraining/."""
    _setup_logging()
    from sed.data.pretraining import pretraining_root, _INDEX_NAME

    root = pretraining_root()
    if not root.exists():
        console.print(f"[yellow]Pretraining root {root} does not exist.[/yellow]")
        return

    clutter_files: list[Path] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Scanning pretraining folder for clutter...[/cyan]", total=None)
        for p in root.rglob("*"):
            if p.is_file() and p.name != _INDEX_NAME and p.suffix.lower() != ".wav":
                clutter_files.append(p)
        progress.update(task_id, completed=True, description="[cyan]Scan complete[/cyan]")

    if clutter_files:
        if dry_run:
            console.print(f"[yellow]Dry-run: the following {len(clutter_files)} file(s) would be deleted:[/yellow]")
            for p in sorted(clutter_files):
                console.print(f"  {p.relative_to(root)}")
            return

        removed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task_id_del = progress.add_task("[red]Deleting clutter[/red]", total=len(clutter_files))
            for p in sorted(clutter_files):
                try:
                    p.unlink(missing_ok=True)
                    removed += 1
                except Exception as exc:
                    console.print(f"[yellow]Failed to delete {p.relative_to(root)}: {exc}[/yellow]")
                progress.advance(task_id_del)
        console.print(f"[green]Successfully removed {removed} clutter file(s) from {root}[/green]")
    else:
        if not dry_run:
            console.print("[green]No clutter files found.[/green]")

    if not dry_run:
        # Clean up empty directories bottom-up
        empty_dirs = []
        for p in sorted(root.rglob("*"), key=lambda x: -len(x.parts)):
            if p.is_dir() and not any(p.iterdir()):
                empty_dirs.append(p)

        if empty_dirs:
            for p in empty_dirs:
                try:
                    p.rmdir()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@dataset_app.command("inspect")
def dataset_inspect_cmd(
    project: str = typer.Option(...),
    open_browser: bool = typer.Option(False, "--open",
                  help="Open the report in your default browser (static mode)."),
    serve: bool = typer.Option(False, "--serve",
                  help="Run an interactive local server: tick clips and press "
                       "'Delete selected' to remove unfit samples from disk. "
                       "Auto-opens the browser. This is the default unless "
                       "--static is passed."),
    static: bool = typer.Option(
        False,
        "--static",
        help="Write a static HTML report instead of starting the inspect server.",
    ),
    port: int = typer.Option(8765, help="Port for --serve mode."),
    max_thumbs: Optional[int] = typer.Option(
        None, help="Max thumbnails per class in the report. Overrides "
                   "[inspect] max_thumbs_per_class. In static mode, 0 = unlimited; "
                   "in serve mode, 0 uses the safe interactive cap.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run",
        help="Frozen-test prediction run to inspect. Defaults to selected run.",
    ),
) -> None:
    """Review a project's recordings + augmented WAVs.

    Writes `dataset/inspect/inspection.html`.

    Server mode is the default. Pass --static to write a self-contained HTML
    report instead.
    """
    _setup_logging()
    from sed.config import load_config
    cfg = load_config(project)
    effective_max = max_thumbs if max_thumbs is not None else cfg.inspect.max_thumbs_per_class
    if effective_max == 0 and static and not serve:
        effective_max = 10_000_000
    elif effective_max == 0:
        effective_max = cfg.inspect.max_thumbs_per_class or 200
    if serve or not static:
        from sed.data.inspect_server import serve as run_server
        run_server(cfg, project, port=port,
                   max_thumbs_per_class=effective_max,
                   run_id=run_id)
        return
    from sed.data.inspect import inspect_project
    out = inspect_project(
        cfg, project, max_thumbs_per_class=effective_max,
        interactive=False,
        run_id=run_id,
    )
    console.print(f"[green]report → {out}[/green]")
    if open_browser:
        import webbrowser
        webbrowser.open(out.as_uri())


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


@dataset_app.command("compose")
def dataset_compose_cmd(
    project: str = typer.Option(...),
    confirm_frozen_test_change: bool = typer.Option(
        False,
        "--confirm-frozen-test-change",
        help="Allow compose to replace or clear an existing non-empty frozen test set.",
    ),
    build_test_split: bool = typer.Option(
        False,
        "--build-test-split",
        help="Explicitly (re)carve the split-derived frozen test slice. Routine "
             "compose never creates or resizes it — only this flag does.",
    ),
) -> None:
    """Build train/val/test from data_library and project label mappings."""
    _setup_logging()
    from sed.config import load_config
    from sed.data.compose import compose_dataset
    cfg = load_config(project)
    summary = compose_dataset(
        cfg,
        project,
        confirm_frozen_test_change=confirm_frozen_test_change,
        build_split_test=build_test_split,
    )
    if summary.get("needs_confirmation"):
        old = summary.get("old_frozen_test", {})
        new = summary.get("new_frozen_test", {})
        console.print("[yellow]Frozen test set would change; dataset was not rewritten.[/yellow]")
        console.print(
            f"  old: {old.get('total', 0)} clips {old.get('hash', '')}\n"
            f"  new: {new.get('total', 0)} clips {new.get('hash', '')}"
        )
        console.print(
            "Re-run with [bold]--confirm-frozen-test-change[/bold] to apply this change."
        )
        raise typer.Exit(2)

    # Rendered the same way the check tables are — same header_style, same
    # row shape — so the terminal output stays consistent across commands.
    splits = summary["splits"]
    by_origin = summary["by_origin"]
    frozen = summary["frozen_test"]
    table = Table(title=f"Dataset — {project}", header_style="bold")
    table.add_column("class", style="bold")
    table.add_column("sounds", justify="right")
    table.add_column("derived", justify="right")
    table.add_column("mined", justify="right")
    table.add_column("train+val", justify="right")
    table.add_column("test (frozen)", justify="right")
    for cls, c in by_origin.items():
        test_n = frozen["by_class"].get(cls, 0)
        table.add_row(cls,
                      str(c.get("sounds", 0)), str(c.get("derived", 0)),
                      str(c.get("mined", 0)), str(c["total"]), str(test_n))
    console.print(table)

    footer = (
        f"  [bold]train+val pool[/bold] {splits['train']}  ·  "
        f"[bold]frozen test[/bold] {splits['test']}"
        "  [dim](train/val split happens at train time)[/dim]"
    )
    if frozen["present"]:
        counts = frozen.get("source_counts", {})
        footer += (
            "  [dim](test = "
            f"{counts.get('selected_buckets', 0)} selected + "
            f"{counts.get('split_derived', 0)} split-derived)[/dim]"
        )
    else:
        footer += "  [yellow](no frozen test — select frozen buckets or run with --build-test-split)[/yellow]"
    console.print(footer)


@dataset_app.command("augment")
def dataset_augment_cmd(
    project: str = typer.Option(..., help="Project ID."),
    seed: Optional[int] = typer.Option(None, help="Override cfg.augmentation.seed."),
    prefix: Optional[str] = typer.Option(
        None,
        "--prefix",
        help="Readable run prefix for generated samples (default: timestamp + seed).",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Clear the selected augmented destination before writing this batch.",
    ),
    frozen_test: bool = typer.Option(
        False,
        "--frozen-test",
        "--frosen-test",
        help=(
            "Use the composition's frozen-test source rules as bases and write derived "
            "test clips under data_library/derived/. The misspelled alias is accepted too."
        ),
    ),
) -> None:
    """Materialise overlay-augmented WAV samples for human review.

    Uses augmentation defaults from the config schema plus Inspect-saved recipes,
    optionally pitch-shifts each base clip, overlays a negative clip at the
    configured SNR, and writes tagged WAVs plus sidecars to:

        data_library/derived/<project>/trainval/<class>/<prefix>_aug_<class>_NNNN.wav
        data_library/derived/<project>/trainval/_negative/<prefix>_aug_negative_NNNN.wav

    With --frozen-test, bases come from frozen-test composition rules and outputs go to
    data_library/derived/<project>/test/<class>/.

    Re-run anytime; batches are kept side by side unless --replace is used.
    Inspect normal batches in the Augmented tab; inspect frozen-test batches in
    the Frozen test tab after `sed dataset compose`.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.data.augment import materialize_augmented
    cfg = load_config(project)
    counts = materialize_augmented(
        cfg,
        project,
        seed=seed,
        run_prefix=prefix,
        replace=replace,
        frozen_test=frozen_test,
    )
    if not counts:
        console.print(
            "[yellow]no augmented samples written — check positives exist "
            "and overlay augmentation is enabled[/yellow]"
        )
        return
    grand_total = sum(sum(tc.values()) for tc in counts.values())
    target = "frozen-test" if frozen_test else "training"
    console.print(
        f"[green]wrote {grand_total} {target} augmented clips across {len(counts)} bucket(s)[/green]"
    )
    for bucket, tc in counts.items():
        parts = "  ".join(f"{t}:{n}" for t, n in sorted(tc.items()))
        console.print(f"  {bucket}/   {parts}")


# ---------------------------------------------------------------------------
# train / export / calibrate / inference
# ---------------------------------------------------------------------------


@app.command("train")
def train_cmd(
    project: str = typer.Option(...),
    description: Optional[str] = typer.Option(
        None, "--description", "-d",
        help="Free-form note for this model (e.g. 'round 2, mined hard negs'). "
             "Stored in the manifest and shown by `sed models compare`; a short slug "
             "is also appended to the model id.",
    ),
    no_test: bool = typer.Option(False, "--no-test",
                                  help="Skip the post-training frozen-test evaluation."),
    test_rule: Optional[List[str]] = typer.Option(
        None, "--test-rule",
        help="Frozen-test decision rule to score when training finishes, e.g. "
             "'fixed@0.90', 'argmax', 'deployed'. Repeatable; overrides "
             "training.auto_test_rules. Same rule ids `sed evaluate benchmark` writes.",
    ),
    no_export: bool = typer.Option(False, "--no-export",
                                    help="Skip ONNX + OpenVINO export."),
) -> None:
    """Fine-tune the model from projects/<id>/dataset/, then auto-test + export.

    Writes one self-contained bundle to projects/<id>/models/<model_id>/ and, on
    a clean export, makes it the active model. Switch later with `sed models select`.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.train.train import train_project
    cfg = load_config(project)
    if test_rule:
        cfg = cfg.model_copy(deep=True)
        cfg.training.auto_test_rules = list(test_rule)
        cfg.training = cfg.training.model_validate(cfg.training.model_dump())
    auto_test = None if not no_test else False
    auto_export = None if not no_export else False
    model_dir = train_project(
        cfg, project, description=description,
        auto_test=auto_test, auto_export=auto_export,
    )
    console.print(f"[green]model → {model_dir}[/green]")


# ---------------------------------------------------------------------------
# models — show / compare / select / delete trained models
# ---------------------------------------------------------------------------


def _model_id_arg(
    project: str, model_id: Optional[str], model_id_opt: Optional[str],
) -> str:
    """Resolve the model from a positional arg or --model-id option (one required)."""
    raw = model_id_opt or model_id
    if not raw:
        raise typer.BadParameter(
            "provide a model id as an argument or via --model-id "
            "(full id or #N handle from `sed models compare`)"
        )
    return _resolve_model_id(project, raw)


@models_app.command("show")
def models_show_cmd(
    project: str = typer.Option(...),
    model_id: Optional[str] = typer.Argument(None, help="Model id or #N handle."),
    model_id_opt: Optional[str] = typer.Option(
        None, "--model-id", "--model_id", help="Model id or #N handle."),
) -> None:
    """Pretty-print a single model's manifest."""
    _setup_logging()
    from sed.train import registry
    resolved = _model_id_arg(project, model_id, model_id_opt)
    m = registry.load_manifest(project, resolved)
    import json as _json
    console.print_json(_json.dumps(m, default=str))


@models_app.command("compare")
def models_compare_cmd(project: str = typer.Option(...)) -> None:
    """Side-by-side comparison across all models: training info + test metrics."""
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.train import registry
    items = registry.load_manifests(project)
    if not items:
        console.print(f"[yellow]no models yet for project '{project}'.[/yellow]")
        return
    layout = ProjectLayout(project)
    from sed.train.eval import read_frozen_test_evals

    # Frozen-test scores per run, keyed by decision rule (legacy manifests are
    # folded in by the shim), plus the union of rules across all runs.
    per_run_evals = [read_frozen_test_evals(m) for m in items]
    rule_ids: list[str] = []
    for evals in per_run_evals:
        for rid in evals:
            if rid not in rule_ids:
                rule_ids.append(rid)

    def _primary_eval(evals: dict) -> dict:
        """The block whose per-class breakdown represents this run."""
        return next(iter(evals.values()), {}) if evals else {}

    # Gather per-class union
    classes: list[str] = []
    seen = set()
    for evals in per_run_evals:
        for c in (_primary_eval(evals).get("per_class") or {}):
            if c not in seen:
                seen.add(c); classes.append(c)

    active = registry.get_active_model(project)
    console.print(f"[bold]Compare - {escape(project)}[/bold]")
    labels = _print_model_id_key(items, active=active)
    console.print()

    mids = [str(m.get("model_id") or "") for m in items]
    metric_rows: list[tuple[str, list[str]]] = []
    metric_rows.append(("created", [_model_created_label(m) for m in items]))
    metric_rows.append(("epochs", [_model_epoch_label(m) for m in items]))
    metric_rows.append(("IR", [_model_ir_label(layout, mid) for mid in mids]))
    metric_rows.append(("calibrated", [_model_calibrated_label(layout, mid) for mid in mids]))
    metric_rows.append((
        "val loss",
        [_fmt_loss((m.get("training") or {}).get("selected_val_loss")) for m in items],
    ))
    metric_rows.append((
        "val acc",
        [_fmt_percent((m.get("training") or {}).get("selected_val_accuracy")) for m in items],
    ))
    # Frozen-test scores, one row per decision rule any model carries — the
    # same vocabulary the inspect Train tab uses.
    for rule_id in rule_ids:
        label = next(
            (
                (evals[rule_id].get("rule") or {}).get("label") or rule_id
                for evals in per_run_evals if rule_id in evals
            ),
            rule_id,
        )
        metric_rows.append((
            f"F1 · {label}",
            [
                _fmt_percent((evals.get(rule_id) or {}).get("f1_macro_targets"))
                for evals in per_run_evals
            ],
        ))
    # AP is threshold-free: identical under every rule, so it gets one row.
    metric_rows.append((
        "AP (frozen test)",
        [
            _fmt_percent(next(
                (
                    b["ap_macro_targets"] for b in evals.values()
                    if b.get("ap_macro_targets") is not None
                ),
                None,
            ))
            for evals in per_run_evals
        ],
    ))
    for c in classes:
        metric_rows.append((
            f"F1 {c}",
            [
                _fmt_percent((_primary_eval(evals).get("per_class") or {}).get(c, {}).get("f1"))
                for evals in per_run_evals
            ],
        ))

    chunk_size = 6
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        table = Table(
            title=f"Scoreboard {labels[start]}-{labels[end - 1]}",
            header_style="bold",
            expand=False,
        )
        table.add_column("metric", no_wrap=True)
        for label in labels[start:end]:
            table.add_column(label, justify="right", no_wrap=True)
        for metric_name, values in metric_rows:
            table.add_row(metric_name, *values[start:end])
        console.print(table)

    descriptions = [
        (labels[idx], str(m.get("description") or "").strip())
        for idx, m in enumerate(items)
        if str(m.get("description") or "").strip()
    ]
    if descriptions:
        console.print("\n[bold]Descriptions[/bold]")
        for label, desc in descriptions:
            console.print(f"  [bold]{label}[/bold]  {escape(desc)}", soft_wrap=True)

    # Warn when models were scored on different test sets.
    hashes = {m.get("dataset", {}).get("test_set_hash", "") for m in items}
    hashes.discard("")
    if len(hashes) > 1:
        console.print(
            "[yellow]warning:[/yellow] models were scored on different test sets "
            "(test_set_hash differs). Re-evaluate all models with `sed evaluate "
            "--project <id> --all` to bring all metrics onto the current test."
        )
    frozen_hashes = {
        str(b.get("test_set_hash") or "")
        for evals in per_run_evals for b in evals.values()
    }
    frozen_hashes.discard("")
    if len(frozen_hashes) > 1:
        console.print(
            "[yellow]warning:[/yellow] frozen-test scores were produced on different "
            "test sets, so the rows above are not comparable. Re-run "
            "`sed evaluate benchmark --project <id>`."
        )


@models_app.command("select")
def models_select_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Argument(
        None, help="Model id or #N handle (see `sed models compare`)."),
    model_id_opt: Optional[str] = typer.Option(
        None, "--model-id", "--model_id", help="Model id or #N handle."),
) -> None:
    """Make a model the active one by setting [deploy] active_model in config.toml.

    On the edge the path-watcher restarts the service on config change; otherwise
    run `sudo systemctl restart sed` to apply.
    """
    _setup_logging()
    from sed.train import registry
    resolved = _model_id_arg(project, model_id, model_id_opt)
    registry.set_active_model(project, resolved)
    console.print(
        f"[green]active_model = {resolved}[/green] for project {project}. "
        "On the edge, restart the service to apply (`sudo systemctl restart sed`)."
    )


@models_app.command("delete")
def models_delete_cmd(
    project: str = typer.Option(...),
    model_id: Optional[str] = typer.Argument(
        None, help="Model id or #N handle (see `sed models compare`)."),
    model_id_opt: Optional[str] = typer.Option(
        None, "--model-id", "--model_id", help="Model id or #N handle."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Remove a model bundle permanently."""
    _setup_logging()
    from sed.train import registry
    resolved = _model_id_arg(project, model_id, model_id_opt)
    if not yes:
        typer.confirm(f"Delete model {resolved} for project {project}?", abort=True)
    registry.delete_model(project, resolved)
    console.print(f"[green]deleted {resolved}[/green]")


@models_app.command("backfill-stats")
def models_backfill_stats_cmd(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Single project; default: all projects."),
    force: bool = typer.Option(False, "--force", help="Recompute even when model_stats exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report without writing manifests."),
) -> None:
    """Fill manifest.model_stats (parameter count + input shape) for existing runs.

    New runs record these at train time; this walks older bundles, sums the
    checkpoint parameters on CPU and derives the input geometry from the
    bundle's config snapshot. Runs without a manifest (aborted) are skipped.
    """
    _setup_logging()
    from sed.config import list_projects
    from sed.train import registry

    projects = [project] if project else list_projects()
    for pid in projects:
        results = registry.backfill_model_stats(pid, force=force, dry_run=dry_run)
        if not results:
            continue
        table = Table(title=f"model_stats backfill — {pid}", header_style="bold", expand=False)
        table.add_column("run", no_wrap=True)
        table.add_column("status", no_wrap=True)
        table.add_column("params", justify="right", no_wrap=True)
        table.add_column("input shape", no_wrap=True)
        for rec in results:
            stats = rec.get("model_stats") or {}
            params = stats.get("total_params")
            shape = (
                f"{stats.get('input_channels')} × {stats.get('n_bands')} × {stats.get('n_frames')}"
                f" ({stats.get('feature_type')})" if stats.get("n_bands") else ""
            )
            table.add_row(
                rec["model_id"], rec["status"],
                f"{params / 1e6:.2f} M" if params else "",
                shape,
            )
        console.print(table)
    if dry_run:
        console.print("[yellow]dry run — nothing written[/yellow]")


@models_app.command("backfill-scores")
def models_backfill_scores_cmd(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Single project; default: all projects."),
    force: bool = typer.Option(False, "--force", help="Rewrite scores that already exist."),
) -> None:
    """Recover per-clip score stores for existing runs — no GPU, no re-inference.

    Runs scored before the score store existed already wrote every class
    probability into reports/<rule>_predictions.tsv. This lifts those columns
    into reports/scores/, which is what lets Runs & compare re-derive metrics at
    any confidence without running the model again.

    Caveat: the prediction table rounds probabilities to six decimals, so an AP
    recomputed from a backfilled store can differ in the last places from the
    one recorded at scoring time. Threshold-based metrics are unaffected —
    decision boundaries are far coarser than 1e-6.
    """
    _setup_logging()
    from sed.config import ProjectLayout, list_projects
    from sed.train import score_store

    projects = [project] if project else list_projects()
    for pid in projects:
        layout = ProjectLayout(pid)
        if not layout.models_dir.is_dir():
            continue
        rows = [
            score_store.backfill_run(run_dir, overwrite=force)
            for run_dir in sorted(layout.models_dir.iterdir())
            if run_dir.is_dir()
        ]
        if not rows:
            continue
        table = Table(title=f"score backfill — {pid}", header_style="bold", expand=False)
        table.add_column("run", no_wrap=True)
        table.add_column("scores", no_wrap=True)
        table.add_column("note")
        for rec in rows:
            table.add_row(rec["run"], rec.get("written") or "—", rec.get("reason") or "")
        console.print(table)
        done = sum(1 for r in rows if r.get("written"))
        console.print(f"{done}/{len(rows)} runs have a score store.")


@models_app.command("calibrate-thresholds")
def models_calibrate_thresholds_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Argument(
        None, help="Model id or #N handle (see `sed models compare`). Omit for the active model."),
    all_models: bool = typer.Option(
        False, "--all",
        help="Calibrate every trained model in the project."),
    mode: str = typer.Option(
        "val_f1", "--mode",
        help="val_f1 fits each class's max-F1 threshold; val_precision fits the "
             "lowest threshold reaching --target-precision, the same operating "
             "point for every model."),
    target_precision: float = typer.Option(
        0.90, "--target-precision",
        help="Shared per-class precision floor for --mode val_precision."),
    rescore: bool = typer.Option(
        True, "--rescore/--no-rescore",
        help="Re-run the frozen-test evaluation (checkpoint + exported IR) with the fitted thresholds."),
) -> None:
    """Fit per-class trigger thresholds on the run's val split and re-score it.

    Existing runs were evaluated at one-size config defaults, which penalizes
    models with soft/uncalibrated probability scales (e.g. mixup-trained BCE).
    This writes val-fitted thresholds into the bundle's runtime_settings.json —
    the overlay both the runtime and evaluation apply — then refreshes the
    test metrics so runs become comparable on their deployed decision layer.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.train.eval import CALIBRATION_MODES, calibrate_run_thresholds, evaluate_exported_run, evaluate_run
    from sed.train import registry

    if all_models and model_id is not None:
        raise typer.BadParameter("pass either a model id or --all, not both")
    if mode not in CALIBRATION_MODES:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(CALIBRATION_MODES)}")
    cfg = load_config(project)
    if all_models:
        ids = [str(m.get("model_id") or "") for m in registry.load_manifests(project) if m.get("model_id")]
        if not ids:
            console.print(f"[yellow]no models yet for project '{project}'.[/yellow]")
            return
    elif model_id is None:
        active = registry.get_active_model(project)
        if active is None:
            raise typer.BadParameter(
                "no model id provided and no active model exists; pass a model id or --all"
            )
        ids = [active]
    else:
        ids = [_model_id_arg(project, model_id, None)]

    failures: list[tuple[str, str]] = []
    for idx, resolved in enumerate(ids, start=1):
        if len(ids) > 1:
            console.print(f"\n[{idx}/{len(ids)}] [cyan]{escape(resolved)}[/cyan]")
        try:
            fitted = calibrate_run_thresholds(
                cfg, project, resolved,
                mode=mode, target_precision=target_precision,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((resolved, str(exc)))
            console.print(f"[red]calibration failed[/red] {escape(str(exc))}")
            continue
        if fitted is None:
            console.print("[yellow]calibration not applicable (no _negative class or no val split)[/yellow]")
            continue
        console.print("[green]calibrated thresholds:[/green] " +
                      ", ".join(f"{k}={v:.3f}" for k, v in fitted.items()))
        if not rescore:
            continue
        metrics = evaluate_run(cfg, project, resolved, use_model_dataset=True)
        console.print(
            f"checkpoint: F1(targets)={metrics['f1_macro_targets']:.4f} "
            f"AP(targets)={metrics['ap_macro_targets']:.4f} acc={metrics['accuracy']:.4f}"
        )
        try:
            exported = evaluate_exported_run(cfg, project, resolved, use_model_dataset=True)
            console.print(
                f"exported ({exported['model_file']}): F1(targets)={exported['f1_macro_targets']:.4f} "
                f"AP(targets)={exported['ap_macro_targets']:.4f} acc={exported['accuracy']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]exported rescore skipped: {exc}[/yellow]")
    if failures:
        console.print("\n[red]failed models:[/red]")
        for mid, message in failures:
            console.print(f"  [cyan]{escape(mid)}[/cyan]  {escape(message)}", soft_wrap=True)
        raise typer.Exit(code=1)


@models_app.command("bn-recalibrate")
def models_bn_recalibrate_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Argument(
        None, help="Model id or #N handle (see `sed models compare`). Omit for the active model."),
    batches: int = typer.Option(
        50, "--batches", min=1,
        help="Training batches streamed through the frozen weights to re-estimate BN stats."),
    reevaluate: bool = typer.Option(
        True, "--reevaluate/--no-reevaluate",
        help="Re-run the frozen-test evaluation with the recalibrated checkpoint."),
    reexport: bool = typer.Option(
        True, "--reexport/--no-reexport",
        help="Re-export ONNX + OpenVINO from the recalibrated checkpoint (only if the run was exported)."),
) -> None:
    """Re-estimate BatchNorm running statistics for an existing run.

    Streams clean training batches through the frozen weights with only BN
    layers in train mode, writes `<checkpoint>_bnrecal.ckpt`, points the
    manifest at it, and optionally re-evaluates/re-exports so the deployed
    artifacts carry the refreshed statistics. Useful after deep freezing,
    mixup, or small-batch fine-tunes. Models without BatchNorm are a no-op.
    """
    _setup_logging()
    from sed.config import ProjectLayout, load_config, load_config_snapshot
    from sed.train import registry
    from sed.train.bn_recalibrate import recalibrate_batchnorm

    cfg = load_config(project)
    if model_id is None:
        active = registry.get_active_model(project)
        if active is None:
            raise typer.BadParameter("no model id provided and no active model exists")
        resolved = active
    else:
        resolved = _model_id_arg(project, model_id, None)

    layout = ProjectLayout(project)
    model_dir = layout.model_dir(resolved)
    snapshot = model_dir / "training_config.json"
    run_cfg = load_config_snapshot(snapshot) if snapshot.exists() else cfg
    ckpt_path = registry.get_checkpoint(project, resolved)
    if ckpt_path.stem.endswith("_bnrecal"):
        console.print("[yellow]checkpoint is already BN-recalibrated; recalibrating from it again.[/yellow]")

    info = recalibrate_batchnorm(
        run_cfg, project, checkpoint_path=ckpt_path, num_batches=batches,
    )
    if not info.get("checkpoint"):
        console.print(f"[yellow]{run_cfg.model.backbone} has no BatchNorm layers — nothing to recalibrate.[/yellow]")
        return
    new_ckpt = Path(info["checkpoint"])
    console.print(
        f"[green]recalibrated[/green] {info['bn_layers']} BN layers on "
        f"{info['batches']} batches ({info['samples']} clips) → {new_ckpt.name}"
    )

    manifest = registry.load_manifest(project, resolved)
    manifest.setdefault("files", {})["checkpoint"] = new_ckpt.name
    training_block = manifest.setdefault("training", {})
    training_block["best_checkpoint_name"] = new_ckpt.name
    training_block["bn_recalibrated"] = {
        "bn_layers": info["bn_layers"], "batches": info["batches"],
        "samples": info["samples"], "checkpoint": new_ckpt.name,
    }
    registry.save_manifest(project, resolved, manifest)

    if reevaluate:
        from sed.train.eval import evaluate_run
        metrics = evaluate_run(cfg, project, resolved, use_model_dataset=True)
        console.print(
            f"checkpoint: F1(targets)={metrics['f1_macro_targets']:.4f} "
            f"AP(targets)={metrics['ap_macro_targets']:.4f} acc={metrics['accuracy']:.4f}"
        )
    if reexport and (model_dir / "model_fp16.xml").exists():
        from sed.export.openvino import export_project
        run_dataset = model_dir / "dataset"
        dataset_dir = run_dataset if (run_dataset / "train.csv").exists() else layout.dataset_dir
        export_info = export_project(
            run_cfg, project, ckpt_path=new_ckpt, out_dir=model_dir, dataset_dir=dataset_dir,
        )
        manifest = registry.load_manifest(project, resolved)
        manifest["exported"] = True
        manifest["export"] = {
            "source_checkpoint": new_ckpt.name,
            "active_precision": export_info.get("active_precision"),
            "active_target": export_info["active_target"],
        }
        registry.save_manifest(project, resolved, manifest)
        console.print(f"[green]re-exported[/green] → {export_info['active_target']}")


@models_app.command("set-precision")
def models_set_precision_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Argument(
        None, help="Model id or #N handle. Omit for the active model."),
    precision: str = typer.Option(
        "auto", "--precision",
        help="fp16 | int8 | auto (score both IRs on the frozen test and keep INT8 "
             "only while it holds recall at the false-alarm budget to within "
             "training.int8_max_regression of FP16)."),
    rescore: bool = typer.Option(
        True, "--rescore/--no-rescore",
        help="Re-score the newly deployed IR on the frozen test."),
) -> None:
    """Point active_openvino.xml at the FP16 or INT8 IR for an existing run.

    Some backbones (e.g. MobileNetV3's SE blocks) lose several points to
    post-training INT8 quantization while FP16 matches the torch checkpoint.
    `--precision auto` scores both IRs on the current frozen test and compares
    the recall each reaches inside the false-alarm budget — the same measurement
    the export runs, so a manual call and an automatic one cannot disagree.
    """
    _setup_logging()
    from sed.config import ProjectLayout, load_config
    from sed.export.openvino import set_active_precision
    from sed.train import registry

    choice = str(precision).lower()
    if choice not in ("auto", "fp16", "int8"):
        console.print("[red]--precision must be auto, fp16 or int8[/red]")
        raise typer.Exit(1)
    cfg = load_config(project)
    resolved = _model_id_arg(project, model_id, None)
    manifest = registry.load_manifest(project, resolved)

    layout = ProjectLayout(project)
    if choice == "auto":
        # Same measurement the export runs: score BOTH IRs on the frozen test
        # and compare recall at the false-alarm budget. Scores both here rather
        # than requiring the operator to have run two evaluations first — the
        # comparison is only meaningful when both sides used the same rule and
        # the same clips.
        from sed.train.eval import decide_ir_precision

        decision = decide_ir_precision(cfg, project, resolved)
        choice = decision["precision"]
        fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        console.print(
            f"auto: {decision['metric']} fp16={fmt(decision['fp16'])} "
            f"int8={fmt(decision['int8'])} → [bold]{choice}[/bold] ({decision['reason']})"
        )
        manifest = registry.load_manifest(project, resolved)
        manifest.setdefault("export", {})["precision_scores"] = {
            "metric": decision["metric"], "fp16": decision["fp16"],
            "int8": decision["int8"], "tolerance": decision["tolerance"],
        }
        decision_note = f"auto: {decision['reason']}"
    else:
        decision_note = f"manual: sed models set-precision {precision}"

    _active, target = set_active_precision(layout.model_dir(resolved), choice)
    manifest.setdefault("export", {})
    manifest["export"]["active_precision"] = choice
    manifest["export"]["active_target"] = target.name
    manifest["export"]["precision_decision"] = decision_note
    registry.save_manifest(project, resolved, manifest)
    console.print(f"[green]active_openvino.xml → {target.name}[/green] for {resolved}")

    if rescore:
        from sed.train.eval import evaluate_exported_run
        metrics = evaluate_exported_run(cfg, project, resolved, use_model_dataset=True)
        console.print(
            f"deployed ({metrics['model_file']}): F1(targets)={metrics['f1_macro_targets']:.4f} "
            f"AP(targets)={metrics['ap_macro_targets']:.4f} acc={metrics['accuracy']:.4f}"
        )


# ---------------------------------------------------------------------------
# evaluate — score against the frozen test set; audit derived clips
# ---------------------------------------------------------------------------


@evaluate_app.callback(invoke_without_command=True)
def evaluate_main(
    ctx: typer.Context,
    project: Optional[str] = typer.Option(None, help="Project ID."),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", "--model_id",
        help="Model id or #N handle to evaluate. Omit to use the active project model.",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        help=(
            "Override all positive-class decision thresholds. By default evaluation "
            "uses per-class thresholds from config, falling back to "
            "inference.default_threshold."
        ),
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Evaluate every trained model in the project, overwriting prior test reports.",
    ),
) -> None:
    """Score trained model(s) against the project's frozen test set.

    Run with no subcommand to regenerate the frozen-test report for the active
    model (or --model-id <id>, or --all). Reads the checkpoint and runs it over
    projects/<id>/dataset/test.csv (clips matched by split="test" rules),
    replacing into each model bundle:

        models/<model_id>/reports/test_metrics.json      (per-class AP/F1/P/R)
        models/<model_id>/reports/test_predictions.tsv   (one row per WAV)
        models/<model_id>/reports/confusion_matrix.{png,json}
        models/<model_id>/manifest.json   (test_metrics block + test_set_hash)

    Re-run with --all after the frozen test set changes, then `sed dataset
    compose` to refresh test.csv. Use `augmented` to audit derived clips.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not project:
        raise typer.BadParameter("--project is required")
    _setup_logging()
    from sed.config import load_config
    from sed.train import registry
    from sed.train.eval import evaluate_run
    if all_models and model_id is not None:
        raise typer.BadParameter("pass either --model-id or --all, not both")

    cfg = load_config(project)
    if all_models:
        items = registry.load_manifests(project)
        if not items:
            console.print(f"[yellow]no models yet for project '{project}'.[/yellow]")
            return
        model_ids = [str(m.get("model_id") or "") for m in items if m.get("model_id")]
        console.print(
            f"[bold]Evaluating {len(model_ids)} model(s) for {escape(project)}[/bold]"
        )
        console.print(
            "[yellow]Existing frozen-test reports in each model bundle will be overwritten.[/yellow]"
        )
        failures: list[tuple[str, str]] = []
        for idx, mid in enumerate(model_ids, start=1):
            console.print(f"\n[{idx}/{len(model_ids)}] [cyan]{escape(mid)}[/cyan]")
            try:
                metrics = evaluate_run(cfg, project, mid, threshold=threshold)
            except Exception as exc:  # noqa: BLE001 - keep evaluating remaining models.
                failures.append((mid, str(exc)))
                console.print(f"[red]failed[/red] {escape(mid)}: {escape(str(exc))}")
                continue
            console.print(_evaluation_result_line(mid, metrics))
        succeeded = len(model_ids) - len(failures)
        console.print(f"\n[green]evaluated {succeeded}/{len(model_ids)} model(s)[/green]")
        if failures:
            console.print("[red]failed models:[/red]")
            for mid, message in failures:
                console.print(f"  [cyan]{escape(mid)}[/cyan]  {escape(message)}", soft_wrap=True)
            raise typer.Exit(code=1)
        return

    if model_id is None:
        resolved = registry.get_active_model(project)
        if resolved is None:
            raise typer.BadParameter(
                "no --model-id provided and no active model exists; pass a model id "
                "or run `sed models select --project <id> <model_id>`"
            )
    else:
        resolved = _resolve_model_id(project, model_id)
    metrics = evaluate_run(cfg, project, resolved, threshold=threshold)
    console.print(_evaluation_result_line(resolved, metrics))


@evaluate_app.command("benchmark")
def evaluate_benchmark_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", "--model_id",
        help="Optional single model id or #N handle. Default: every model.",
    ),
    rule: str = typer.Option(
        "fixed", "--rule",
        help="Decision rule: fixed | argmax | val-f1 | val-precision | deployed."),
    threshold: float = typer.Option(
        0.90, "--threshold", "-t",
        help="Rule parameter: the confidence gate for --rule fixed, the val "
             "precision floor for --rule val-precision. Ignored by other rules."),
    calibrated: bool = typer.Option(
        False, "--calibrated",
        help="Deprecated alias for --rule deployed."),
) -> None:
    """Score model(s) on the current frozen test under one decision rule.

    A rule is how probabilities become a pass/fail. `fixed` (the default)
    applies the same confidence gate to every model and ignores each bundle's
    runtime_settings.json — the like-for-like comparison. `argmax` drops
    thresholds entirely; `val-f1` / `val-precision` fit per-class thresholds on
    each run's own val split without saving them; `deployed` uses the bundle's
    own decision layer. Each rule is stored separately under `frozen_test` in
    the manifest, so rules accumulate instead of overwriting each other.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.train import registry
    from sed.train.eval import TestRule, evaluate_run_under_rule

    kind = "deployed" if calibrated else rule.strip().replace("-", "_")
    try:
        test_rule = TestRule(kind, threshold)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    cfg = load_config(project)
    if model_id is None:
        items = registry.load_manifests(project)
        model_ids = [str(m.get("model_id") or "") for m in items if m.get("model_id")]
        if not model_ids:
            console.print(f"[yellow]no models yet for project '{project}'.[/yellow]")
            return
    else:
        model_ids = [_resolve_model_id(project, model_id)]

    console.print(
        f"[bold]Scoring {len(model_ids)} model(s) for {escape(project)}[/bold] "
        f"(rule: {escape(test_rule.label)})"
    )
    table = Table(title=f"Frozen test — {test_rule.label}", header_style="bold", expand=False)
    table.add_column("model")
    table.add_column("F1", justify="right")
    table.add_column("AP", justify="right")
    table.add_column("acc", justify="right")
    table.add_column("n", justify="right")
    table.add_column("thresholds")

    failures: list[tuple[str, str]] = []
    for idx, mid in enumerate(model_ids, start=1):
        console.print(f"[{idx}/{len(model_ids)}] {escape(mid)}")
        try:
            metrics = evaluate_run_under_rule(cfg, project, mid, test_rule)
        except Exception as exc:  # noqa: BLE001
            failures.append((mid, str(exc)))
            table.add_row(mid, "failed", "", "", "", escape(str(exc)))
            continue
        table.add_row(
            mid,
            _fmt_percent(metrics.get("f1_macro_targets")),
            _fmt_percent(metrics.get("ap_macro_targets")),
            _fmt_percent(metrics.get("accuracy")),
            str(metrics.get("n_samples") or 0),
            str(metrics.get("thresholds_source") or ""),
        )
    console.print(table)
    if failures:
        console.print("[red]failed models:[/red]")
        for mid, message in failures:
            console.print(f"  [cyan]{escape(mid)}[/cyan]  {escape(message)}", soft_wrap=True)
        raise typer.Exit(code=1)


@evaluate_app.command("deployed")
def evaluate_deployed_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", "--model_id",
        help="Optional single model id or #N handle. Default: every exported model.",
    ),
    precision: Optional[List[str]] = typer.Option(
        None, "--precision",
        help="IR precision to score: fp16 | int8 | auto (the deployed one). "
             "Repeatable — each precision is stored separately."),
    rule: str = typer.Option(
        "deployed", "--rule",
        help="Decision rule: deployed (the run's own layer — what production "
             "delivers) | fixed | argmax (hold the decision fixed to isolate "
             "quantisation cost)."),
    threshold: float = typer.Option(
        0.90, "--threshold", "-t", help="Confidence gate for --rule fixed."),
) -> None:
    """Score the exported OpenVINO IR(s) of model(s) on the current frozen test.

    The checkpoint metrics describe what was trained; this is what production
    actually loads. Each precision × rule is stored separately under
    `deployed_test` in the manifest, so FP16 and INT8 can be compared side by
    side — which is what `sed models set-precision --precision auto` reads.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.train import registry
    from sed.train.eval import EXPORTED_RULE_KINDS, TestRule, evaluate_exported_run

    kind = rule.strip().replace("-", "_")
    if kind not in EXPORTED_RULE_KINDS:
        raise typer.BadParameter(
            f"--rule must be one of {', '.join(EXPORTED_RULE_KINDS)} for an exported IR"
        )
    test_rule = TestRule(kind, threshold)
    precisions = [p.strip().lower() for p in (precision or ["auto"])]
    for p in precisions:
        if p not in ("auto", "fp16", "int8"):
            raise typer.BadParameter("--precision must be auto, fp16 or int8")

    cfg = load_config(project)
    if model_id is None:
        items = registry.load_manifests(project)
        model_ids = [str(m.get("model_id") or "") for m in items
                     if m.get("model_id") and m.get("exported")]
        if not model_ids:
            console.print(f"[yellow]no exported models yet for project '{project}'.[/yellow]")
            return
    else:
        model_ids = [_resolve_model_id(project, model_id)]

    table = Table(title=f"Deployed IR — {test_rule.label}", header_style="bold", expand=False)
    for column, justify in (("model", "left"), ("precision", "left"), ("file", "left"),
                            ("F1", "right"), ("AP", "right"), ("acc", "right"), ("n", "right")):
        table.add_column(column, justify=justify)

    failures: list[tuple[str, str]] = []
    for mid in model_ids:
        for prec in precisions:
            try:
                metrics = evaluate_exported_run(
                    cfg, project, mid,
                    use_model_dataset=False, precision=prec, rule=test_rule,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((f"{mid} ({prec})", str(exc)))
                table.add_row(mid, prec, "failed", "", "", "", "")
                continue
            table.add_row(
                mid, str(metrics.get("precision") or prec), str(metrics.get("model_file") or ""),
                _fmt_percent(metrics.get("f1_macro_targets")),
                _fmt_percent(metrics.get("ap_macro_targets")),
                _fmt_percent(metrics.get("accuracy")),
                str(metrics.get("n_samples") or 0),
            )
    console.print(table)
    if failures:
        console.print("[red]failed:[/red]")
        for label, message in failures:
            console.print(f"  [cyan]{escape(label)}[/cyan]  {escape(message)}", soft_wrap=True)
        raise typer.Exit(code=1)


@evaluate_app.command("augmented")
def evaluate_augmented_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Argument(
        None,
        help="Model ID to audit with. Omit to use the active project model.",
    ),
    exported: bool = typer.Option(
        False, "--exported",
        help="Score the exported OpenVINO IR instead of the torch checkpoint.",
    ),
) -> None:
    """Audit augmented clips: which ones does a trained model recognize?

    Runs the model (same unsmoothed runtime gates as the frozen test) over every
    derived clip in data_library/derived/<project>/ and writes
    projects/<id>/dataset/inspect/augment_audit.json with per-clip predictions
    plus recognition rates per recipe and per mask-score / effective-SNR bin.
    The Inspect UI's Augmented tab picks the report up on the next refresh
    ("model misses" filter).

    Clips the model trained on are flagged seen_in_training — their numbers are
    optimistic; trust the unseen rates (frozen-test augmentations or clips
    generated after training).
    """
    _setup_logging()
    from sed.config import ProjectLayout, load_config
    from sed.train.eval import audit_augmented

    cfg = load_config(project)
    report = audit_augmented(cfg, project, model_id, exported=exported)
    summary = report["summary"]
    overall = summary["overall"]["all"]
    console.print(
        f"[green]{report['model_id']}[/green]  clips={report['n_clips']}  "
        f"recognized={overall['recognized']}/{overall['total']} "
        f"({100.0 * overall['rate']:.1f}%)"
        + (
            f"  unseen={overall['unseen_recognized']}/{overall['unseen_total']} "
            f"({100.0 * overall['unseen_rate']:.1f}%)"
            if overall.get("unseen_total") else ""
        )
    )

    def _print_table(title: str, table: dict) -> None:
        t = Table(title=title)
        t.add_column(title.split(" by ")[-1])
        t.add_column("recognized", justify="right")
        t.add_column("rate", justify="right")
        t.add_column("unseen rate", justify="right")
        for key in sorted(table):
            entry = table[key]
            unseen = (
                f"{100.0 * entry['unseen_rate']:.1f}%"
                if entry.get("unseen_rate") is not None else "—"
            )
            t.add_row(
                str(key),
                f"{entry['recognized']}/{entry['total']}",
                f"{100.0 * entry['rate']:.1f}%",
                unseen,
            )
        console.print(t)

    _print_table("recognition by recipe", summary["by_recipe"])
    _print_table("recognition by mean-mask dB bin", summary["by_mask_bin"])
    _print_table("recognition by effective-SNR dB bin", summary["by_effective_snr_bin"])
    if report.get("skipped_labels"):
        console.print(
            f"[yellow]skipped labels not known to the model: "
            f"{', '.join(report['skipped_labels'])}[/yellow]"
        )
    console.print(
        f"report: {ProjectLayout(project).dataset_inspect_dir / 'augment_audit.json'}"
    )


@evaluate_app.command("explain")
def evaluate_explain_cmd(
    project: str = typer.Option(..., help="Project ID."),
    model_id: Optional[str] = typer.Argument(
        None,
        help="Model id or #N handle to analyze. Omit to use the active project model.",
    ),
    scope: str = typer.Option(
        "frozen,mined",
        "--scope",
        help="Comma-separated review scopes: frozen, mined.",
    ),
    method: str = typer.Option(
        "auto",
        "--method",
        help="auto, cam, gradcam, frame, or occlusion.",
    ),
    classes: str = typer.Option(
        "top",
        "--classes",
        help="top, predicted, true, all, or class:<name>.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Recompute cached clip explanations.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Optional clip limit for a quick analysis pass.",
    ),
) -> None:
    """Generate cached prediction-localization artifacts for review tabs."""
    _setup_logging()
    from sed.config import ProjectLayout, load_config
    from sed.train import registry
    from sed.train.explain import generate_explanations, explanation_report_dir

    cfg = load_config(project)
    if model_id is None:
        resolved = registry.get_active_model(project)
        if resolved is None:
            raise typer.BadParameter(
                "no model id provided and no active model exists; pass a model id "
                "or run `sed models select --project <id> <model_id>`"
            )
    else:
        resolved = _resolve_model_id(project, model_id)

    scopes = [s.strip().lower() for s in str(scope or "").split(",") if s.strip()]
    if not scopes:
        raise typer.BadParameter("--scope must include frozen and/or mined")
    layout = ProjectLayout(project)
    for sc in scopes:
        summary = generate_explanations(
            cfg,
            project,
            resolved,
            scope=sc,
            method=method,
            classes=classes,
            force=force,
            limit=limit,
        )
        report_dir = explanation_report_dir(layout.model_dir(resolved), sc)
        console.print(
            f"[green]{escape(resolved)}[/green]  scope={escape(sc)}  "
            f"clips={summary.get('n_clips', 0)}  maps={summary.get('n_maps', 0)}  "
            f"methods={escape(str(summary.get('methods', {})))}"
        )
        console.print(f"report: {report_dir / 'summary.json'}")


@evaluate_app.command("list")
def evaluate_list_cmd(project: str = typer.Option(..., help="Project ID.")) -> None:
    """Show the currently composed frozen-test clips for this project."""
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.train.testset_health import read_class_counts

    test_csv = ProjectLayout(project).dataset_dir / "test.csv"
    table = Table(title=f"Frozen test set — {project}")
    table.add_column("destination"); table.add_column("label")
    table.add_column("clips", justify="right")
    counts = read_class_counts(test_csv) if test_csv.exists() else {}
    if not counts:
        table.add_row("frozen test", "—", "0")
    for label, count in sorted(counts.items()):
        table.add_row("frozen test", label, str(count))
    console.print(table)


@app.command("calibrate")
def calibrate_cmd(
    project: str = typer.Option(...),
    target_per_1000: float = typer.Option(
        1.0, "--target-per-1000", "--target-fp-per-hour",
        help="Tolerated false fires per 1000 negative clips."),
    negative_dir: Optional[Path] = typer.Option(None,
                  help="Override path; defaults to library clips assigned to _negative."),
    negative_source: str = typer.Option("venue",
                  help="venue | library. Choose project negatives or shared library negatives."),
) -> None:
    """Calibrate per-class detection thresholds against negative/background clips."""
    _setup_logging()
    from sed.config import load_config_for_inference
    from sed.runtime.calibrate import calibrate
    if negative_source not in ("venue", "library"):
        raise typer.BadParameter(
            "negative_source must be one of: venue, library"
        )
    cfg = load_config_for_inference(project)
    result = calibrate(
        cfg, project,
        negative_dir=negative_dir,
        target_per_1000=target_per_1000,
        negative_source=negative_source,
    )
    console.print(result["thresholds"])
    console.print(
        f"[green]achieved[/green] {result['system_per_1000']:.2f} false fires per 1000 clips "
        f"(target ≤ {result['target_per_1000']:.2f}) over "
        f"{result['clips_scored']} {negative_source} negative clip(s)"
    )


@app.command("run", cls=InferenceCommand, hidden=True)
@app.command("inference", cls=InferenceCommand)
def inference_cmd(
    project: str = typer.Option(..., help="Project/venue id to load from projects/<id>."),
    device: Optional[str] = typer.Option(
        None, "--device", callback=_device_callback, is_eager=True,
        help='Audio input device: name substring or index (e.g. "Focusrite" or 1). '
             'Use a bare "--device" with no value to list available devices.'),
    no_artnet: bool = typer.Option(
        False, "--no-artnet",
        help="Do not send ArtNet/DMX output; useful for terminal debugging."),
    target: Optional[list[str]] = typer.Option(None, "--target", help=
        "Comma-separated active trigger classes, e.g. cheers_loud,cheers_quiet. "
        "Omit to use enabled target blocks from projects/<id>/config.toml."),
    verbose: Optional[str] = typer.Option(
        None, "--verbose", callback=_verbose_callback, metavar="LEVEL",
        help=(
            "Override deploy.verbose. Output verbosity: 'none' (quiet), 'info' (log probabilities, "
            "mic levels, latency, triggers), or 'live' (info plus the Rich live "
            'monitor). A bare "--verbose" means "live".'
        )),
    min_dbfs: Optional[float] = typer.Option(
        None, "--min-dbfs",
        help="Peak dBFS gate. Skip model calls below this input peak; omit to disable."),
    mine_detections: bool = typer.Option(
        False, "--mine",
        help="Save detected clips to data_library/mined/<class>/ as mined library "
             "buckets. Mines the classes with mine=true in targets (else the "
             "active targets); turns mining on regardless of mining.enabled."),
) -> None:
    """Start the 24/7 inference loop."""
    from sed.runtime.run import _parse_target_csv, run as runtime_run
    runtime_run(
        project,
        with_artnet=False if no_artnet else None,
        live=verbose == "live" if verbose is not None else None,
        targets_override=_parse_target_csv(target),
        verbose_level=verbose,
        min_dbfs=min_dbfs,
        mine_detections=True if mine_detections else None,
        device=device,
    )


# ---------------------------------------------------------------------------
# check — pre-flight checks for each phase
# ---------------------------------------------------------------------------


@check_app.command("record")
def check_record_cmd(
    project: Optional[str] = typer.Option(
        None,
        help="Use this project's editable audio config.",
    ),
    seconds: float = typer.Option(5.0, help="Capture window for the test."),
) -> None:
    """Pre-record check: audio device, gains, all mics functional, formats OK."""
    _setup_logging("WARNING")          # quiet — check prints its own table
    from sed import doctor as checks_mod
    cfg = _load_capture_config(project)
    checks = checks_mod.doctor_record(cfg, seconds=seconds)
    checks_mod.render(checks, "Check — pre-record")
    if checks_mod.all_failed(checks):
        raise typer.Exit(code=2)


@check_app.command("train")
def check_train_cmd(project: str = typer.Option(...)) -> None:
    """Pre-train check: dataset stats, leakage, negative pool, hyperparam sanity."""
    _setup_logging("WARNING")
    from sed.config import load_config
    from sed import doctor as checks_mod
    cfg = load_config(project)
    checks = checks_mod.doctor_train(cfg, project)
    recs = checks_mod.recommend_training_hyperparams(cfg, project)
    checks_mod.render(checks, f"Check — pre-train · {project}",
                      recommendations=recs)
    if checks_mod.all_failed(checks):
        raise typer.Exit(code=2)


@check_app.command("deploy")
def check_deploy_cmd(project: str = typer.Option(...)) -> None:
    """Pre-deploy check: model loads, inference latency OK, ArtNet reachable."""
    _setup_logging("WARNING")
    from sed.config import load_config, load_config_for_inference
    from sed import doctor as checks_mod
    try:
        cfg = load_config_for_inference(project)
    except FileNotFoundError:
        cfg = load_config(project)
    checks = checks_mod.doctor_deploy(cfg, project)
    checks_mod.render(checks, f"Check — pre-deploy · {project}")
    if checks_mod.all_failed(checks):
        raise typer.Exit(code=2)


@check_app.command("all")
def check_all_cmd(project: str = typer.Option(...)) -> None:
    """Run all three check suites in sequence."""
    _setup_logging("WARNING")
    from sed.config import load_config, load_config_for_inference
    from sed import doctor as checks_mod
    cfg = load_config(project)
    try:
        deploy_cfg = load_config_for_inference(project)
    except FileNotFoundError:
        deploy_cfg = cfg
    for fn, label in [
        (lambda: checks_mod.doctor_record(cfg, seconds=3.0), "pre-record"),
        (lambda: checks_mod.doctor_train(cfg, project), "pre-train"),
        (lambda: checks_mod.doctor_deploy(deploy_cfg, project), "pre-deploy"),
    ]:
        try:
            checks = fn()
            checks_mod.render(checks, f"Check — {label} · {project}")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]check {label} failed: {e}[/red]")


testset_app = typer.Typer(no_args_is_help=True,
                          help="Named frozen test sets: save, list, apply, derive.")
evaluate_app.add_typer(testset_app, name="test-set")


@testset_app.command("list")
def testset_list_cmd(project: str = typer.Option(..., "--project", "-p", help="Project ID.")) -> None:
    """Named test sets for this project, and which one is live."""
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.data.frozen_test_presets import frozen_test_preset_state

    state = frozen_test_preset_state(ProjectLayout(project))
    table = Table(title=f"Frozen test sets — {project}", header_style="bold")
    for col in ("name", "clips", "classes", "origin", "live"):
        table.add_column(col, justify="right" if col == "clips" else "left")
    for p in state["presets"]:
        targets = [c for c in p["classes"] if c != "_negative"]
        table.add_row(
            p["name"], str(p["n_clips"]), f"{len(targets)} target",
            p["derived_from"] or "snapshot", "●" if p["is_current"] else "",
        )
    if not state["presets"]:
        table.add_row("—", "0", "", "", "")
    console.print(table)
    console.print(
        f"live test.csv: {state['current_total']} clips"
        + ("  [yellow](unsaved — a rebuild would lose it)[/yellow]" if state["current_unsaved"] else "")
    )


@testset_app.command("save")
def testset_save_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project ID."),
    name: str = typer.Argument(..., help="Name for this test set."),
    notes: str = typer.Option("", "--notes", help="Free-text note stored with it."),
) -> None:
    """Snapshot the current test.csv under a name so it can be restored."""
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.data.frozen_test_presets import save_frozen_test_preset

    try:
        record = save_frozen_test_preset(ProjectLayout(project), name, notes=notes)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]saved '{record['name']}' — {record['n_clips']} clips, "
        f"hash {record['hash'][:19]}…[/green]"
    )


@testset_app.command("apply")
def testset_apply_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project ID."),
    name: str = typer.Argument(..., help="Preset name or slug."),
) -> None:
    """Make a saved test set the live frozen test.

    Scores computed against a different test set keep their numbers but are
    flagged as scored on an older benchmark — the hash is what detects that.
    """
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.data.frozen_test_presets import apply_frozen_test_preset

    try:
        record = apply_frozen_test_preset(ProjectLayout(project), name)
    except KeyError:
        console.print(f"[red]no test set named '{name}'[/red]")
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]live test set is now '{record['name']}' ({record['n_clips']} clips)[/green]")
    console.print("[dim]re-score runs against it with `sed evaluate benchmark`.[/dim]")


@testset_app.command("check")
def testset_check_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project ID."),
    fix_name: Optional[str] = typer.Option(
        None, "--derive-clean",
        help="Write a new test set with every leaked clip removed, under this name."),
) -> None:
    """Check the live test set against every run: class space and leakage.

    A run scored on clips it trained on produces a high number that looks
    legitimate, so this is worth running before any benchmark comparison.
    """
    _setup_logging()
    import json

    from sed.config import ProjectLayout
    from sed.data.frozen_test_presets import derive_preset_excluding
    from sed.train.test_compat import check_runs

    layout = ProjectLayout(project)
    if not layout.models_dir.exists():
        console.print("[yellow]no runs to check[/yellow]")
        return
    runs = []
    for mdir in sorted(p for p in layout.models_dir.iterdir() if p.is_dir()):
        classes: list[str] = []
        cj = mdir / "classes.json"
        if cj.exists():
            try:
                classes = list(json.loads(cj.read_text(encoding="utf-8")).get("classes") or [])
            except Exception:  # noqa: BLE001 — a broken bundle is reported, not fatal
                classes = []
        runs.append({"id": mdir.name, "model_dir": mdir, "classes": classes})

    report = check_runs(layout.dataset_dir / "test.csv", runs)
    table = Table(title=f"Frozen-test compatibility — {project}", header_style="bold")
    for col in ("run", "verdict", "leaked", "issues"):
        table.add_column(col)
    for run in report["runs"]:
        leak = run["leakage"]
        leaked = ("unknown" if leak.get("available") is False
                  else (f"{leak['leak_count']} ({leak['leak_fraction']:.0%})"
                        if leak["leak_count"] else "—"))
        table.add_row(
            run["run_id"], run["verdict"], leaked,
            ", ".join(f["code"] for f in run["findings"]) or "—",
        )
    console.print(table)
    counts = report["counts"]
    console.print(
        f"{report['n_test_clips']} test clips · {counts['ok']} ready, "
        f"{counts['fixable']} fixable, {counts['incompatible']} cannot score"
    )

    leaked = report["leaked_paths_union"]
    if not leaked:
        return
    console.print(f"[red]{len(leaked)} clip(s) leak for at least one run[/red]")
    if not fix_name:
        console.print("[dim]--derive-clean <name> writes a clean copy; the live set is untouched.[/dim]")
        return
    try:
        record = derive_preset_excluding(layout, fix_name, leaked)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]saved '{record['name']}' — {record['n_clips']} clips "
        f"({record['excluded_count']} removed)[/green]"
    )
    console.print(f"[dim]apply it with `sed evaluate test-set apply --project {project} {record['slug']}`.[/dim]")


@app.command("advise")
def advise_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project ID."),
) -> None:
    """What the current feature and model choices cost on this task.

    Advisory only — every feature and backbone stays supported. This explains
    the trade-offs so an unusual choice is deliberate rather than accidental.
    """
    _setup_logging()
    from sed.config import load_config
    from sed.train.advice import config_advice

    cfg = load_config(project)
    advice = config_advice(cfg)
    colours = {"error": "red", "warn": "yellow", "info": "dim"}
    marks = {"error": "✗", "warn": "!", "info": "ℹ"}

    for area, findings in advice.items():
        console.print(f"\n[bold]{area}[/bold]")
        if not findings:
            console.print("  [green]✓ nothing to flag[/green]")
            continue
        for f in findings:
            colour, mark = colours[f["severity"]], marks[f["severity"]]
            console.print(f"  [{colour}]{mark} {f['message']}[/{colour}]")
            if f.get("suggest"):
                hint = ", ".join(f"{k} = {v}" for k, v in f["suggest"].items())
                console.print(f"    [dim]try: {hint}[/dim]")
    console.print(
        "\n[dim]Advisory only — nothing here blocks training or removes a model.[/dim]"
    )


@evaluate_app.command("test-health")
def evaluate_test_health_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project ID."),
) -> None:
    """Check whether the frozen test set can answer what is asked of it.

    Needs no model: reports per-class support and the recall resolution it buys,
    the negative prior every prior-bound metric depends on, and whether false
    positives are measurable at all.
    """
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.train.testset_health import health_for_test_csv

    layout = ProjectLayout(project)
    health = health_for_test_csv(layout.dataset_dir / "test.csv")

    table = Table(title=f"Frozen test set — {project}", header_style="bold")
    table.add_column("class")
    table.add_column("clips", justify="right")
    table.add_column("recall resolution", justify="right")
    for name, row in health["per_class"].items():
        hw = row["recall_ci_halfwidth"]
        table.add_row(
            name, str(row["support"]),
            "—" if hw is None else f"±{hw * 100:.0f} pts",
        )
    console.print(table)

    prior = health["negative_prior"]
    console.print(
        f"{health['n_samples']} clips · {health['n_negatives']} negative "
        f"({'n/a' if prior is None else f'{prior:.1%}'} prior) · "
        f"{health['n_target_clips']} target clips across "
        f"{health['n_target_classes']} class(es)"
    )
    for finding in health["findings"]:
        colour = {"error": "red", "warn": "yellow"}.get(finding["severity"], "dim")
        mark = {"error": "✗", "warn": "!"}.get(finding["severity"], "ℹ")
        console.print(f"[{colour}]{mark} {finding['message']}[/{colour}]")

    if not health["ok"]:
        raise typer.Exit(1)


@evaluate_app.command("onset-gate")
def evaluate_onset_gate_cmd(
    paths: list[Path] = typer.Argument(
        ..., help="WAV file(s) or directories to scan (directories recurse)."),
    project: str = typer.Option(..., "--project", "-p", help="Project ID (supplies [onset] config)."),
    channels: Optional[str] = typer.Option(
        None, "--channels",
        help="Comma-separated physical channels. Default: [inference] inference_channels."),
    bins: int = typer.Option(12, "--bins", help="Flux histogram bin count."),
    per_file: bool = typer.Option(False, "--per-file", help="Also print a row per file."),
) -> None:
    """Measure the onset gate — no model, no annotation needed.

    With onset gating on, the detector decides what the model ever sees, so its
    behaviour bounds the whole system: a clink it drops is unrecoverable, and
    onsets/hour is what converts a per-onset false-alarm rate into FP/hour.
    """
    _setup_logging()
    from sed.runtime.onset_scan import aggregate, flux_histogram, scan_files

    cfg = _load_capture_config(project)

    wavs: list[Path] = []
    for p in paths:
        if p.is_dir():
            wavs.extend(sorted(p.rglob("*.wav")))
        elif p.exists():
            wavs.append(p)
        else:
            console.print(f"[yellow]skipping missing path: {p}[/yellow]")
    if not wavs:
        console.print("[red]no WAV files found[/red]")
        raise typer.Exit(1)

    chans = _parse_channels(channels) if channels else None
    console.print(f"scanning {len(wavs)} file(s) with the production onset gate ...")
    results = scan_files(cfg, wavs, channels=chans)
    if not results:
        console.print("[red]no files could be scanned[/red]")
        raise typer.Exit(1)

    agg = aggregate(results)
    hours = agg["duration_s"] / 3600.0

    summary = Table(title=f"Onset gate — {project}", header_style="bold")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("files scanned", str(agg["files"]))
    summary.add_row("audio duration", f"{agg['duration_s']:.1f} s ({hours:.2f} h)")
    summary.add_row("raw onsets (all mics)", str(agg["raw_onsets"]))
    summary.add_row("merged onsets (scored)", str(agg["merged_onsets"]))
    summary.add_row("[bold]onsets / hour[/bold]", f"[bold]{agg['onsets_per_hour']:.1f}[/bold]")
    summary.add_row("merge collapse ratio", f"{agg['merge_ratio']:.2f}×")
    if agg["median_interval_s"] is not None:
        summary.add_row("median onset interval", f"{agg['median_interval_s'] * 1000:.0f} ms")
    console.print(summary)

    hist = flux_histogram(results, bins=bins)
    if hist["bins"]:
        peak = max(b["count"] for b in hist["bins"]) or 1
        table = Table(title="Spectral flux distribution", header_style="bold")
        table.add_column("flux (dB)", justify="right")
        table.add_column("frames", justify="right")
        table.add_column("")
        for b in hist["bins"]:
            table.add_row(
                f"{b['lo_db']:.0f}..{b['hi_db']:.0f}",
                str(b["count"]),
                "█" * int(round(28 * b["count"] / peak)),
            )
        console.print(table)
        if hist.get("median_gate_db") is not None:
            console.print(
                f"median flux {hist['median_flux_db']:.1f} dB vs median gate "
                f"{hist['median_gate_db']:.1f} dB "
                f"(min_flux floor {cfg.onset.min_flux_db:.1f} dB, "
                f"mad_k {cfg.onset.mad_k})"
            )

    if per_file:
        table = Table(title="Per file", header_style="bold")
        for col, just in (("file", "left"), ("seconds", "right"),
                          ("raw", "right"), ("merged", "right"), ("onsets/h", "right")):
            table.add_column(col, justify=just)
        for r in results:
            table.add_row(
                Path(r.path).name, f"{r.duration_s:.1f}",
                str(r.raw_onset_count), str(r.merged_onset_count),
                f"{r.onsets_per_hour:.1f}",
            )
        console.print(table)

    console.print(
        "\n[dim]onsets/hour is the denominator that turns false-alarms-per-1000-onsets "
        "into FP/hour. Scan continuous venue audio for a deployment-valid figure — "
        "onset-triggered clips are already gate survivors and will overstate it.[/dim]"
    )


if __name__ == "__main__":
    app()
