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
from typing import Optional

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
app.add_typer(audio_app, name="audio")
app.add_typer(library_app, name="library")
app.add_typer(project_app, name="project")
app.add_typer(dataset_app, name="dataset")

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


def _first_metric(metrics: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in metrics:
            return metrics.get(key)
    return None


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


def _model_artifact_label(layout, model_id: str) -> str:
    return f"IR={_model_ir_label(layout, model_id)}  calibrated={_model_calibrated_label(layout, model_id)}"


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
) -> None:
    """Scaffold a new project under projects/<id>/."""
    from datetime import datetime, timezone
    from sed.config import ProjectLayout, write_project_config_template
    layout = ProjectLayout(project_id)
    created_at = datetime.now(timezone.utc).isoformat()
    if layout.root.exists():
        layout.create()
        if not layout.config_toml.exists():
            write_project_config_template(layout, name=name or project_id, created_at=created_at)
            console.print(f"[green]created missing {layout.config_toml}[/green]")
        else:
            console.print(f"[yellow]exists: {layout.root}[/yellow]")
        return
    layout.create()
    write_project_config_template(layout, name=name or project_id, created_at=created_at)
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
    """Validate derived onset timing loaded from config.toml."""
    clip_ms = int(cfg.onset.record_seconds * 1000)
    if cfg.onset.preroll_ms >= clip_ms:
        raise typer.BadParameter(
            f"[onset] preroll_ms ({cfg.onset.preroll_ms} ms) must be below "
            f"the configured recording length ({cfg.onset.record_seconds:g}s)"
        )
    cfg.onset.postroll_ms = clip_ms - cfg.onset.preroll_ms


_DENSITIES = ("empty", "quiet", "normal", "busy", "peak")


@app.command("record", cls=ConfigHelpCommand)
def record_cmd(
    class_name: str = typer.Option(..., "--class", "-c", help="Sound label to save under data_library/sounds/<label>/."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p",
        help="Optional project association. Must already exist.",
    ),
    num: int = typer.Option(50, "--num", "-n", help="Stop after this many onset-triggered clips. Set 0 to rely on duration/Ctrl+C."),
    duration: str = typer.Option("", "--duration", "-du", help="Optional wall-clock cap, e.g. '15m'. Empty = no time limit."),
    density: str = typer.Option("normal", "--density", "-de", help="empty | quiet | normal | busy | peak"),
    description: str = typer.Option("", "--description", "-d", help="Free-form note saved in each sidecar."),
    mic_position: str = typer.Option("", "--mic-position", "-mp", help="Free-form mic placement note saved in each sidecar."),
    device: Optional[str] = typer.Option(
        None, "--device", "-dev", callback=_device_callback, is_eager=True,
        help='Override recording device. A bare "--device" lists available devices.',
    ),
    channels: Optional[str] = typer.Option(None, "--channels", "--channel", "-ch", help="Ordered list of 0-based channel INDEXES to record (not a channel count): interface input N is index N-1, so inputs 3 & 4 are '-ch 2,3'. E.g. '3,0,1' or '0' or '1,2'. Default: config [audio] record_channels."),
    onset_threshold: Optional[float] = typer.Option(
        None, "--onset-threshold", "-ot",
        help="Override onset floor in dB for this recording session. Lower catches quieter onsets.",
    ),
    test: bool = typer.Option(
        False, "--test", "-t",
        help="Mark clips as this project's frozen test material. Requires --project.",
    ),
    tags: str = typer.Option(
        "", "--tags", "-tg",
        help='Comma-separated tags saved in each sidecar, e.g. "wine-glass,full-50".',
    ),
    split: bool = typer.Option(
        False, "--split",
        help="Also split each multichannel clip into per-channel mono clips as it is recorded.",
    ),
) -> None:
    """Record onset-triggered clips into the unified shared data library."""
    _setup_logging()
    from sed.config import ProjectLayout
    from sed.data.library import Library, RAW_CATEGORY
    from sed.data.recorder import record_auto

    if density not in _DENSITIES:
        raise typer.BadParameter(
            f"--density must be one of: {', '.join(_DENSITIES)}"
        )
    parsed_tags = _parse_tags(tags)
    if test and not project:
        raise typer.BadParameter("--test requires --project")
    if project:
        layout = ProjectLayout(project)
        if not layout.root.exists():
            raise typer.BadParameter(
                f"Project '{project}' does not exist. Run 'sed project new {project}' first."
            )
    project_id = project or "_library"
    cfg = _load_capture_config(project)
    channels_list = _parse_channels(channels) if channels else None
    _apply_audio_overrides(cfg, device=device, channels=channels_list)
    _apply_onset_threshold_override(cfg, onset_threshold)
    _finalize_onset_config(cfg)

    lib = Library()
    out = lib.raw_dir(class_name)
    dataset_role = "test" if test else "train"
    console.print(
        f"[bold]recording[/bold] [cyan]{class_name}[/cyan] "
        f"[dim]project={project_id} role={dataset_role} density={density} -> {out}[/dim]"
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
        class_name=class_name,
        label=class_name,
        category=RAW_CATEGORY,
        project_id=project_id,
        dataset_role=dataset_role,
        density=density,
        review_status="accepted",
        duration_s=duration_s,
        mic_position=mic_position,
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
    test: bool = typer.Option(False, "--test", help="Mark imported clips as frozen test material for --project."),
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
    if test and not project:
        raise typer.BadParameter("--test requires --project")
    parsed_tags = _parse_tags(tags)
    lib = Library()
    n = 0
    for wav in sorted(Path(src_dir).rglob("*.wav")):
        lib.import_clip(
            wav,
            label=label,
            project_id=project or "_library",
            dataset_role="test" if test else "train",
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
    test: bool = typer.Option(False, "--test", help="Mark sliced clips as frozen test material for --project."),
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
    if test and not project:
        raise typer.BadParameter("--test requires --project")
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
            dataset_role="test" if test else "train",
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
def dataset_compose_cmd(project: str = typer.Option(...)) -> None:
    """Build train/val/test from data_library and project label mappings."""
    _setup_logging()
    from sed.config import load_config
    from sed.data.compose import compose_dataset
    cfg = load_config(project)
    summary = compose_dataset(cfg, project)

    # Rendered the same way the check tables are — same header_style, same
    # row shape — so the terminal output stays consistent across commands.
    splits = summary["splits"]
    by_origin = summary["by_origin"]
    frozen = summary["frozen_test"]
    table = Table(title=f"Dataset — {project}", header_style="bold")
    table.add_column("class", style="bold")
    table.add_column("sounds", justify="right")
    table.add_column("derived", justify="right")
    table.add_column("train+val", justify="right")
    table.add_column("test (frozen)", justify="right")
    for cls, c in by_origin.items():
        test_n = frozen["by_class"].get(cls, 0)
        table.add_row(cls,
                      str(c.get("sounds", 0)), str(c.get("derived", 0)),
                      str(c["total"]), str(test_n))
    console.print(table)

    footer = (
        f"  [bold]train[/bold] {splits['train']}  ·  "
        f"[bold]val[/bold] {splits['val']}  ·  "
        f"[bold]test[/bold] {splits['test']}"
    )
    if frozen["present"]:
        footer += "  [dim](test = library clips with dataset_role=test)[/dim]"
    else:
        footer += "  [yellow](no frozen test — record/import clips with --test for stable eval)[/yellow]"
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
            "Use library clips marked dataset_role=test as bases and write derived "
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

    With --frozen-test, bases come from library test clips and outputs go to
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
                                  help="Skip the post-training test-set evaluation."),
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
    # Gather per-class union
    classes: list[str] = []
    seen = set()
    for m in items:
        for c in (m.get("test_metrics") or {}).get("per_class", {}).keys():
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
    metric_rows.append((
        "test acc",
        [_fmt_percent((m.get("test_metrics") or {}).get("accuracy")) for m in items],
    ))
    metric_rows.append((
        "test F1",
        [
            _fmt_percent(
                _first_metric(m.get("test_metrics") or {}, ("f1_macro_targets", "f1_macro"))
            )
            for m in items
        ],
    ))
    metric_rows.append((
        "test AP",
        [
            _fmt_percent(
                _first_metric(m.get("test_metrics") or {}, ("ap_macro_targets", "ap_macro"))
            )
            for m in items
        ],
    ))
    for c in classes:
        metric_rows.append((
            f"F1 {c}",
            [
                _fmt_percent(
                    ((m.get("test_metrics") or {}).get("per_class") or {})
                    .get(c, {})
                    .get("f1")
                )
                for m in items
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
    projects/<id>/dataset/test.csv (library clips marked dataset_role=test),
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


@evaluate_app.command("list")
def evaluate_list_cmd(project: str = typer.Option(..., help="Project ID.")) -> None:
    """Show library clips marked as this project's frozen test set."""
    _setup_logging()
    from sed.data.library import Library
    clips = Library().query(project_ids={project}, dataset_role="test", review_status="accepted")
    table = Table(title=f"Frozen test set — {project}")
    table.add_column("role"); table.add_column("label")
    table.add_column("clips", justify="right")
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip.label] = counts.get(clip.label, 0) + 1
    if not counts:
        table.add_row("test", "—", "0")
    for label, count in sorted(counts.items()):
        table.add_row("test", label, str(count))
    console.print(table)


@app.command("calibrate")
def calibrate_cmd(
    project: str = typer.Option(...),
    target_fp_per_hour: float = typer.Option(1.0, help="Tolerated false-positive rate."),
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
    chosen = calibrate(
        cfg, project,
        negative_dir=negative_dir,
        target_fp_per_hour=target_fp_per_hour,
        negative_source=negative_source,
    )
    console.print(chosen)


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
        help="Save detected clips to data_library/mined/<class>/ for review. Mines the "
             "classes with mine=true in targets (else the active targets); turns "
             "mining on regardless of mining.enabled."),
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


if __name__ == "__main__":
    app()
