"""The recording console page.

One page, no tabs, no build step: the same vanilla-JS-and-Jinja shape the
inspect UI uses, rendering the shared shell from :mod:`sed.data.ui_theme` so
the two look like one application.

The layout follows what an operator actually does on site, top to bottom:
confirm the interface is alive (Setup), say what is being recorded (Session),
then watch the microphones and adjust the gate while it runs (Live). Setup
collapses once recording starts, because after that it is the meters and the
onset floor you are looking at.
"""

from __future__ import annotations

import json

from sed.config import Config
from sed.data import ui_theme

# Density is the recording context, and it is the one piece of metadata people
# forget to change when the room fills up. Number keys 1-5 switch it.
DENSITIES = ("empty", "quiet", "normal", "busy", "peak")

SAMPLE_RATES = (16000, 22050, 32000, 44100, 48000, 96000)
BLOCKSIZES = (256, 512, 1024, 2048, 4096)
TRIGGER_MODES = (
    ("any", "any — each mic triggers, save the mics that fired"),
    ("any_all", "any_all — each mic triggers, save every mic"),
    ("mix", "mix — one detector on the downmix"),
    ("ch0", "ch0 — first channel only"),
)


def render_record_html(cfg: Config, project_id: str, form: dict | None = None) -> str:
    from jinja2 import Template

    onset = cfg.onset
    boot = {
        "projectId": project_id,
        "densities": list(DENSITIES),
        "sampleRates": list(SAMPLE_RATES),
        "blocksizes": list(BLOCKSIZES),
        "triggerModes": [m[0] for m in TRIGGER_MODES],
        "capture": {
            "device": cfg.deploy.record_device,
            "sample_rate": cfg.audio.record_sample_rate,
            "blocksize": cfg.audio.blocksize,
            "channels": list(cfg.audio.record_channels),
        },
        "onset": {
            "min_flux_db": onset.min_flux_db,
            "mad_k": onset.mad_k,
            "recording_cooldown_ms": onset.recording_cooldown_ms,
            "cross_channel_merge_ms": onset.cross_channel_merge_ms,
            "record_seconds": onset.record_seconds,
            "preroll_ms": onset.preroll_ms,
            "randomize_preroll": onset.randomize_preroll,
            "preroll_ms_range": list(onset.preroll_ms_range),
            "multi_mic_trigger": onset.multi_mic_trigger,
        },
        "form": dict(form or {}),
    }
    return Template(_TEMPLATE).render(
        project_id=project_id or "library",
        boot_json=json.dumps(boot),
        trigger_modes=TRIGGER_MODES,
        sample_rates=SAMPLE_RATES,
        blocksizes=BLOCKSIZES,
        densities=DENSITIES,
        theme_prepaint_js=ui_theme.THEME_PREPAINT_JS,
        font_link=ui_theme.FONT_LINK,
        shell_css=ui_theme.shell_css(),
        shell_js=ui_theme.shell_js(),
        tooltip_element=ui_theme.TOOLTIP_ELEMENT,
        toast_element=ui_theme.TOAST_ELEMENT,
        record_css=_RECORD_CSS,
        record_js=_RECORD_JS,
    )


_RECORD_CSS = r"""
/* ---- page rhythm -------------------------------------------------------
   Three stacked bands. Setup is a <details> because it is answered once per
   session; Session and Live stay open because they are answered continuously. */
.rec-band { margin-bottom: 1rem; }
.rec-panel { border: 1px solid var(--bd); border-radius: 8px; padding: 0.85rem;
             background: var(--surface); }
.rec-panel > h3 { margin: 0 0 0.7rem; font-size: 0.95rem; color: var(--tx-strong);
                  display: flex; align-items: center; justify-content: space-between;
                  gap: 0.5rem; }
.rec-panel > h3 .section-head { display: inline-flex; align-items: center; gap: 0.4rem; }
details.rec-panel > summary { list-style: none; cursor: pointer; margin: -0.85rem;
    padding: 0.85rem; font-size: 0.95rem; font-weight: 600; color: var(--tx-strong);
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
details.rec-panel > summary::-webkit-details-marker { display: none; }
details.rec-panel[open] > summary { margin-bottom: 0.2rem; border-bottom: 1px solid var(--bd-soft); }
details.rec-panel > summary::before { content: "▸"; margin-right: 0.45rem; color: var(--tx-faint); }
details.rec-panel[open] > summary::before { content: "▾"; }
.rec-live { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr);
            gap: 0.85rem; align-items: start; }
@media (max-width: 900px) { .rec-live { grid-template-columns: 1fr; } }

/* ---- forms ------------------------------------------------------------- */
.rec-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 0.6rem; }
.rec-fields.wide { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.rec-field { display: flex; flex-direction: column; gap: 0.22rem; min-width: 0; }
.rec-field > span { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.02em;
                    text-transform: uppercase; color: var(--tx-muted); }
.rec-field input[type="text"], .rec-field input[type="number"], .rec-field select {
    padding: 0.42rem 0.5rem; border: 1px solid var(--bd-2); border-radius: 6px;
    background: var(--surface); color: var(--tx-strong-2); font: inherit; font-size: 0.85rem;
    min-width: 0;
}
.rec-field input:focus, .rec-field select:focus {
    outline: none; border-color: var(--blue-mid); box-shadow: 0 0 0 2px rgba(99,102,241,0.12);
}
.rec-hint { font-size: 0.72rem; color: var(--tx-muted); margin-top: 0.3rem; line-height: 1.5; }
/* While a session runs, the capture note is a warning rather than a caption. */
.rec-hint.live { color: var(--amber-solid-2); font-weight: 600; }
.rec-check { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem;
             font-weight: 600; color: var(--tx-2); cursor: pointer; }

/* Segmented controls: density and mode are small closed sets, and on site a
   row of targets beats a dropdown you have to open. */
.seg { display: inline-flex; border: 1px solid var(--bd-2); border-radius: 6px; overflow: hidden; }
.seg button { border: none; background: var(--surface); color: var(--tx-2); cursor: pointer;
    font: inherit; font-size: 0.76rem; font-weight: 700; padding: 0.36rem 0.6rem;
    border-right: 1px solid var(--bd-2); transition: background 0.15s, color 0.15s; }
.seg button:last-child { border-right: none; }
.seg button:hover { background: var(--surface-3); }
.seg button.on { background: var(--inv-bg); color: var(--inv-fg); }

/* ---- transport --------------------------------------------------------- */
.rec-transport { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
                 margin-bottom: 0.85rem; }
.rec-go { min-height: 2.6rem; padding: 0.5rem 1.6rem; border-radius: 8px; border: none;
    cursor: pointer; font: inherit; font-size: 1rem; font-weight: 700; color: #fff;
    background: var(--green-solid-4); transition: background 0.15s; }
.rec-go:hover { background: var(--green-deep-4); }
.rec-go.stop { background: var(--red-solid-3); }
.rec-go.stop:hover { background: var(--red-solid-2); }
.rec-go:disabled { background: var(--fill-muted); cursor: default; }
.rec-counter { font-size: 1.35rem; font-weight: 800; font-variant-numeric: tabular-nums;
               color: var(--tx-strong); }
.rec-counter small { font-size: 0.8rem; font-weight: 600; color: var(--tx-muted); }

/* ---- channel meters ---------------------------------------------------- */
.meter-list { display: grid; gap: 0.55rem; }
.meter { border: 1px solid var(--bd); border-radius: 8px; padding: 0.55rem 0.7rem;
         background: var(--surface-2); transition: border-color 0.12s, background 0.12s; }
.meter.fired { border-color: var(--teal-solid); background: var(--teal-tint); }
.meter.silent { border-color: var(--amber-solid); }
.meter.clipping { border-color: var(--red-mid); }
.meter-head { display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.35rem; }
.meter-ch { font-weight: 800; font-size: 0.85rem; color: var(--tx-strong); }
.meter-mic { font-size: 0.75rem; color: var(--tx-muted); font-weight: 600; flex: 1 1 auto;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.meter-db { font-variant-numeric: tabular-nums; font-weight: 700; font-size: 0.8rem;
            color: var(--tx-2); }
/* Peak fills the bar; RMS is a tick inside it, so one glance reads both the
   transient and the level it sits on. */
.bar { position: relative; height: 14px; border-radius: 4px; background: var(--surface-4);
       overflow: hidden; }
.bar-peak { position: absolute; inset: 0 auto 0 0; border-radius: 4px;
            background: var(--green-solid-2);
            /* Matched to the poll interval so each sample glides into the
               next instead of stepping. */
            transition: width 110ms cubic-bezier(0.33, 1, 0.68, 1),
                        background-color 140ms linear; }
.bar-peak.warm { background: var(--amber-solid); }
.bar-peak.hot { background: var(--red-mid); }
.bar-rms { position: absolute; top: 0; bottom: 0; width: 2px;
           background: rgba(15,23,42,0.55); transition: left 110ms ease-out; }
:root[data-theme="dark"] .bar-rms { background: rgba(226,232,240,0.7); }
.meter-gate { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.35rem;
              font-size: 0.74rem; font-variant-numeric: tabular-nums; color: var(--tx-muted); }
.meter-gate .go { color: var(--green-solid-4); font-weight: 800; }
.meter-gate .no { color: var(--tx-faint); font-weight: 800; }
.meter-spark { display: block; width: 100%; height: 22px; margin-top: 0.3rem; }
.meter-spark path { fill: none; stroke: var(--blue-mid); stroke-width: 1.2; }
.meter-badges { display: inline-flex; gap: 0.3rem; }

/* ---- onset controls ---------------------------------------------------- */
.knob { margin-bottom: 0.7rem; }
.knob-head { display: flex; align-items: baseline; justify-content: space-between;
             gap: 0.5rem; margin-bottom: 0.15rem; }
.knob-head label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: 0.02em; color: var(--tx-muted); }
.knob-val { font-size: 0.82rem; font-weight: 800; font-variant-numeric: tabular-nums;
            color: var(--tx-strong); }
.knob input[type="range"] { width: 100%; accent-color: var(--blue-solid-2); }
/* Paired min/max sliders: tag and value are fixed-width so the two tracks line
   up and the numbers do not jump around as they change. */
.range-row { display: grid; grid-template-columns: 2.1rem 1fr 4.2rem; gap: 0.4rem;
             align-items: center; }
.range-row + .range-row { margin-top: 0.2rem; }
.range-tag { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.02em; color: var(--tx-faint); }
.range-val { font-size: 0.76rem; font-weight: 700; font-variant-numeric: tabular-nums;
             text-align: right; color: var(--tx-2); }
.knob-note { font-size: 0.7rem; color: var(--tx-faint); }
.group-foot { display: flex; align-items: center; justify-content: space-between;
              gap: 0.5rem; margin-top: 0.6rem; padding-top: 0.55rem;
              border-top: 1px solid var(--bd-soft); }

/* The clip is longer than the model window on purpose, and where the onset
   sits inside it decides what the model is trained on — so show it.
   Built from divs rather than an SVG: the old one stretched its own viewBox to
   the panel width, which distorted the glyphs and pushed the labels through the
   bar. Here the text is real text and lays itself out. */
.geom { margin: 0.45rem 0 0.1rem; }
/* Three regions, left to right: the preroll every clip is guaranteed, the
   window the randomized onset actually lands in, and the postroll. The track's
   own colour is the postroll, and the first two are painted over it. */
.geom-track { position: relative; height: 20px; border-radius: 4px;
              background: var(--blue-tint-3); border: 1px solid var(--bd-2);
              overflow: hidden; }
.geom-pre-fill { position: absolute; top: 0; bottom: 0; left: 0;
                 background: var(--surface-4); }
.geom-jitter { position: absolute; top: 0; bottom: 0;
               background: repeating-linear-gradient(
                   -45deg, var(--blue-soft) 0 4px, var(--blue-tint-3) 4px 8px); }
.geom-onset { position: absolute; top: -2px; bottom: -2px; width: 2px;
              background: var(--blue-solid-2); }
.geom-onset[hidden] { display: none; }
#geom-jitter-key[hidden] { display: none; }
.geom-scale { display: flex; justify-content: space-between; margin-top: 0.15rem;
              font-size: 0.66rem; color: var(--tx-faint);
              font-variant-numeric: tabular-nums; }
.geom-legend { display: flex; flex-wrap: wrap; gap: 0.15rem 0.85rem; margin-top: 0.3rem;
               font-size: 0.7rem; color: var(--tx-muted); }
.geom-legend span { display: inline-flex; align-items: center; gap: 0.3rem; }
.geom-legend b { color: var(--tx-2); font-variant-numeric: tabular-nums; }
.geom-legend .sw { width: 9px; height: 9px; border-radius: 2px; border: 1px solid var(--bd-2); }
.geom-legend .sw.pre { background: var(--surface-4); }
.geom-legend .sw.post { background: var(--blue-tint-3); }
.geom-legend .sw.jit { background: repeating-linear-gradient(
    -45deg, var(--blue-soft) 0 3px, var(--blue-tint-3) 3px 6px); }

/* ---- take inspector ----------------------------------------------------
   A take is worth a close look while you can still re-record it: a linear,
   full-range spectrogram shows mains hum as a hard horizontal line and cable
   hash as a bright top octave, neither of which is visible in the mel preview
   the dataset UI uses. */
.take-modal { position: fixed; inset: 0; z-index: 1500; display: flex;
              align-items: center; justify-content: center; padding: 1.5rem;
              background: rgba(15,23,42,0.82); }
.take-modal[hidden] { display: none; }
.take-modal-card { background: var(--surface); border: 1px solid var(--bd);
    border-radius: 12px; padding: 0.85rem; max-width: min(1060px, 96vw);
    width: 100%; box-shadow: 0 24px 60px rgba(0,0,0,0.5); }
.take-modal-head { display: flex; align-items: center; gap: 0.6rem;
                   flex-wrap: wrap; margin-bottom: 0.6rem; }
.take-modal-head strong { font-size: 0.85rem; color: var(--tx-strong);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 40ch; }
.take-modal-meta { font-size: 0.75rem; color: var(--tx-muted);
                   font-variant-numeric: tabular-nums; flex: 1 1 auto; }
.take-modal img { display: block; width: 100%; border-radius: 8px;
                  background: #0f172a; }
.tm-verdict { margin-top: 0.6rem; display: grid; gap: 0.3rem; font-size: 0.8rem; }
.tm-ok { color: var(--green-solid-4); font-weight: 700; }
.tm-warn { color: var(--amber-solid-2); font-weight: 600; }
.tm-stats { color: var(--tx-muted); font-size: 0.75rem;
            font-variant-numeric: tabular-nums; }
.take img { cursor: zoom-in; }

/* ---- recent clips ------------------------------------------------------ */
.recent { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
          gap: 0.5rem; }
.take { border: 1px solid var(--bd); border-radius: 8px; padding: 0.4rem; background: var(--surface);
        position: relative; }
.take img { display: block; width: 100%; height: 54px; object-fit: cover; border-radius: 5px;
            background: #0f172a; }
.take-meta { font-size: 0.68rem; color: var(--tx-muted); margin-top: 0.25rem;
             font-variant-numeric: tabular-nums; line-height: 1.4; }
.take-actions { display: flex; gap: 0.25rem; margin-top: 0.3rem; }
.take-actions button { flex: 1 1 auto; }
.rec-log { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.72rem;
    line-height: 1.65; max-height: 190px; overflow-y: auto; color: var(--tx-2);
    background: var(--surface-2); border: 1px solid var(--bd-soft); border-radius: 6px;
    padding: 0.5rem 0.65rem; }
.rec-log .t { color: var(--tx-faint); }
.rec-log .n { color: var(--tx-strong); font-weight: 700; }
.rec-empty { color: var(--tx-faint); font-style: italic; font-size: 0.8rem; }

/* ---- run history -------------------------------------------------------
   One line per pass. Deliberately dense: it is a glanceable record, not a
   browser — reviewing the clips themselves is what Inspect is for. */
.runs { display: grid; gap: 0.25rem; max-height: 230px; overflow-y: auto; }
.run { display: grid; grid-template-columns: 0.5rem 3.2rem minmax(0, 1fr) auto 1.7rem;
       gap: 0.5rem; align-items: center; padding: 0.32rem 0.45rem; border-radius: 6px;
       background: var(--surface-2); border: 1px solid var(--bd-soft); font-size: 0.76rem; }
.run.live { border-color: var(--teal-solid); background: var(--teal-tint); }
.run-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--fill-muted); }
.run.live .run-dot { background: var(--red-mid); animation: rec-pulse 1.1s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) { .run.live .run-dot { animation: none; } }
.run-when { color: var(--tx-faint); font-variant-numeric: tabular-nums; }
.run-label { font-weight: 700; color: var(--tx-strong); overflow: hidden;
             text-overflow: ellipsis; white-space: nowrap; }
.run-label small { display: block; font-weight: 500; color: var(--tx-muted);
                   font-size: 0.68rem; overflow: hidden; text-overflow: ellipsis; }
.run-clips { font-weight: 800; font-variant-numeric: tabular-nums; color: var(--tx-2);
             white-space: nowrap; }
.run-clips small { font-weight: 600; color: var(--tx-faint); }
.run button { padding: 0.15rem 0.35rem; min-height: 1.4rem; }

/* ---- pre-flight -------------------------------------------------------- */
.checks { display: grid; gap: 0.3rem; margin-top: 0.6rem; }
.check { display: grid; grid-template-columns: 1.6rem minmax(120px, max-content) 1fr;
         gap: 0.5rem; align-items: baseline; font-size: 0.8rem; }
.check .st { font-weight: 800; }
.check.PASS .st { color: var(--green-solid-4); }
.check.WARN .st { color: var(--amber-solid-2); }
.check.FAIL .st { color: var(--red-solid-2); }
.check .nm { font-weight: 700; color: var(--tx-2); }
.check .dt { color: var(--tx-muted); }
.check .fx { grid-column: 3; color: var(--amber-solid-2); font-size: 0.75rem; }

/* ---- channel picker ---------------------------------------------------- */
.chan-pick { display: grid; gap: 0.3rem; margin-top: 0.5rem; }
.chan-row { display: grid; grid-template-columns: 1.4rem 3.2rem 1fr 5rem;
            gap: 0.5rem; align-items: center; font-size: 0.8rem; }
.chan-row .ci { font-weight: 800; color: var(--tx-strong); }
.chan-row .cm { color: var(--tx-muted); font-size: 0.75rem; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
.chan-row .cd { text-align: right; font-variant-numeric: tabular-nums; font-size: 0.74rem;
                color: var(--tx-muted); }
.chan-row input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--blue-solid-2); }

.topbar-project { display: inline-flex; align-items: center; gap: 0.35rem; }
.topbar-project > span { color: var(--muted); font-size: 0.68rem; font-weight: 700;
                         text-transform: uppercase; letter-spacing: 0.02em; }
/* The closed control sits on the dark bar, but the OPEN list is drawn by the
   platform against the page, so white-on-translucent left the option text
   invisible in both themes. Give the popup its own opaque surface, and pin the
   colour scheme so the native widget picks matching chrome. */
.topbar-project select { background: #1e293b; color: #fff; font: inherit;
    font-size: 0.8rem; font-weight: 600; border: 1px solid var(--chip-bd);
    border-radius: 6px; padding: 0.2rem 0.35rem; cursor: pointer;
    color-scheme: dark; }
.topbar-project select option { background: #1e293b; color: #fff; }
.topbar-project select:disabled { opacity: 0.55; cursor: default; }
.capture-alert { margin: 0.6rem 0; }
.topbar-state { display: inline-flex; align-items: center; gap: 0.4rem; font-weight: 700;
                font-size: 0.85rem; }
.topbar-state .dot { width: 9px; height: 9px; border-radius: 999px; background: var(--muted); }
.topbar-state.rec .dot { background: #ef4444; animation: rec-pulse 1.1s ease-in-out infinite; }
.topbar-state.rec { color: #fca5a5; }
@keyframes rec-pulse { 50% { opacity: 0.25; } }
@media (prefers-reduced-motion: reduce) { .topbar-state.rec .dot { animation: none; } }
.topbar-meta { color: var(--muted); font-size: 0.8rem; font-variant-numeric: tabular-nums; }
"""


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SED Recording — {{ project_id }}</title>
{{ theme_prepaint_js|safe }}
{{ font_link|safe }}
<style>
{{ shell_css|safe }}
{{ record_css|safe }}
</style>
</head>
<body>

<div class="toolbar" id="toolbar">
  <div style="display:flex; align-items:center; gap:1rem; min-width:0;">
    <strong style="font-size:0.95rem;">Recording</strong>
    <label class="topbar-project" data-tip="Most recording just fills the shared library, so no project is the default — those clips are usable by every project.\n\nPick a project to record **for** it: the clips are stamped with it and its own source rules pick them up, and another project can still use them by naming it in a rule. Choosing one also loads that venue's device, channel and onset settings.">
      <span>project</span>
      <select id="f-project"><option value="">— none —</option></select>
    </label>
    <span class="topbar-state" id="state-chip"><span class="dot"></span><span id="state-text">idle</span></span>
    <span class="topbar-meta" id="topbar-device"></span>
  </div>
  <div style="display:flex; align-items:center; gap:0.75rem;">
    <span class="topbar-meta" id="topbar-elapsed"></span>
    <button class="theme-toggle" id="theme-toggle" title="Switch light / dark"
            data-tip="Switch between the light and dark theme"
            style="background:var(--chip-bg); border:1px solid var(--chip-bd); color:#fff;
                   border-radius:8px; width:32px; height:32px; cursor:pointer;">
      <span class="theme-icon-moon">🌙</span><span class="theme-icon-sun">☀️</span>
    </button>
    <button class="done" id="btn-done" data-tip="Stop the session and shut the server down">Done</button>
  </div>
</div>

<h1>Recording console</h1>
<div class="notice danger capture-alert" id="capture-alert" hidden></div>
<p class="rec-hint" style="margin-top:-0.5rem;">
  Onset-triggered capture into <code>data_library/sounds/&lt;label&gt;/</code>.
  Shortcuts: <span class="kbd">Space</span> capture now ·
  <span class="kbd">R</span> start/stop · <span class="kbd">U</span> undo last take ·
  <span class="kbd">1</span>–<span class="kbd">5</span> density.
</p>

{# ---- A. Setup: answered once per session, then folded away ------------- #}
<details class="rec-panel rec-band" id="setup-panel" open>
  <summary>
    <span>Setup — interface, channels, pre-flight</span>
    <span class="section-status" id="setup-status"></span>
  </summary>

  <div class="rec-hint" id="capture-note" style="margin-top:0.7rem;">
    Device, rate, block size and channels define the capture stream. Changing
    one mid-session reopens it, which costs about a second of audio.
  </div>
  <div class="rec-fields wide" style="margin-top:0.5rem;">
    <label class="rec-field">
      <span data-tip="PortAudio input. Pin a stable name substring rather than 'default' — OS default inputs change.">Device</span>
      <select id="f-device"></select>
    </label>
    <label class="rec-field">
      <span data-tip="Capture rate of the card. Clips are stored at this rate; training and inference resample at the model boundary.">Sample rate</span>
      <select id="f-sample-rate">
        {% for sr in sample_rates %}<option value="{{ sr }}">{{ '{:,}'.format(sr) }} Hz</option>{% endfor %}
      </select>
    </label>
    <label class="rec-field">
      <span data-tip="PortAudio callback chunk. Raise it if the log reports sustained input overflow.">Block size</span>
      <select id="f-blocksize">
        {% for bs in blocksizes %}<option value="{{ bs }}">{{ bs }} frames</option>{% endfor %}
      </select>
    </label>
  </div>

  <h4 style="margin:0.9rem 0 0.2rem; font-size:0.8rem; text-transform:uppercase;
             letter-spacing:0.02em; color:var(--tx-2);">Channels</h4>
  <div class="rec-hint">
    Zero-based physical indexes — interface input N is index N-1 (input 1 → ch0, input 4 → ch3).
    Tick a channel to record it; the level next to it updates live so you can see which inputs are alive.
  </div>
  <div class="chan-pick" id="chan-pick"></div>
  <div class="group-foot">
    <span style="display:inline-flex; gap:0.4rem;">
      <button class="btn small" id="btn-meter" data-tip="Open the input stream and show levels without recording anything">Start meter</button>
      <button class="btn small" id="btn-preflight" data-tip="Run the pre-record checklist: device, channel count, sample rate, dead mics, clipping, dropouts">Pre-flight check</button>
    </span>
    <button class="btn small" data-save="capture" data-tip="Write device, sample rate, block size and channels into projects/&lt;id&gt;/config.toml">Save to config.toml</button>
  </div>
  <div class="checks" id="checks"></div>
</details>

{# ---- B. Session: editable while recording ------------------------------ #}
<div class="rec-panel rec-band">
  <h3><span>Session</span><span class="section-status" id="session-status"></span></h3>

  <div class="rec-transport">
    <button class="rec-go" id="btn-go">Start recording</button>
    <span class="rec-counter" id="counter">0 <small>clips</small></span>
    <span class="seg" id="seg-mode">
      <button data-mode="onset" data-tip="Save a clip per detected transient. The normal way to collect target sounds.">Onset</button>
      <button data-mode="continuous" data-tip="Gate off: save back-to-back clips that tile the timeline. The fast way to collect room tone, crowd beds and calibration negatives.">Continuous</button>
      <button data-mode="manual" data-tip="Nothing is saved until you press Space or Capture now.">Manual</button>
    </span>
    <button class="btn" id="btn-capture" data-tip="Capture around this instant, ignoring the gate and the cooldown (Space)">Capture now</button>
    <button class="btn danger" id="btn-cancel" hidden
            data-tip="Stop this pass and delete every clip it recorded — files, sidecars and manifest rows">Discard run</button>
  </div>

  <div class="rec-fields wide">
    <label class="rec-field">
      <span data-tip="Clips are saved under data_library/sounds/&lt;label&gt;/. Changing it mid-session starts a new recording group.">Label</span>
      <input type="text" id="f-label" list="known-labels" placeholder="e.g. cheers_quiet" autocomplete="off">
      <datalist id="known-labels"></datalist>
    </label>
    <label class="rec-field">
      <span data-tip="Free-form venue or site note saved in every sidecar, e.g. 'bar-zone-1'.">Location</span>
      <input type="text" id="f-location" list="known-locations" placeholder="rnd-office"
             autocomplete="off">
      <datalist id="known-locations"></datalist>
    </label>
    <label class="rec-field">
      <span data-tip="Where the microphone sits. Recorded per clip so mic geometry stays traceable.">Mic position</span>
      <input type="text" id="f-mic-position" placeholder="above bar, 1.5 m">
    </label>
    <label class="rec-field">
      <span data-tip="Free-form note saved in every sidecar, e.g. 'half-full wine glass'.">Description</span>
      <input type="text" id="f-description" placeholder="thin wine glass, half full">
    </label>
    <label class="rec-field">
      <span data-tip="Comma-separated tags saved in every sidecar.">Tags</span>
      <input type="text" id="f-tags" list="known-tags" placeholder="wine-glass,quiet"
             autocomplete="off">
      <datalist id="known-tags"></datalist>
    </label>
    <label class="rec-field">
      <span data-tip="One name per recorded channel, in channel order, e.g. 'rode,samson'. Saved per clip and inherited by each mono split, so per-microphone analysis stays honest. Leave empty to use the project's configured microphones.">Microphones</span>
      <input type="text" id="f-mics" placeholder="rode,samson">
    </label>
    <label class="rec-field">
      <span data-tip="Stop after this many clips. 0 records until you stop or the time cap runs out.">Target clips</span>
      <input type="number" id="f-num" min="0" step="1" value="0">
    </label>
    <label class="rec-field">
      <span data-tip="Wall-clock cap, e.g. 15m or 2h. Empty means no limit.">Time cap</span>
      <input type="text" id="f-duration" placeholder="15m">
    </label>
  </div>

  <div style="display:flex; gap:1.2rem; align-items:center; flex-wrap:wrap; margin-top:0.7rem;">
    <span style="display:inline-flex; align-items:center; gap:0.5rem;">
      <span style="font-size:0.68rem; font-weight:700; text-transform:uppercase;
                   letter-spacing:0.02em; color:var(--tx-muted);"
            data-tip="Noise context of the room. Recorded as metadata, never as a class — intensity is a nuisance variable the model should be invariant to.">Density</span>
      <span class="seg" id="seg-density">
        {% for d in densities %}<button data-density="{{ d }}">{{ d }}</button>{% endfor %}
      </span>
    </span>
    <label class="rec-check" data-tip="Tags clips 'holdout' so a frozen-test source rule can select them. A recording carries no split by itself — membership is a composition decision.">
      <input type="checkbox" id="f-holdout"> Frozen-test material
    </label>
    <label class="rec-check" data-tip="Also write per-channel mono clips as each multichannel clip is saved, with sidecars linking back to the parent.">
      <input type="checkbox" id="f-split"> Split channels on save
    </label>
  </div>
  <div class="rec-hint" id="session-target"></div>
</div>

{# ---- C. Live: meters left, gate right ---------------------------------- #}
<div class="rec-live rec-band">
  <div style="display:grid; gap:0.85rem; min-width:0;">
    <div class="rec-panel">
      <h3><span>Microphones</span><span class="section-status" id="mic-status"></span></h3>
      <div class="meter-list" id="meters">
        <div class="rec-empty">Start the meter or a session to see live levels.</div>
      </div>
    </div>

    <div class="rec-panel">
      <h3>
        <span>Recent takes</span>
        <span class="section-head"><span class="section-status" id="recent-status"></span></span>
      </h3>
      <div class="recent" id="recent"><div class="rec-empty">Nothing recorded yet.</div></div>
      <h4 style="margin:0.9rem 0 0.35rem; font-size:0.8rem; text-transform:uppercase;
                 letter-spacing:0.02em; color:var(--tx-2);">Session log</h4>
      <div class="rec-log" id="log"><span class="rec-empty">—</span></div>
    </div>

    {# Every pass made since this server started. In memory only — it is a
       working list for the session, not a second index of the library — so it
       exists to answer "was that last pass any good?" while the answer still
       matters, and to throw a bad one away in one action. #}
    <div class="rec-panel">
      <h3>
        <span class="tip" data-tip="Passes recorded since this server started. Deleting one removes its clips from the library — files, sidecars and manifest rows. Closing the console forgets the list, not the clips.">This server run</span>
        <span class="section-status" id="runs-status"></span>
      </h3>
      <div class="runs" id="runs"><div class="rec-empty">No passes recorded yet.</div></div>
    </div>
  </div>

  <div class="rec-panel" style="min-width:0;">
    <h3><span>Onset gate &amp; clip</span><span class="section-status" id="onset-status"></span></h3>

    <div class="knob">
      <div class="knob-head">
        <label for="k-floor" data-tip="Absolute floor on spectral flux. The adaptive MAD threshold follows slow room changes; this floor stops tiny continuous background motion from becoming recordings.\n\nThe effective gate is **max(floor, median + mad_k·MAD)**.">Onset floor (min_flux_db)</label>
        <span class="knob-val" id="v-floor"></span>
      </div>
      <input type="range" id="k-floor" min="-110" max="-20" step="1">
      <div class="knob-note" id="cal-note">Lower catches quieter onsets.</div>
      <div class="group-foot" style="border:none; padding:0; margin-top:0.35rem;">
        <span class="seg" id="seg-strict">
          <button data-strict="relaxed" data-tip="Floor just above the room's median. Catches the quietest events, but movement and handling will trigger too.">relaxed</button>
          <button data-strict="normal" data-tip="Floor above 99% of what the room does on its own. Ignores incidental noise; the usual choice.">normal</button>
          <button data-strict="strict" data-tip="Floor above almost everything the room does. Only deliberate, distinct sounds trigger.">strict</button>
        </span>
        <button class="btn small" id="btn-calibrate"
                data-tip="Listen to the room for 10 s and place the floor above what it does on its own.\n\n**Move around while it listens** — walk, set a glass down, handle the stand. Whatever it hears becomes what it ignores.">Auto-calibrate (10 s)</button>
      </div>
    </div>

    <div class="knob">
      <div class="knob-head">
        <label for="k-madk" data-tip="How far above the rolling median (in MADs) the flux must rise. Lower is more sensitive.">Sensitivity (mad_k)</label>
        <span class="knob-val" id="v-madk"></span>
      </div>
      <input type="range" id="k-madk" min="1" max="10" step="0.1">
    </div>

    <div class="knob">
      <div class="knob-head">
        <label for="k-cooldown" data-tip="Global delay before ANY mic can start another recording. Stops one physical event becoming several near-duplicate clips.">Cooldown</label>
        <span class="knob-val" id="v-cooldown"></span>
      </div>
      <input type="range" id="k-cooldown" min="0" max="5000" step="50">
    </div>

    <div class="knob">
      <div class="knob-head">
        <label for="k-clip" data-tip="Saved clip length. Deliberately longer than the model window so one library serves projects at any window length — the window is cropped around the onset later.">Clip length</label>
        <span class="knob-val" id="v-clip"></span>
      </div>
      <input type="range" id="k-clip" min="0.2" max="4" step="0.05">

      {# Where the transient sits inside the saved clip. The model window is
         cropped around the onset, not the clip centre, so this is the geometry
         that decides what the model is actually trained on. #}
      <div class="geom" id="geom">
        <div class="geom-track">
          <div class="geom-pre-fill" id="geom-pre-fill"></div>
          <div class="geom-jitter" id="geom-jitter"></div>
          <div class="geom-onset" id="geom-onset-lo"></div>
          <div class="geom-onset" id="geom-onset-hi" hidden></div>
        </div>
        <div class="geom-scale"><span>0</span><span id="geom-end"></span></div>
        <div class="geom-legend">
          <span><i class="sw pre"></i>preroll <b id="geom-pre"></b></span>
          <span id="geom-jitter-key"><i class="sw jit"></i>onset lands here</span>
          <span><i class="sw post"></i>postroll <b id="geom-post"></b></span>
        </div>
      </div>
      <div class="knob-note" id="geom-note"></div>
    </div>

    <div class="knob">
      <div class="knob-head">
        <label data-tip="How far into the clip the detected transient sits. Randomizing it spreads the event across the clip, which is why time-shift augmentation is not needed later — the jitter is baked into the recording instead of wrapped or zero-padded afterwards.">Preroll</label>
        <span class="knob-val" id="v-preroll"></span>
      </div>

      <label class="rec-check" style="margin-bottom:0.4rem;">
        <input type="checkbox" id="k-rand"> Randomize per clip
      </label>

      {# Fixed preroll: one value for every clip. #}
      <div id="preroll-fixed">
        <input type="range" id="k-preroll" min="0" step="10">
      </div>

      {# Randomized: the window each clip's preroll is drawn from. Both ends
         are tunable — a range that never moves is the same as a fixed preroll,
         and one that reaches the end of the clip leaves no postroll. #}
      <div id="preroll-range" hidden>
        <div class="range-row">
          <span class="range-tag">min</span>
          <input type="range" id="k-pre-lo" min="0" step="10">
          <span class="range-val" id="v-pre-lo"></span>
        </div>
        <div class="range-row">
          <span class="range-tag">max</span>
          <input type="range" id="k-pre-hi" min="0" step="10">
          <span class="range-val" id="v-pre-hi"></span>
        </div>
      </div>
      <div class="knob-note" id="preroll-note"></div>
    </div>

    <div class="rec-fields">
      <label class="rec-field">
        <span data-tip="With 'any' each selected mic runs its own detector and any of them can trigger; near-simultaneous hits merge into one clip and only the mics that fired are saved.">Trigger mode</span>
        <select id="k-trigger">
          {% for value, text in trigger_modes %}<option value="{{ value }}">{{ text }}</option>{% endfor %}
        </select>
      </label>
      <label class="rec-field">
        <span data-tip="Window for collapsing the same physical transient heard on several mics into a single clip.">Cross-mic merge</span>
        <input type="number" id="k-merge" min="0" max="2000" step="10">
      </label>
    </div>

    <div class="group-foot">
      <span class="knob-note">Changes apply live; the stream keeps running.</span>
      <button class="btn small" data-save="onset"
              data-tip="Write these onset values into projects/&lt;id&gt;/config.toml, preserving its comments">Save to config.toml</button>
    </div>
  </div>
</div>

<div class="take-modal" id="take-modal" hidden>
  <div class="take-modal-card">
    <div class="take-modal-head">
      <strong id="tm-name"></strong>
      <span class="take-modal-meta" id="tm-meta"></span>
      <span class="seg" id="tm-channels"></span>
      <button class="btn small" id="tm-play">▶ Play</button>
      <button class="btn small" id="tm-close">Close</button>
    </div>
    <img id="tm-image" alt="linear spectrogram">
    <div id="tm-verdict"></div>
  </div>
</div>

{{ tooltip_element|safe }}
{{ toast_element|safe }}
<script id="boot-json" type="application/json">{{ boot_json|safe }}</script>
<script>
{{ shell_js|safe }}
{{ record_js|safe }}
</script>
</body>
</html>
"""


_RECORD_JS = r"""
// ---------------------------------------------------------------------------
// Recording console. Vanilla JS, one polling loop, no build step — same shape
// as the inspect UI so the two stay readable side by side.
// ---------------------------------------------------------------------------

const BOOT = JSON.parse(document.getElementById("boot-json").textContent);
const THEME_KEY = "sed-theme";

// Poll fast enough that a trigger flash and a level peak are actually visible
// while recording, slow enough to stay cheap when the page is just sitting open.
const POLL_RECORDING_MS = 400;
const POLL_IDLE_MS = 800;
// Meters get their own loop against a tiny endpoint. The full status carries
// recent takes, the session log and the run list — far too much to fetch at the
// rate a VU meter has to move to look live. 80 ms is 12 frames a second, which
// reads as continuous once the bar eases between samples; faster than this only
// adds jitter, because the eye cannot resolve it anyway.
const POLL_LEVELS_MS = 80;

let recording = false;
let devices = [];
let deviceError = "";
let channelChoice = BOOT.capture.channels.slice();
let densityChoice = BOOT.form.density || "normal";
let modeChoice = BOOT.form.mode || "onset";
let onset = Object.assign({}, BOOT.onset);
let lastSnapshot = null;
let pollTimer = null;
let meterOn = false;
let busy = false;
// The engine can finish on its own (target or time cap). The poll that notices
// has to release the server-side session exactly once, not on every tick until
// the state catches up.
let autoStopped = false;
let knownTags = [];
let strictnessChoice = "normal";

const $ = (id) => document.getElementById(id);
const fmtDb = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : v.toFixed(1);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, options) {
    const r = await fetch(path, Object.assign({ cache: "no-store" }, options || {}));
    let data = {};
    try { data = await r.json(); } catch (e) { /* non-JSON body */ }
    if (!r.ok || data.ok === false) {
        throw new Error(data.error || `HTTP ${r.status}`);
    }
    return data;
}

async function post(path, body) {
    return api(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
    });
}

function fail(err, where) {
    console.error(where || "request failed", err);
    toast(err.message || String(err), "err");
}

// ---- theme -----------------------------------------------------------------

function initThemeToggle() {
    const btn = $("theme-toggle");
    const current = () => document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const apply = (t) => {
        if (t === "dark") document.documentElement.setAttribute("data-theme", "dark");
        else document.documentElement.removeAttribute("data-theme");
    };
    btn.addEventListener("click", () => {
        const next = current() === "dark" ? "light" : "dark";
        apply(next);
        try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* no storage */ }
    });
    // Follow the OS while the user has not made an explicit choice.
    try {
        const mq = window.matchMedia("(prefers-color-scheme: dark)");
        mq.addEventListener("change", (ev) => {
            let saved = null;
            try { saved = localStorage.getItem(THEME_KEY); } catch (e) { /* ignore */ }
            if (!saved) apply(ev.matches ? "dark" : "light");
        });
    } catch (e) { /* matchMedia unavailable */ }
}

// ---- setup: devices and channels -------------------------------------------

function deviceByName(name) {
    const needle = String(name || "").toLowerCase();
    const exact = devices.filter((d) => d.name.toLowerCase() === needle);
    if (exact.length === 1) return exact[0];
    const hits = devices.filter((d) => d.name.toLowerCase().includes(needle));
    return hits.length ? hits[0] : null;
}

function selectedDevice() {
    const sel = $("f-device");
    return deviceByName(sel.value) || devices[0] || null;
}

function renderDevices() {
    const sel = $("f-device");
    sel.innerHTML = devices.map((d) =>
        `<option value="${esc(d.name)}">[${d.index}] ${esc(d.name)} — ${d.max_input_channels} in</option>`
    ).join("");
    const match = deviceByName(BOOT.capture.device);
    if (match) sel.value = match.name;
    if (deviceError) {
        $("setup-status").className = "section-status err";
        $("setup-status").textContent = deviceError;
    }
    renderChannelPicker();
}

function renderChannelPicker() {
    const dev = selectedDevice();
    const max = dev ? dev.max_input_channels : (Math.max(0, ...channelChoice) + 1);
    const mics = BOOT.capture.microphone_type || [];
    const rows = [];
    for (let c = 0; c < max; c++) {
        const on = channelChoice.includes(c);
        const order = channelChoice.indexOf(c);
        const mic = on && order >= 0 && order < mics.length ? mics[order] : "";
        rows.push(
            `<label class="chan-row">
               <input type="checkbox" data-chan="${c}" ${on ? "checked" : ""}>
               <span class="ci">ch${c}</span>
               <span class="cm">${esc(mic)}${on ? "" : '<span style="opacity:.6"> — not recorded</span>'}</span>
               <span class="cd" data-chan-level="${c}">—</span>
             </label>`);
    }
    $("chan-pick").innerHTML = rows.join("") ||
        '<div class="rec-empty">No input channels reported by this device.</div>';
    $("chan-pick").querySelectorAll("input[data-chan]").forEach((box) => {
        box.addEventListener("change", () => {
            const c = Number(box.dataset.chan);
            // Keep the click order: record_channels is an ordered selection, and
            // channel N of the saved WAV is whatever sits at position N here.
            if (box.checked) {
                if (!channelChoice.includes(c)) channelChoice.push(c);
            } else {
                channelChoice = channelChoice.filter((x) => x !== c);
            }
            if (!channelChoice.length) {
                channelChoice = [c];
                box.checked = true;
                toast("at least one channel has to be recorded", "warn");
            }
            renderChannelPicker();
            pushCapture();
        });
    });
}

function captureFields() {
    const dev = selectedDevice();
    return {
        device: dev ? dev.name : BOOT.capture.device,
        sample_rate: Number($("f-sample-rate").value),
        blocksize: Number($("f-blocksize").value),
        channels: channelChoice.slice(),
    };
}

async function pushCapture() {
    try {
        await post("/api/record/settings", { capture: captureFields() });
        if (meterOn) { await stopMeter(); await startMeter(); }
    } catch (err) { fail(err, "capture settings"); }
}

async function loadDevices() {
    try {
        const data = await api("/api/record/devices");
        devices = data.devices || [];
        deviceError = data.device_error || "";
        BOOT.capture.microphone_type = data.microphone_type || [];
        channelChoice = (data.channels || channelChoice).slice();
        $("f-sample-rate").value = String(data.sample_rate);
        $("f-blocksize").value = String(data.blocksize);
        $("known-labels").innerHTML = (data.labels || []).map((l) =>
            `<option value="${esc(l.label)}">${l.clips} clips</option>`).join("");
        $("known-locations").innerHTML = (data.locations || []).map((l) =>
            `<option value="${esc(l)}"></option>`).join("");
        // Tags are a comma-separated field, so completing one has to keep the
        // ones already typed — the datalist offers whole values, not fragments.
        knownTags = data.tags || [];
        renderTagOptions();
        renderProjects(data.projects || [], data.project || "");
        renderDevices();
        renderLabelTarget(data.labels || []);
    } catch (err) { fail(err, "devices"); }
}

function renderTagOptions() {
    // Suggest each known tag appended to what is already in the field, so
    // picking one from the list adds to the set rather than replacing it.
    const raw = $("f-tags").value;
    const cut = raw.lastIndexOf(",");
    const prefix = cut >= 0 ? raw.slice(0, cut + 1) : "";
    const chosen = new Set(raw.split(",").map((t) => t.trim()).filter(Boolean));
    $("known-tags").innerHTML = knownTags
        .filter((t) => !chosen.has(t.tag))
        .map((t) => `<option value="${esc(prefix + t.tag)}">${t.tag} — ${t.clips} clips</option>`)
        .join("");
}

function renderProjects(projects, selected) {
    const sel = $("f-project");
    sel.innerHTML = ['<option value="">— none —</option>']
        .concat(projects.map((p) => `<option value="${esc(p)}">${esc(p)}</option>`))
        .join("");
    sel.value = selected || "";
}

async function selectProject(project) {
    try {
        await post("/api/record/project", { project: project });
        // The project supplies the venue's device, channels and onset settings,
        // so re-read everything rather than guessing what changed.
        await loadDevices();
        const st = await api("/api/record/status");
        if (!st.recording) { onset = Object.assign({}, st.onset); syncKnobs(); }
        toast(project ? `recording for ${project}` : "recording to the shared library", "ok");
        schedulePoll(0);
    } catch (err) {
        fail(err, "project");
        loadDevices();
    }
}

function renderLabelTarget(labels) {
    const label = $("f-label").value.trim();
    const hit = labels.find((l) => l.label === label);
    $("session-target").textContent = hit
        ? `${hit.clips} clip(s) already in the library under "${label}".`
        : (label ? `New label — "${label}" has no clips yet.` : "");
}

// ---- pre-flight ------------------------------------------------------------

async function runPreflight(btn) {
    btn.disabled = true;
    btn.textContent = "Checking…";
    try {
        const data = await api("/api/record/preflight?seconds=3");
        $("checks").innerHTML = (data.checks || []).map((c) =>
            `<div class="check ${esc(c.status)}">
               <span class="st">${c.status === "PASS" ? "✓" : c.status === "WARN" ? "!" : "✕"}</span>
               <span class="nm">${esc(c.name)}</span>
               <span class="dt">${esc(c.detail)}</span>
               ${c.fix ? `<span class="fx">${esc(c.fix)}</span>` : ""}
             </div>`).join("");
        const cls = data.worst === "PASS" ? "ok" : data.worst === "WARN" ? "warn" : "err";
        $("setup-status").className = "section-status " + cls;
        $("setup-status").textContent = data.worst === "PASS" ? "ready" : data.worst.toLowerCase();
    } catch (err) {
        fail(err, "preflight");
    } finally {
        btn.disabled = false;
        btn.textContent = "Pre-flight check";
    }
}

// ---- meter (idle only) -----------------------------------------------------

async function startMeter() {
    try {
        await post("/api/record/meter/start", captureFields());
        meterOn = true;
        $("btn-meter").textContent = "Stop meter";
    } catch (err) { fail(err, "meter"); }
}

async function stopMeter() {
    try {
        await post("/api/record/meter/stop", {});
    } catch (err) { fail(err, "meter"); }
    meterOn = false;
    $("btn-meter").textContent = "Start meter";
}

// ---- transport -------------------------------------------------------------

function sessionFields() {
    return {
        label: $("f-label").value.trim(),
        density: densityChoice,
        mode: modeChoice,
        location: $("f-location").value,
        mic_position: $("f-mic-position").value,
        description: $("f-description").value,
        tags: $("f-tags").value,
        microphone_type: $("f-mics").value,
        holdout: $("f-holdout").checked,
        split_channels: $("f-split").checked,
        num: Number($("f-num").value || 0),
        duration: $("f-duration").value,
    };
}

async function startSession() {
    if (!$("f-label").value.trim()) {
        toast("give the clips a label first", "warn");
        $("f-label").focus();
        return;
    }
    busy = true;
    // Start can block for a few seconds on a cold process while the clip
    // writer's imports finish, so say so rather than looking frozen.
    const go = $("btn-go");
    go.disabled = true;
    go.textContent = "Starting…";
    try {
        if (meterOn) await stopMeter();
        autoStopped = false;
        await post("/api/record/start",
            Object.assign({}, sessionFields(), captureFields(), { onset: onset }));
        recording = true;
        $("setup-panel").open = false;
        toast("recording", "ok");
    } catch (err) {
        fail(err, "start");
    } finally {
        busy = false;
        go.disabled = false;
        schedulePoll(0);
    }
}

async function stopSession() {
    busy = true;
    $("btn-go").disabled = true;
    try {
        const data = await post("/api/record/stop", {});
        recording = false;
        const n = (data.summary || {}).clips || 0;
        toast(`stopped — ${n} clip(s) saved`, "ok");
    } catch (err) {
        fail(err, "stop");
    } finally {
        busy = false;
        $("btn-go").disabled = false;
        schedulePoll(0);
    }
}

async function captureNow() {
    if (!recording) { toast("start a session first", "warn"); return; }
    try {
        await post("/api/record/trigger", {});
    } catch (err) { fail(err, "capture"); }
}

async function undoLast() {
    const take = (lastSnapshot && lastSnapshot.recent || [])[0];
    if (!take) { toast("nothing to undo", "warn"); return; }
    await deleteTake(take.uuid, take.name);
}

async function deleteTake(uuid, name) {
    try {
        await post("/api/record/clip/delete", { uuid: uuid });
        toast(`deleted ${name}`, "ok");
        schedulePoll(0);
    } catch (err) { fail(err, "delete"); }
}

// ---- live settings ---------------------------------------------------------

let pushTimer = null;
function pushOnset(patch) {
    Object.assign(onset, patch);
    // Dragging a slider fires continuously; coalesce so one gesture is one
    // retune rather than fifty.
    clearTimeout(pushTimer);
    pushTimer = setTimeout(async () => {
        try {
            await post("/api/record/settings", { onset: onset });
            $("onset-status").className = "section-status ok";
            $("onset-status").textContent = "applied";
            setTimeout(() => { $("onset-status").textContent = ""; }, 1200);
        } catch (err) {
            $("onset-status").className = "section-status err";
            $("onset-status").textContent = err.message;
            fail(err, "onset");
        }
    }, 120);
}

function pushMetadata(patch) {
    post("/api/record/settings", { metadata: patch }).catch((err) => fail(err, "metadata"));
}

async function saveGroup(btn) {
    const group = btn.dataset.save;
    btn.disabled = true;
    try {
        await post("/api/record/save-settings", { group: group });
        toast(`${group} settings written to config.toml`, "ok");
    } catch (err) {
        fail(err, "save");
    } finally {
        btn.disabled = false;
    }
}

async function calibrate(btn) {
    // Works from the idle meter as well as a running session — setting the gate
    // is something to do before the first take.
    if (!recording && !meterOn) {
        try { await startMeter(); } catch (e) { /* reported below */ }
    }
    btn.disabled = true;
    try {
        await post("/api/record/calibrate",
                   { seconds: 10, strictness: strictnessChoice });
        $("cal-note").textContent =
            "Listening — move around normally, so the floor learns to ignore you.";
    } catch (err) {
        fail(err, "calibrate");
        btn.disabled = false;
    }
}

function setStrictness(value) {
    strictnessChoice = value;
    $("seg-strict").querySelectorAll("button").forEach((b) =>
        b.classList.toggle("on", b.dataset.strict === value));
}

// ---- knobs -----------------------------------------------------------------

function syncKnobs() {
    $("k-floor").value = onset.min_flux_db;
    $("v-floor").textContent = `${Number(onset.min_flux_db).toFixed(0)} dB`;
    $("k-madk").value = onset.mad_k;
    $("v-madk").textContent = Number(onset.mad_k).toFixed(1);
    $("k-cooldown").value = onset.recording_cooldown_ms;
    $("v-cooldown").textContent = `${onset.recording_cooldown_ms} ms`;
    $("k-clip").value = onset.record_seconds;
    $("k-rand").checked = !!onset.randomize_preroll;
    $("k-trigger").value = onset.multi_mic_trigger;
    $("k-merge").value = onset.cross_channel_merge_ms;
    syncPrerollControls();
    renderClipGeometry();
}

// The preroll has to stay inside the clip: a preroll at or past the end would
// leave zero postroll, which the server refuses. Every bound below is therefore
// derived from the current clip length rather than hardcoded.
function clipMs() { return Math.round(Number(onset.record_seconds) * 1000); }
function prerollCeiling() { return Math.max(10, clipMs() - 10); }

function syncPrerollControls() {
    const random = !!onset.randomize_preroll;
    const ceiling = prerollCeiling();
    $("preroll-fixed").hidden = random;
    $("preroll-range").hidden = !random;

    $("k-preroll").max = ceiling;
    $("k-preroll").value = onset.preroll_ms;
    $("k-pre-lo").max = ceiling;
    $("k-pre-hi").max = ceiling;
    $("k-pre-lo").value = onset.preroll_ms_range[0];
    $("k-pre-hi").value = onset.preroll_ms_range[1];
    $("v-pre-lo").textContent = `${onset.preroll_ms_range[0]} ms`;
    $("v-pre-hi").textContent = `${onset.preroll_ms_range[1]} ms`;

    $("v-preroll").textContent = random
        ? `${onset.preroll_ms_range[0]}–${onset.preroll_ms_range[1]} ms`
        : `${onset.preroll_ms} ms`;
    $("preroll-note").textContent = random
        ? "Each clip draws its own preroll from this window, so the transient "
          + "lands in a different place every time."
        : "Every clip puts the transient at the same offset.";
}

// Keep the preroll and its jitter window inside the clip and in order.
// `moved` says which end the operator just dragged, so the other one yields.
function clampPreroll(moved) {
    const ceiling = prerollCeiling();
    onset.preroll_ms = Math.max(0, Math.min(onset.preroll_ms, ceiling));
    let [lo, hi] = onset.preroll_ms_range.map(Number);
    lo = Math.max(0, Math.min(lo, ceiling));
    hi = Math.max(0, Math.min(hi, ceiling));
    if (lo > hi) {
        if (moved === "hi") lo = hi;
        else hi = lo;
    }
    onset.preroll_ms_range = [Math.round(lo), Math.round(hi)];
}

function renderClipGeometry() {
    const sr = Number($("f-sample-rate").value) || BOOT.capture.sample_rate;
    const secs = Number(onset.record_seconds);
    const ms = clipMs();
    $("v-clip").textContent = `${secs.toFixed(2)} s`;
    $("geom-note").textContent =
        `${secs.toFixed(2)} s · ${Math.round(secs * sr).toLocaleString()} frames `
        + `@ ${sr.toLocaleString()} Hz`;

    const random = !!onset.randomize_preroll;
    const lo = random ? onset.preroll_ms_range[0] : onset.preroll_ms;
    const hi = random ? onset.preroll_ms_range[1] : onset.preroll_ms;
    const jitter = random && hi > lo;
    const pct = (v) => Math.max(0, Math.min(100, (v / ms) * 100));

    // 0..lo is preroll on every clip; lo..hi is where a randomized onset can
    // land; the rest of the track is postroll.
    $("geom-pre-fill").style.width = pct(lo) + "%";
    $("geom-jitter").style.left = pct(lo) + "%";
    $("geom-jitter").style.width = (jitter ? pct(hi) - pct(lo) : 0) + "%";
    $("geom-onset-lo").style.left = `calc(${pct(lo)}% - 1px)`;
    $("geom-onset-hi").style.left = `calc(${pct(hi)}% - 1px)`;
    $("geom-onset-hi").hidden = !jitter;
    $("geom-jitter-key").hidden = !jitter;

    $("geom-end").textContent = `${secs.toFixed(2)} s`;
    $("geom-pre").textContent = jitter ? `${lo}–${hi} ms` : `${lo} ms`;
    $("geom-post").textContent = jitter
        ? `${ms - hi}–${ms - lo} ms`
        : `${ms - lo} ms`;
}

// ---- live rendering --------------------------------------------------------

function barClass(db) { return db >= -3 ? "hot" : db >= -12 ? "warm" : ""; }
function barWidth(db) { return Math.max(0, Math.min(100, ((db + 60) / 60) * 100)); }

function sparkPath(history) {
    if (!history || history.length < 2) return "";
    const w = 100, h = 22;
    const pts = history.map((db, i) => {
        const x = (i / (history.length - 1)) * w;
        const y = h - Math.max(0, Math.min(1, (db + 60) / 60)) * h;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return "M" + pts.join(" L");
}

// Meter rows are built once and then mutated in place. Rebuilding innerHTML at
// the poll rate destroyed and recreated every element 16 times a second, which
// made the numbers unreadable, killed text selection, and — because the same
// loop redrew the calibration panel — meant the Apply button was replaced out
// from under the pointer before a click could land on it.
let meterRows = new Map();
let meterSignature = "";
// Displayed level is smoothed towards the measured one: instant on the way up
// so a transient is never missed, eased on the way down so the bar glides.
const METER_ATTACK = 1.0;
const METER_RELEASE = 0.35;
// Numbers are only legible if they hold still; the bars stay at the poll rate.
const METER_TEXT_MS = 120;
let meterTextAt = 0;

function meterSmooth(previous, next) {
    if (previous === undefined || !isFinite(previous)) return next;
    const k = next > previous ? METER_ATTACK : METER_RELEASE;
    return previous + (next - previous) * k;
}

function buildMeterRows(channels, gated) {
    const el = $("meters");
    el.innerHTML = "";
    meterRows = new Map();
    channels.forEach((c) => {
        const row = document.createElement("div");
        row.className = "meter";
        row.innerHTML =
            `<div class="meter-head">
               <span class="meter-ch">ch${c.channel}</span>
               <span class="meter-mic">${esc(c.mic_label || "")}</span>
               <span class="meter-badges"></span>
               <span class="meter-db">—</span>
             </div>
             <div class="bar">
               <div class="bar-peak"></div>
               <div class="bar-rms"></div>
             </div>
             ${gated ? '<div class="meter-gate"></div><div class="meter-gate meter-clips"></div>' : ""}
             <svg class="meter-spark" viewBox="0 0 100 22" preserveAspectRatio="none" hidden><path></path></svg>`;
        el.appendChild(row);
        meterRows.set(c.channel, {
            row: row,
            db: row.querySelector(".meter-db"),
            peak: row.querySelector(".bar-peak"),
            rms: row.querySelector(".bar-rms"),
            badges: row.querySelector(".meter-badges"),
            gate: row.querySelector(".meter-gate"),
            clips: row.querySelector(".meter-clips"),
            spark: row.querySelector(".meter-spark"),
            sparkPath: row.querySelector(".meter-spark path"),
            mic: row.querySelector(".meter-mic"),
            shown: undefined, badgeKey: "", gateKey: "", clipKey: "",
        });
    });
    meterSignature = channels.map((c) => c.channel).join(",") + (gated ? "|g" : "");
}

function renderMeters(channels, opts) {
    const el = $("meters");
    if (!channels || !channels.length) {
        if (meterSignature !== "empty") {
            el.innerHTML = '<div class="rec-empty">Start the meter or a session to see live levels.</div>';
            meterRows = new Map();
            meterSignature = "empty";
        }
        return;
    }
    const signature = channels.map((c) => c.channel).join(",") + (opts.gated ? "|g" : "");
    if (signature !== meterSignature) buildMeterRows(channels, opts.gated);

    const now = performance.now();
    const withText = now - meterTextAt >= METER_TEXT_MS;
    if (withText) meterTextAt = now;

    channels.forEach((c) => {
        const r = meterRows.get(c.channel);
        if (!r) return;
        r.shown = meterSmooth(r.shown, c.peak_dbfs);
        r.peak.style.width = barWidth(r.shown) + "%";
        r.peak.className = "bar-peak " + barClass(r.shown);
        r.rms.style.left = barWidth(c.rms_dbfs) + "%";
        r.row.classList.toggle("fired", !!c.flash);
        r.row.classList.toggle("silent", !!c.silent);
        r.row.classList.toggle("clipping", !!c.clipping);

        if (!withText) return;
        r.db.textContent = `${fmtDb(r.shown)} dBFS`;
        if (r.mic.textContent !== (c.mic_label || "")) r.mic.textContent = c.mic_label || "";

        const badgeKey = `${c.silent ? 1 : 0}${c.clipping ? 1 : 0}`;
        if (badgeKey !== r.badgeKey) {
            r.badgeKey = badgeKey;
            const badges = [];
            if (c.silent) badges.push('<span class="pill bad" data-tip="This channel never rose above the preamp noise floor. Channels are 0-based — interface input N is index N-1. Check the index or the connection.">silent</span>');
            if (c.clipping) badges.push('<span class="pill bad" data-tip="Peaking at full scale — the transient is already destroyed. Lower the input gain.">clipping</span>');
            r.badges.innerHTML = badges.join("");
        }
        if (r.gate) {
            const gateKey = `${c.flux_db}|${c.gate_db}|${c.will_trigger}|${c.gate_source}`;
            if (gateKey !== r.gateKey) {
                r.gateKey = gateKey;
                r.gate.innerHTML = (c.flux_db === null || c.flux_db === undefined)
                    ? '<span class="no">·</span> warming up'
                    : `<span class="${c.will_trigger ? "go" : "no"}">${c.will_trigger ? "✓" : "✕"}</span>`
                      + ` flux ${fmtDb(c.flux_db)} dB vs gate ${fmtDb(c.gate_db)} dB`
                      + ` <span style="opacity:.7">(${esc(c.gate_source || "")})</span>`;
            }
        }
        if (r.clips && c.clips !== undefined && String(c.clips) !== r.clipKey) {
            r.clipKey = String(c.clips);
            r.clips.textContent = `${c.clips} clip(s) from this mic`;
        }
        if (c.peak_history && c.peak_history.length > 1) {
            r.spark.hidden = false;
            r.sparkPath.setAttribute("d", sparkPath(c.peak_history));
        }
        const cell = document.querySelector(`[data-chan-level="${c.channel}"]`);
        if (cell) cell.textContent = `${fmtDb(c.peak_dbfs)} dB`;
    });
}

function renderRecent(takes) {
    const el = $("recent");
    if (!takes || !takes.length) {
        el.innerHTML = '<div class="rec-empty">Nothing recorded yet.</div>';
        return;
    }
    el.innerHTML = takes.map((t) => {
        const src = "/thumbnail?path=" + encodeURIComponent(t.path);
        const flux = Object.values(t.flux_db || {});
        const detail = [
            `#${t.index}`,
            `ch ${(t.channels || []).join(",")}`,
            flux.length ? `flux ${Number(flux[0]).toFixed(0)} dB` : t.kind,
            `peak ${fmtDb(t.peak_dbfs)}`,
        ].join(" · ");
        return `<div class="take">
            <img loading="lazy" src="${esc(src)}" alt="" title="Click to inspect"
                 data-zoom="${esc(t.path)}" data-name="${esc(t.name)}">
            <div class="take-meta">${esc(detail)}</div>
            <div class="take-actions">
              <button class="btn small" data-play="${esc(t.path)}">▶</button>
              <button class="btn small danger" data-del="${esc(t.uuid)}" data-name="${esc(t.name)}">✕</button>
            </div>
          </div>`;
    }).join("");
}

let modalTake = null;
let modalChannel = 0;

async function openTake(path, name) {
    modalTake = { path: path, name: name };
    modalChannel = 0;
    $("take-modal").hidden = false;
    $("tm-name").textContent = name;
    $("tm-meta").textContent = "measuring…";
    $("tm-verdict").innerHTML = "";
    $("tm-channels").innerHTML = "";
    renderTakeImage();
    try {
        const data = await api("/api/record/clip-check?path=" + encodeURIComponent(path));
        renderTakeVerdict(data);
    } catch (err) {
        $("tm-meta").textContent = "";
        $("tm-verdict").innerHTML =
            `<div class="tm-warn">${esc(err.message || String(err))}</div>`;
    }
}

function renderTakeImage() {
    if (!modalTake) return;
    $("tm-image").src = "/spectrogram?path=" + encodeURIComponent(modalTake.path)
        + "&channel=" + modalChannel;
}

function renderTakeVerdict(data) {
    $("tm-meta").textContent =
        `${data.duration_s.toFixed(2)} s · ${data.sample_rate.toLocaleString()} Hz · `
        + `${data.channels.length} channel(s)`;
    if (data.channels.length > 1) {
        $("tm-channels").innerHTML = data.channels.map((c, i) =>
            `<button data-tm-ch="${i}" class="${i === modalChannel ? "on" : ""}">ch${c.channel}</button>`
        ).join("");
    }
    const rows = data.channels.map((c) => {
        const issues = c.warnings.length
            ? c.warnings.map((w) => `<div class="tm-warn">⚠ ch${c.channel}: ${esc(w)}</div>`).join("")
            : `<div class="tm-ok">✓ ch${c.channel} looks clean</div>`;
        return issues + `<div class="tm-stats">ch${c.channel}: peak ${c.peak_dbfs} dBFS · `
            + `rms ${c.rms_dbfs} dBFS · crest ${c.crest_db} dB · DC ${c.dc_offset} · `
            + `hum ${c.hum.excess_db} dB @ ${c.hum.mains_hz || "—"} Hz</div>`;
    }).join("");
    $("tm-verdict").className = "tm-verdict";
    $("tm-verdict").innerHTML = rows;
}

function closeTake() {
    $("take-modal").hidden = true;
    modalTake = null;
}

function renderLog(entries) {
    const el = $("log");
    if (!entries || !entries.length) { el.innerHTML = '<span class="rec-empty">—</span>'; return; }
    el.innerHTML = entries.map((e) => {
        const t = String(e.recorded_at || "").slice(11, 23);
        const flux = Object.entries(e.flux_db || {})
            .map(([ch, v]) => `${Number(v).toFixed(0)}`).join("/");
        return `<div><span class="t">${esc(t)}</span> <span class="n">#${e.index}</span> ${esc(e.label)}
                 · ch ${esc((e.channels || []).join(","))}${flux ? ` · flux ${esc(flux)} dB` : ""}
                 · peak ${fmtDb(e.peak_dbfs)} dBFS</div>`;
    }).join("");
}

function fmtClock(epoch) {
    if (!epoch) return "—";
    const d = new Date(epoch * 1000);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function fmtDuration(seconds) {
    const s = Math.max(0, Math.round(seconds || 0));
    return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

function renderRuns(runs) {
    const el = $("runs");
    if (!runs || !runs.length) {
        el.innerHTML = '<div class="rec-empty">No passes recorded yet.</div>';
        $("runs-status").textContent = "";
        return;
    }
    const total = runs.reduce((n, r) => n + (r.clips || 0), 0);
    $("runs-status").textContent = `${runs.length} pass(es) · ${total} clip(s)`;
    const now = Date.now() / 1000;
    el.innerHTML = runs.map((r) => {
        const live = r.state === "recording";
        const secs = (r.ended_at || now) - r.started_at;
        const labels = (r.labels && r.labels.length ? r.labels : [r.label]);
        const detail = [r.mode, r.density, fmtDuration(secs)]
            .concat(r.location ? [r.location] : [])
            .concat((r.tags && r.tags.length) ? [r.tags.join(" ")] : [])
            .concat(r.description ? [`“${r.description}”`] : [])
            .concat(r.reason ? [r.reason] : []).join(" · ");
        return `<div class="run ${live ? "live" : ""}">
            <span class="run-dot"></span>
            <span class="run-when">${fmtClock(r.started_at)}</span>
            <span class="run-label">${esc(labels.join(", "))}<small>${esc(detail)}</small></span>
            <span class="run-clips">${r.clips || 0}<small> clips</small></span>
            ${live
                ? '<span></span>'
                : `<button class="btn small danger" data-run-del="${esc(r.id)}"
                           data-run-name="${esc(labels.join(", "))}" data-run-clips="${r.clips || 0}"
                           title="Delete this pass and all its clips">✕</button>`}
          </div>`;
    }).join("");
}

async function deleteRun(id, name, clips) {
    const n = Number(clips || 0);
    if (!confirm(
        `Delete the "${name}" pass and its ${n} clip(s)?\n\n` +
        `The WAVs, their sidecars and their library rows are removed. This cannot be undone.`
    )) return;
    try {
        const data = await post("/api/record/run/delete", { id: id });
        toast(`deleted ${data.clips} clip(s)`, "ok");
        schedulePoll(0);
    } catch (err) { fail(err, "delete run"); }
}

async function cancelRun() {
    const s = lastSnapshot;
    const n = s ? s.clips_saved : 0;
    if (!confirm(
        `Discard this run and delete its ${n} clip(s)?\n\n` +
        `Recording stops and everything it saved is removed. This cannot be undone.`
    )) return;
    busy = true;
    $("btn-cancel").disabled = true;
    try {
        const data = await post("/api/record/cancel", {});
        recording = false;
        toast(`run discarded — ${data.discarded} clip(s) deleted`, "ok");
    } catch (err) {
        fail(err, "cancel");
    } finally {
        busy = false;
        $("btn-cancel").disabled = false;
        schedulePoll(0);
    }
}

function renderCaptureAlert(data) {
    // A stream that cannot open, or a writer that cannot write, is otherwise
    // indistinguishable from a silent room: meters at -120 and a console that
    // still says "recording".
    const s = data.session || {};
    const problems = [];
    if (s.capture_error) problems.push(`Audio input: ${s.capture_error}`);
    if (s.write_error) problems.push(`Saving clips: ${s.write_error}`);
    if (s.dropouts) problems.push(
        `${s.dropouts} audio dropout(s) — clips spanning the gap were discarded.`);
    if (s.dropped_clips) problems.push(`${s.dropped_clips} clip(s) dropped.`);
    if (s.error) problems.push(s.error);
    const el = $("capture-alert");
    el.hidden = problems.length === 0;
    el.textContent = problems.join("  ");
}

// Redrawn only when something actually changed. The poll that feeds this runs
// many times a second; rewriting the panel each time replaced the Apply button
// between a mousedown and its mouseup, so the click landed on an element that
// no longer existed and nothing happened.
let calKey = "";
let calSuggestion = null;

function renderCalibration(cal) {
    const btn = $("btn-calibrate");
    const key = !cal ? "idle" : [
        cal.state, cal.remaining_s, cal.suggested_min_flux_db,
        cal.background_p50_db, cal.background_max_db, cal.strictness, cal.error,
    ].join("|");
    if (key === calKey) return;
    calKey = key;

    if (!cal || cal.state === "idle") { btn.disabled = false; return; }
    if (cal.state === "listening") {
        btn.disabled = true;
        calSuggestion = null;
        $("cal-note").textContent =
            `Listening… ${cal.remaining_s}s left — move around normally, so the `
            + `floor learns to ignore you.`;
        return;
    }
    btn.disabled = false;
    if (cal.state === "failed") {
        calSuggestion = null;
        $("cal-note").textContent = cal.error || "calibration failed";
        return;
    }
    if (cal.state === "done" && cal.suggested_min_flux_db !== undefined) {
        calSuggestion = cal.suggested_min_flux_db;
        const headroom = (calSuggestion - cal.background_max_db).toFixed(1);
        $("cal-note").innerHTML =
            `Room: median <b>${cal.background_p50_db}</b>, 95th <b>${cal.background_p95_db}</b>, `
            + `loudest <b>${cal.background_max_db}</b> dB. `
            + `Suggested floor <b>${calSuggestion} dB</b> (${cal.strictness}, `
            + `${headroom} dB above the loudest thing it heard) `
            + `<button class="btn small" id="btn-apply-cal">Apply</button>`;
    }
}

function applyCalibration() {
    if (calSuggestion === null) return;
    const suggested = calSuggestion;
    onset.min_flux_db = suggested;
    syncKnobs();
    pushOnset({ min_flux_db: suggested });
    calSuggestion = null;
    calKey = "applied";
    $("cal-note").textContent = `Floor set to ${suggested} dB.`;
    toast(`onset floor set to ${suggested} dB`, "ok");
}

function setState(text, isRec) {
    $("state-text").textContent = text;
    $("state-chip").className = "topbar-state" + (isRec ? " rec" : "");
}

function applySnapshot(data) {
    if (data.recording) {
        const s = data.session;
        lastSnapshot = s;
        recording = s.state === "running";
        setState(recording ? `● ${s.mode}` : s.state, recording);
        $("topbar-device").textContent = `${s.device} · ${s.sample_rate.toLocaleString()} Hz`;
        $("topbar-elapsed").textContent =
            `${Math.floor(s.elapsed_s / 60)}m ${String(Math.floor(s.elapsed_s % 60)).padStart(2, "0")}s`;
        $("counter").innerHTML = s.target
            ? `${s.clips_saved} <small>/ ${s.target} clips</small>`
            : `${s.clips_saved} <small>clips</small>`;
        renderRecent(s.recent);
        renderLog(s.log);
        $("mic-status").textContent = s.stats && s.stats.xruns
            ? `${s.stats.xruns} dropped buffer(s)` : "";
        $("mic-status").className = "section-status" + (s.stats && s.stats.xruns ? " warn" : "");
        if (s.state !== "running" && !busy && !autoStopped) {
            // The engine stopped itself (target or time cap reached).
            autoStopped = true;
            recording = false;
            post("/api/record/stop", {}).catch(() => {});
            toast(s.stop_reason ? `stopped — ${s.stop_reason}` : "stopped", "ok");
        }
        if (s.error) {
            $("session-status").className = "section-status err";
            $("session-status").textContent = s.error;
        }
    } else {
        recording = false;
        lastSnapshot = null;
        const meter = data.meter || {};
        setState(meter.state === "running" ? "metering" : "idle", false);
        $("topbar-elapsed").textContent = "";
        $("topbar-device").textContent = meter.device_name || "";
    }
    renderRuns(data.runs);
    renderCaptureAlert(data);
    if (document.activeElement !== $("f-project")) {
        $("f-project").value = data.project || "";
    }
    $("f-project").disabled = recording;
    document.querySelectorAll("[data-save]").forEach((b) => {
        const ok = !!(data.project || (data.session && data.clip_project_id !== "_library"));
        b.disabled = !ok;
        b.title = ok ? "" : "Pick a project to save settings — otherwise they last for this session only";
    });
    $("f-holdout").disabled = !data.project;
    $("btn-go").textContent = recording ? "Stop recording" : "Start recording";
    $("btn-go").className = "rec-go" + (recording ? " stop" : "");
    $("btn-meter").disabled = recording;
    $("btn-capture").disabled = !recording;
    $("btn-cancel").hidden = !recording;
    $("capture-note").classList.toggle("live", recording);
}

// ---- polling ---------------------------------------------------------------

let levelsTimer = null;
let levelsBusy = false;

async function pollLevels() {
    if (levelsBusy) return;
    levelsBusy = true;
    try {
        const data = await api("/api/record/levels");
        // Only the meters are redrawn here; everything else belongs to the
        // slower status poll.
        renderMeters(data.channels, { gated: true });
        if (data.calibration) renderCalibration(data.calibration);
    } catch (err) {
        // A dropped meter frame is not worth a toast; the status poll reports
        // anything that actually went wrong.
        console.debug("levels poll failed", err);
    } finally {
        levelsBusy = false;
    }
}

function startLevelsLoop() {
    clearInterval(levelsTimer);
    levelsTimer = setInterval(pollLevels, POLL_LEVELS_MS);
}

async function poll() {
    try {
        applySnapshot(await api("/api/record/status"));
    } catch (err) {
        console.error("status poll failed", err);
    } finally {
        schedulePoll();
    }
}

function schedulePoll(delay) {
    clearTimeout(pollTimer);
    const ms = delay !== undefined ? delay : (recording ? POLL_RECORDING_MS : POLL_IDLE_MS);
    pollTimer = setTimeout(poll, ms);
}

// ---- wiring ----------------------------------------------------------------

function wireSegments() {
    $("seg-density").querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => setDensity(b.dataset.density));
    });
    $("seg-mode").querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => setMode(b.dataset.mode));
    });
    $("seg-strict").querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => setStrictness(b.dataset.strict));
    });
    setStrictness(strictnessChoice);
}

function setDensity(value) {
    densityChoice = value;
    $("seg-density").querySelectorAll("button").forEach((b) =>
        b.classList.toggle("on", b.dataset.density === value));
    if (recording) pushMetadata({ density: value });
}

function setMode(value) {
    modeChoice = value;
    $("seg-mode").querySelectorAll("button").forEach((b) =>
        b.classList.toggle("on", b.dataset.mode === value));
    if (recording) {
        post("/api/record/settings", { mode: value }).catch((err) => fail(err, "mode"));
    }
}

function wireKnobs() {
    $("k-floor").addEventListener("input", (e) => {
        onset.min_flux_db = Number(e.target.value);
        $("v-floor").textContent = `${onset.min_flux_db.toFixed(0)} dB`;
        pushOnset({ min_flux_db: onset.min_flux_db });
    });
    $("k-madk").addEventListener("input", (e) => {
        onset.mad_k = Number(e.target.value);
        $("v-madk").textContent = onset.mad_k.toFixed(1);
        pushOnset({ mad_k: onset.mad_k });
    });
    $("k-cooldown").addEventListener("input", (e) => {
        onset.recording_cooldown_ms = Number(e.target.value);
        $("v-cooldown").textContent = `${onset.recording_cooldown_ms} ms`;
        pushOnset({ recording_cooldown_ms: onset.recording_cooldown_ms });
    });
    $("k-clip").addEventListener("input", (e) => {
        onset.record_seconds = Number(e.target.value);
        clampPreroll();
        syncKnobs();
        pushOnset({
            record_seconds: onset.record_seconds,
            preroll_ms: onset.preroll_ms,
            preroll_ms_range: onset.preroll_ms_range,
        });
    });
    $("k-preroll").addEventListener("input", (e) => {
        onset.preroll_ms = Number(e.target.value);
        clampPreroll();
        syncPrerollControls();
        renderClipGeometry();
        pushOnset({ preroll_ms: onset.preroll_ms });
    });
    // Dragging either end past the other would invert the range, so each end
    // pushes the other along instead of producing a value the server rejects.
    $("k-pre-lo").addEventListener("input", (e) => {
        onset.preroll_ms_range = [Number(e.target.value), onset.preroll_ms_range[1]];
        clampPreroll("lo");
        syncPrerollControls();
        renderClipGeometry();
        pushOnset({ preroll_ms_range: onset.preroll_ms_range });
    });
    $("k-pre-hi").addEventListener("input", (e) => {
        onset.preroll_ms_range = [onset.preroll_ms_range[0], Number(e.target.value)];
        clampPreroll("hi");
        syncPrerollControls();
        renderClipGeometry();
        pushOnset({ preroll_ms_range: onset.preroll_ms_range });
    });
    $("k-rand").addEventListener("change", (e) => {
        onset.randomize_preroll = e.target.checked;
        clampPreroll();
        syncKnobs();
        pushOnset({
            randomize_preroll: onset.randomize_preroll,
            preroll_ms: onset.preroll_ms,
            preroll_ms_range: onset.preroll_ms_range,
        });
    });
    $("k-trigger").addEventListener("change", (e) => {
        onset.multi_mic_trigger = e.target.value;
        pushOnset({ multi_mic_trigger: onset.multi_mic_trigger });
    });
    $("k-merge").addEventListener("change", (e) => {
        onset.cross_channel_merge_ms = Number(e.target.value);
        pushOnset({ cross_channel_merge_ms: onset.cross_channel_merge_ms });
    });
}

function wireMetadata() {
    const live = {
        "f-location": "location",
        "f-mic-position": "mic_position",
        "f-description": "description",
        "f-tags": "tags",
        "f-mics": "microphone_type",
    };
    Object.entries(live).forEach(([id, key]) => {
        $(id).addEventListener("change", () => {
            if (recording) pushMetadata({ [key]: $(id).value });
        });
    });
    $("f-label").addEventListener("change", () => {
        if (recording && $("f-label").value.trim()) {
            pushMetadata({ label: $("f-label").value.trim() });
            toast("new label — clips now land in a new recording group", "ok");
        }
        api("/api/record/devices").then((d) => renderLabelTarget(d.labels || []))
            .catch(() => {});
    });
    $("f-split").addEventListener("change", () => {
        if (recording) pushMetadata({ split_channels: $("f-split").checked });
    });
    $("f-holdout").addEventListener("change", () => {
        if (recording) pushMetadata({ holdout: $("f-holdout").checked });
    });
    $("f-project").addEventListener("change", (e) => selectProject(e.target.value));
    $("f-tags").addEventListener("input", renderTagOptions);
    $("f-device").addEventListener("change", () => { renderChannelPicker(); pushCapture(); });
    $("f-sample-rate").addEventListener("change", () => { renderClipGeometry(); pushCapture(); });
    $("f-blocksize").addEventListener("change", pushCapture);
}

function wireButtons() {
    $("btn-go").addEventListener("click", () => (recording ? stopSession() : startSession()));
    $("btn-capture").addEventListener("click", captureNow);
    $("btn-cancel").addEventListener("click", cancelRun);
    $("tm-close").addEventListener("click", closeTake);
    $("take-modal").addEventListener("click", (ev) => {
        if (ev.target === $("take-modal")) closeTake();
    });
    $("tm-play").addEventListener("click", () => {
        if (modalTake) {
            new Audio("/audio?path=" + encodeURIComponent(modalTake.path)).play()
                .catch((err) => toast("playback failed: " + err.message, "err"));
        }
    });
    $("btn-meter").addEventListener("click", () => (meterOn ? stopMeter() : startMeter()));
    $("btn-preflight").addEventListener("click", (e) => runPreflight(e.currentTarget));
    $("btn-calibrate").addEventListener("click", (e) => calibrate(e.currentTarget));
    document.querySelectorAll("[data-save]").forEach((b) =>
        b.addEventListener("click", () => saveGroup(b)));
    $("btn-done").addEventListener("click", async () => {
        if (!confirm("Stop the session and shut the recording server down?")) return;
        try { await post("/api/done", {}); } catch (e) { /* the server is going away */ }
        document.body.innerHTML =
            '<h1 style="text-align:center;margin-top:20vh;">Recording server stopped.</h1>';
    });

    document.addEventListener("click", (ev) => {
        if (ev.target.closest("#btn-apply-cal")) { applyCalibration(); return; }
        const zoom = ev.target.closest("[data-zoom]");
        if (zoom) { openTake(zoom.dataset.zoom, zoom.dataset.name || ""); return; }
        const chBtn = ev.target.closest("[data-tm-ch]");
        if (chBtn) {
            modalChannel = Number(chBtn.dataset.tmCh);
            $("tm-channels").querySelectorAll("button").forEach((b) =>
                b.classList.toggle("on", b === chBtn));
            renderTakeImage();
            return;
        }
        const play = ev.target.closest("[data-play]");
        if (play) {
            new Audio("/audio?path=" + encodeURIComponent(play.dataset.play)).play()
                .catch((err) => toast("playback failed: " + err.message, "err"));
            return;
        }
        const del = ev.target.closest("[data-del]");
        if (del) { deleteTake(del.dataset.del, del.dataset.name); return; }
        const runDel = ev.target.closest("[data-run-del]");
        if (runDel) {
            deleteRun(runDel.dataset.runDel, runDel.dataset.runName, runDel.dataset.runClips);
        }
    });
}

function wireKeyboard() {
    document.addEventListener("keydown", (ev) => {
        // Never steal a key from someone filling in the label or a tag.
        const tag = (ev.target.tagName || "").toLowerCase();
        if (tag === "input" || tag === "select" || tag === "textarea") return;
        if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
        if (ev.key === "Escape" && modalTake) { closeTake(); return; }
        if (ev.code === "Space") { ev.preventDefault(); captureNow(); return; }
        const key = ev.key.toLowerCase();
        if (key === "r") { ev.preventDefault(); recording ? stopSession() : startSession(); return; }
        if (key === "u") { ev.preventDefault(); undoLast(); return; }
        const n = Number(ev.key);
        if (n >= 1 && n <= BOOT.densities.length) setDensity(BOOT.densities[n - 1]);
    });
}

function applyForm() {
    const f = BOOT.form || {};
    if (f.label) $("f-label").value = f.label;
    if (f.location) $("f-location").value = f.location;
    if (f.mic_position) $("f-mic-position").value = f.mic_position;
    if (f.description) $("f-description").value = f.description;
    if (f.tags) $("f-tags").value = Array.isArray(f.tags) ? f.tags.join(",") : f.tags;
    if (f.microphone_type) {
        $("f-mics").value = Array.isArray(f.microphone_type)
            ? f.microphone_type.join(",") : f.microphone_type;
    }
    if (f.num) $("f-num").value = f.num;
    if (f.duration) $("f-duration").value = f.duration;
    $("f-holdout").checked = !!f.holdout;
    $("f-split").checked = !!f.split_channels;
    setDensity(densityChoice);
    setMode(modeChoice);
}

initThemeToggle();
wireSegments();
wireKnobs();
wireMetadata();
wireButtons();
wireKeyboard();
applyForm();
syncKnobs();
loadDevices();
schedulePoll(0);
startLevelsLoop();
"""
