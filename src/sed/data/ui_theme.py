"""Design tokens and shell components shared by every ``sed`` web UI.

The inspect UI (:mod:`sed.data.inspect`) grew a complete visual language — a
light/dark token set, a hover-tooltip system, a sticky top bar, toasts, and one
button system. The recording console (:mod:`sed.data.record_ui`) has to look
like the same application, and a copy of 300 lines of CSS would drift the first
time a colour changed. Both templates therefore render these constants instead
of carrying their own literals.

Every constant here is the text that used to live inline in
``inspect._HTML_TEMPLATE``, at the position the corresponding placeholder now
occupies, so CSS order — and therefore the cascade — is unchanged. The one
deliberate edit is in :data:`BUTTON_SYSTEM_CSS`, whose selector lists gained a
UI-neutral ``.btn`` family; inspect has no ``.btn`` elements, so that addition
is inert there.

Components that are dark in BOTH themes (the top bar, tooltips, popovers,
toasts) keep their colour literals rather than tokens, exactly as they did
before the move.
"""

from __future__ import annotations

# Resolve the stored/OS theme before first paint. Goes in <head>, ahead of
# the stylesheet, so a dark session never flashes white.
THEME_PREPAINT_JS = r"""<script>
/* Resolve the theme before the first paint, so a dark session never flashes
   white. Stored choice wins; with none, follow the OS setting. */
(function () {
    try {
        var saved = localStorage.getItem("sed-theme");
        var dark = saved
            ? saved === "dark"
            : window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (dark) document.documentElement.setAttribute("data-theme", "dark");
    } catch (e) { /* private mode / no storage: stay light */ }
})();
</script>"""

# The Inter webfont link. The only external resource any page loads.
FONT_LINK = r"""<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap">"""

# The complete light and dark token sets, plus the theme-toggle icon swap.
# Must be the first rule block in a page's <style>.
THEME_TOKENS_CSS = r"""/* ========================================================================
   Theme tokens. Light values are the literals this sheet always used, so the
   light UI is unchanged; [data-theme="dark"] on <html> swaps in the dark set.
   Components that are dark in BOTH themes (top bar, tooltips, spectrogram
   backdrops, the training log) keep their literals and are not tokenised.
   ===================================================================== */
:root {
  color-scheme: light;
  /* page & surfaces */
  --bg-page:    #fafafa;
  --surface:    #ffffff;
  --surface-b1: #fbfcfe;
  --surface-b2: #fbfdff;
  --surface-b3: #f8fbff;
  --surface-2:  #f8fafc;
  --surface-b4: #f5f8ff;
  --surface-3:  #f1f5f9;
  --surface-3b: #eef2f7;
  --surface-3c: #eceff4;
  --surface-4:  #e2e8f0;
  --surface-v1: #faf8ff;
  --fill-muted: #94a3b8;
  /* borders */
  --bd-soft:     #f1f5f9;
  --bd-soft-2:   #eef2f7;
  --bd:          #e2e8f0;
  --bd-b1:       #dbe4ef;
  --bd-b2:       #dbe3ee;
  --bd-2:        #cbd5e1;
  --bd-strong:   #94a3b8;
  --bd-strong-2: #64748b;
  /* Non-event class identity: intentionally achromatic, never blue. */
  --negative-class-bg:     #eef0f2;
  --negative-class-border: #7d858e;
  /* text */
  --tx-strong:   #0f172a;
  --tx-strong-2: #1e293b;
  --tx:          #334155;
  --tx-2:        #475569;
  --tx-muted:    #64748b;
  --tx-faint:    #94a3b8;
  --tx-vfaint:   #cbd5e1;
  /* inverted chips / primary fills (dark chip on a light page, and vice versa) */
  --inv-bg-strong: #0f172a;
  --inv-bg:        #1e293b;
  --inv-bd-strong: #0f172a;
  --inv-bd:        #1e293b;
  --inv-fg:        #ffffff;
  /* red family */
  --red-tint:    #fef2f2;
  --red-tint-2:  #fee2e2;
  --red-tint-3:  #fff5f5;
  --red-tint-4:  #fff1f2;
  --red-tint-5:  #ffe4e6;
  --red-tint-6:  #fff7f7;
  --red-soft:    #fca5a5;
  --red-soft-2:  #fecaca;
  --red-mid:     #ef4444;
  --red-mid-2:   #f87171;
  --red-solid:   #991b1b;
  --red-solid-2: #b91c1c;
  --red-solid-3: #dc2626;
  --red-deep:    #9f1239;
  /* amber family */
  --amber-tint:    #fffbeb;
  --amber-tint-2:  #fffdf6;
  --amber-soft:    #fef3c7;
  --amber-soft-2:  #f0d9a8;
  --amber-mid:     #fde68a;
  --amber-mid-2:   #fcd34d;
  --amber-solid:   #f59e0b;
  --amber-solid-2: #b45309;
  --amber-solid-3: #d97706;
  --amber-solid-4: #facc15;
  --amber-deep:    #78350f;
  --amber-deep-2:  #422006;
  --amber-deep-3:  #92400e;
  /* Ink for text placed ON a solid amber fill. The amber-deep-* tokens are
     "amber text on a tinted background" and therefore flip light in the dark
     theme — using one of those on --amber-solid gives light-on-light. This
     token stays dark in both themes because --amber-solid is bright in both. */
  --amber-ink:     #3a1d05;
  /* green family */
  --green-tint:    #ecfdf5;
  --green-tint-2:  #f0fdf4;
  --green-soft:    #bbf7d0;
  --green-soft-2:  #d1fae5;
  --green-soft-3:  #dcfce7;
  --green-soft-4:  #a7f3d0;
  --green-mid:     #86efac;
  --green-mid-2:   #6ee7b7;
  --green-solid:   #10b981;
  --green-solid-2: #22c55e;
  --green-solid-3: #34d399;
  --green-solid-4: #16a34a;
  --green-deep:    #065f46;
  --green-deep-2:  #166534;
  --green-deep-3:  #14532d;
  --green-deep-4:  #047857;
  /* teal family */
  --teal-tint:   #ecfeff;
  --teal-soft:   #a5f3fc;
  --teal-soft-2: #ccfbf1;
  --teal-mid:    #38bdf8;
  --teal-mid-2:  #5eead4;
  --teal-solid:  #22d3ee;
  --teal-deep:   #0e7490;
  --teal-deep-2: #164e63;
  --teal-deep-3: #0f766e;
  --teal-deep-4: #134e4a;
  /* blue family */
  --blue-tint:    #eef2ff;
  --blue-tint-2:  #eff6ff;
  --blue-tint-3:  #e0e7ff;
  --blue-soft:    #c7d2fe;
  --blue-soft-2:  #a5b4fc;
  --blue-soft-3:  #bfdbfe;
  --blue-soft-4:  #dbeafe;
  --blue-mid:     #6366f1;
  --blue-mid-2:   #818cf8;
  --blue-mid-3:   #4f46e5;
  --blue-solid:   #3730a3;
  --blue-solid-2: #2563eb;
  --blue-solid-3: #1d4ed8;
  --blue-solid-4: #1e40af;
  --blue-deep:    #1e3a8a;
  --blue-deep-2:  #312e81;
  /* violet family */
  --violet-tint:    #faf5ff;
  --violet-tint-2:  #f5f3ff;
  --violet-tint-3:  #ede9fe;
  --violet-soft:    #ddd6fe;
  --violet-soft-2:  #c4b5fd;
  --violet-mid:     #a78bfa;
  --violet-mid-2:   #7c3aed;
  --violet-mid-3:   #8b5cf6;
  --violet-solid:   #6d28d9;
  --violet-solid-2: #5b21b6;
  /* pink family */
  --pink-solid: #c026d3;
}

:root[data-theme="dark"] {
  color-scheme: dark;
  /* page & surfaces */
  --bg-page:    #0d1420;
  --surface:    #161f2f;
  --surface-b1: #182131;
  --surface-b2: #182233;
  --surface-b3: #1a2435;
  --surface-2:  #1c2536;
  --surface-b4: #1d2739;
  --surface-3:  #222c3e;
  --surface-3b: #263144;
  --surface-3c: #273248;
  --surface-4:  #2d3a4e;
  --surface-v1: #1e1b33;
  --fill-muted: #5d6b80;
  /* borders */
  --bd-soft:     #232d3e;
  --bd-soft-2:   #263143;
  --bd:          #2c384b;
  --bd-b1:       #303d52;
  --bd-b2:       #313e53;
  --bd-2:        #3a4860;
  --bd-strong:   #4d5c74;
  --bd-strong-2: #5d6c84;
  /* Non-event class identity: intentionally achromatic, never blue. */
  --negative-class-bg:     #383d43;
  --negative-class-border: #858c94;
  /* text */
  --tx-strong:   #e8eef7;
  --tx-strong-2: #dbe4f0;
  --tx:          #c6d1e0;
  --tx-2:        #aab8ca;
  --tx-muted:    #8f9cb0;
  --tx-faint:    #79869a;
  --tx-vfaint:   #6b7a91;
  /* inverted chips / primary fills (dark chip on a light page, and vice versa) */
  --inv-bg-strong: #f1f5f9;
  --inv-bg:        #e2e8f0;
  --inv-bd-strong: #f1f5f9;
  --inv-bd:        #e2e8f0;
  --inv-fg:        #0f172a;
  /* red family */
  --red-tint:    #311414;
  --red-tint-2:  #311414;
  --red-tint-3:  #311414;
  --red-tint-4:  #311416;
  --red-tint-5:  #311416;
  --red-tint-6:  #311414;
  --red-soft:    #632c2c;
  --red-soft-2:  #632c2c;
  --red-mid:     #d86464;
  --red-mid-2:   #d86464;
  --red-solid:   #e25a5a;
  --red-solid-2: #e45858;
  --red-solid-3: #e45858;
  --red-deep:    #ea86a1;
  /* amber family */
  --amber-tint:    #312b14;
  --amber-tint-2:  #312a14;
  --amber-soft:    #63582c;
  --amber-soft-2:  #63512c;
  --amber-mid:     #d8c164;
  --amber-mid-2:   #d8bd64;
  --amber-solid:   #e0a43e;
  --amber-solid-2: #e0843e;
  --amber-solid-3: #e0953e;
  --amber-solid-4: #e0bf3e;
  --amber-deep:    #eaaa86;
  --amber-deep-2:  #eab186;
  --amber-deep-3:  #eaab86;
  --amber-ink:     #2b1a05;
  /* green family */
  --green-tint:    #143123;
  --green-tint-2:  #14311d;
  --green-soft:    #2c633f;
  --green-soft-2:  #2c6347;
  --green-soft-3:  #2c633f;
  --green-soft-4:  #2c634a;
  --green-mid:     #64d88e;
  --green-mid-2:   #64d8aa;
  --green-solid:   #3ee0aa;
  --green-solid-2: #40de7a;
  --green-solid-3: #47d7a2;
  --green-solid-4: #3ee07a;
  --green-deep:    #86eace;
  --green-deep-2:  #8ae5ad;
  --green-deep-3:  #8ce3af;
  --green-deep-4:  #86eacd;
  /* teal family */
  --teal-tint:   #142f31;
  --teal-soft:   #2c5d63;
  --teal-soft-2: #2c6357;
  --teal-mid:    #64b5d8;
  --teal-mid-2:  #64d8c6;
  --teal-solid:  #3ecae0;
  --teal-deep:   #86d4ea;
  --teal-deep-2: #8acce5;
  --teal-deep-3: #86eae2;
  --teal-deep-4: #8ce3dd;
  /* blue family */
  --blue-tint:    #141b31;
  --blue-tint-2:  #142131;
  --blue-tint-3:  #141a31;
  --blue-soft:    #2c3763;
  --blue-soft-2:  #2c3663;
  --blue-soft-3:  #2c4463;
  --blue-soft-4:  #2c4463;
  --blue-mid:     #7c7ede;
  --blue-mid-2:   #646fd8;
  --blue-mid-3:   #827cde;
  --blue-solid:   #8681da;
  --blue-solid-2: #5884e4;
  --blue-solid-3: #6184e6;
  --blue-solid-4: #6281e4;
  --blue-deep:    #8aa2e5;
  --blue-deep-2:  #9896d9;
  /* violet family */
  --violet-tint:    #221431;
  --violet-tint-2:  #191431;
  --violet-tint-3:  #191431;
  --violet-soft:    #362c63;
  --violet-soft-2:  #382c63;
  --violet-mid:     #8164d8;
  --violet-mid-2:   #9b74dc;
  --violet-mid-3:   #9474dc;
  --violet-solid:   #a174e6;
  --violet-solid-2: #a175e6;
  /* pink family */
  --pink-solid: #d152e0;
}

/* The bar is dark in both themes, so on a dark page it needs an edge to read
   as a separate surface rather than blending into the content behind it. */
:root[data-theme="dark"] .toolbar {
  box-shadow: 0 1px 0 rgba(148,163,184,0.16), 0 4px 12px rgba(0,0,0,0.5);
}

/* Spectrograms and plots are rendered as images on a dark backdrop already;
   soften them slightly so they do not glare against the dark page. */
:root[data-theme="dark"] .clip img.spec { filter: brightness(0.92); }

/* ---- top-bar theme switch ----
   The icon shows what the click will do, matching the tooltip: a moon while
   the page is light, a sun while it is dark. */
.theme-toggle .theme-icon-sun { display: none; }
:root[data-theme="dark"] .theme-toggle .theme-icon-moon { display: none; }
:root[data-theme="dark"] .theme-toggle .theme-icon-sun { display: block; }"""

# body / headings / .summary / .card.
BASE_TYPOGRAPHY_CSS = r"""body { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: var(--bg-page); color: var(--tx-strong-2); max-width: 1300px; margin: 0 auto;
       padding: 0 1.5rem 2rem 1.5rem; -webkit-font-smoothing: antialiased; }
h1 { font-size: 2.25rem; font-weight: 700; border-bottom: 2px solid var(--bd); padding-bottom: 0.5rem; margin-top: 2rem; color: var(--tx-strong); }
h2 { margin-top: 1.8rem; font-size: 1.5rem; font-weight: 600; color: var(--tx-strong-2); }
h3.section { background: var(--inv-bg); color: var(--inv-fg); padding: 0.6rem 1rem; border-radius: 8px; font-weight: 500; font-size: 1.1rem; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
           gap: 0.75rem; margin-bottom: 1.5rem; }
.card { background: var(--surface); padding: 1rem 1.25rem; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.02);
        border: 1px solid var(--bd-soft); }
.card[data-summary-class][data-class-token] {
  background: linear-gradient(135deg, var(--class-bg), var(--surface) 58%);
  border-color: var(--class-border); box-shadow: inset 4px 0 0 hsl(var(--class-hue), var(--class-sat), var(--class-bd-lum)), 0 1px 3px rgba(0,0,0,0.05);
}
.summary-breakdown { margin-top: 0.25rem; color: var(--tx-muted); font-size: 0.76rem; font-weight: 600; }"""

# The `.tip` underline and the single `.hover-tip` element it drives.
TOOLTIP_CSS = r""".tip {
  border-bottom: 1px dotted var(--bd-strong);
  text-decoration: none;
  text-underline-offset: 2px;
}
.hover-tip {
  position: fixed; z-index: 2000; max-width: min(480px, calc(100vw - 24px));
  background: rgba(15,23,42,0.98); color: #fff; padding: 10px 12px;
  border-radius: 8px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
  font-size: 0.75rem; line-height: 1.5; white-space: pre-line; pointer-events: none;
  opacity: 0; transform: translateY(3px); transition: opacity 120ms ease, transform 120ms ease;
  border: 1px solid #334155;
}
.hover-tip.show { opacity: 1; transform: translateY(0); }
/* Metric tooltips lead with the values they depend on, then the formula. Bold
   carries the "this number is threshold-dependent, and here is the threshold"
   line; italic carries the formula, set apart so it reads as an equation. */
.hover-tip b { color: #fbbf24; font-weight: 700; }
.hover-tip i { display: inline-block; margin: 2px 0; padding: 1px 5px; border-radius: 4px;
  background: rgba(148,163,184,0.18); color: #e2e8f0; font-style: normal;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.72rem; }"""

# audio player, `.search-filter` inputs, `.notice` banners.
CONTROL_CSS = r"""audio { width: 100%; height: 36px; margin-top: 0.5rem; border-radius: 8px; }
.search-filter { display: inline-flex; align-items: center; gap: 0.5rem; margin: 0 0.5rem 0.5rem 0; }
.search-filter input {
  width: 18rem; max-width: 72vw; padding: 0.45rem 0.75rem; border: 1px solid var(--bd-2);
  border-radius: 8px; font: inherit; font-size: 0.875rem; background: var(--surface); outline: none; transition: border-color 0.2s;
}
.search-filter input:focus { border-color: var(--blue-mid); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1); }
.search-filter button {
  background: var(--surface-3); border: 1px solid var(--bd-2); border-radius: 8px;
  padding: 0.45rem 0.9rem; cursor: pointer; font-size: 0.875rem; font-weight: 500; transition: background 0.2s;
}
.search-filter button:hover { background: var(--surface-4); }
.notice { background: var(--amber-soft); padding: 0.75rem 1.25rem; border-left: 4px solid var(--amber-solid);
          margin: 1.5rem 0; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: var(--amber-deep); }
.notice.danger { background: var(--red-tint-2); border-left-color: var(--red-mid); color: var(--red-solid); }
.mask-good { color: var(--green-solid); font-weight: 600; }
.mask-warn { color: var(--amber-solid); font-weight: 600; }
.mask-bad  { color: var(--red-mid); font-weight: 600; }"""

# The sticky dark top bar and its bar-local tokens.
TOOLBAR_CSS = r"""/* ============ Activity / resource top bar ============ */
.toolbar {
    position: sticky; top: 0; z-index: 100;
    background: #0f172a; color: #fff;
    padding: 0.55rem 1.25rem; margin: 0 -1.5rem 1.5rem -1.5rem;
    display: flex; gap: 1rem; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    /* local bar tokens (scoped so they never leak into the light UI below) */
    --muted: #94a3b8;
    --chip-bg: rgba(148,163,184,0.12);
    --chip-bd: rgba(148,163,184,0.22);
    --chip-bg-hover: rgba(148,163,184,0.22);
    --lvl-free: #22c55e; --lvl-busy: #f59e0b; --lvl-heavy: #ef4444;
}"""

# Solid action buttons that live inside the top bar.
TOOLBAR_BUTTON_CSS = r"""/* shared bar buttons */
.toolbar button.done, .toolbar button.del, .toolbar button.primary-action {
    border: none; padding: 0.5rem 1.15rem; border-radius: 8px; cursor: pointer;
    font-weight: 600; font-size: 0.9rem; color: #fff; transition: background 0.2s;
}
.toolbar button.done { background: #10b981; }
.toolbar button.done:hover { background: #059669; }
.toolbar button.done:disabled { background: #475569; cursor: default; }
.toolbar button.del { background: #ef4444; }
.toolbar button.del:hover { background: #dc2626; }
.toolbar button.del:disabled { background: #475569; cursor: default; }
.toolbar button.primary-action { background: #7c3aed; }
.toolbar button.primary-action:hover { background: #6d28d9; }
.toolbar button.primary-action:disabled { background: #475569; cursor: wait; }
body.ui-busy, body.ui-busy * { cursor: wait !important; }"""

# Click-anchored dark detail popovers hanging off the top bar.
POPOVER_CSS = r"""/* ---- click-anchored detail popovers ---- */
.bar-popover {
    position: absolute; top: calc(100% + 6px); right: 1.25rem; z-index: 200;
    min-width: 240px; max-width: 360px; padding: 0.85rem 1rem;
    background: #1e293b; color: #e2e8f0; border: 1px solid rgba(148,163,184,0.25);
    border-radius: 10px; box-shadow: 0 12px 28px rgba(0,0,0,0.4); font-size: 0.8rem;
}
.bar-popover[hidden] { display: none; }
.bar-popover h4 { margin: 0 0 0.5rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 700; }
.popover-row { display: flex; justify-content: space-between; gap: 1rem; padding: 0.2rem 0; }
.popover-row .k { color: var(--muted); }
.popover-row .v { font-weight: 700; font-variant-numeric: tabular-nums; }
.popover-legend { margin-top: 0.6rem; padding-top: 0.55rem; border-top: 1px solid rgba(148,163,184,0.18); color: var(--muted); font-size: 0.72rem; line-height: 1.5; }"""

# The transient toast stack, plus the reduced-motion opt-out.
TOAST_CSS = r"""/* ---- transient toasts ---- */
.toast-stack { position: fixed; top: 64px; right: 18px; z-index: 1200; display: flex; flex-direction: column; gap: 0.5rem; pointer-events: none; }
.toast {
    pointer-events: auto; min-width: 220px; max-width: 340px; display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0.85rem; background: #1e293b; color: #e2e8f0; border: 1px solid rgba(148,163,184,0.25);
    border-left: 4px solid #38bdf8; border-radius: 10px; box-shadow: 0 10px 24px rgba(0,0,0,0.4);
    font-size: 0.82rem; font-weight: 600; cursor: pointer; animation: toast-in 0.22s ease;
}
.toast.ok { border-left-color: #22c55e; }
.toast.err { border-left-color: #ef4444; }
.toast.warn { border-left-color: #f59e0b; }
.toast.hide { animation: toast-out 0.25s ease forwards; }
@keyframes toast-in { from { opacity: 0; transform: translateX(12px); } to { opacity: 1; transform: none; } }
@keyframes toast-out { to { opacity: 0; transform: translateX(12px); } }

@media (prefers-reduced-motion: reduce) {
    .activity-chip.indeterminate .chip-ring { animation: none; }
    .load-fill, .chip-ring .ring-val { transition: none; }
    .toast { animation: none; }
}"""

# Inline keyboard-key chips.
KBD_CSS = r""".kbd { background: #334155; padding: 2px 6px; border-radius: 4px; font-family: monospace;
       font-size: 0.8rem; color: #f1f5f9; border: 1px solid #475569; }"""

# Small uppercase status pills.
PILL_CSS = r""".pill { display: inline-flex; align-items: center; justify-content: center;
         padding: 1px 7px; border-radius: 10px; font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
.pill.good { background: var(--green-soft-2); color: var(--green-deep); }
.pill.bad { background: var(--red-tint-2); color: var(--red-solid); }
.pill.neg { background: var(--surface-3); color: var(--tx-2); }"""

# One button system: default / .primary / .danger / .accent / .warn, in two
# sizes chosen by placement. `.btn` is the UI-neutral entry point.
BUTTON_SYSTEM_CSS = r"""/* ==== One button system ==================================================
   Every button in the dataset-building UI (Library tab and the review tabs'
   section headers) is one of: default / .primary / .danger / .accent, in one
   of two sizes. Size is decided by *where* the button is, not by which family
   it came from: anything in a section header or an inline toolbar is small,
   anything in a main action row is regular. Before this, "Rescan from disk"
   and "corpus headers" sat side by side at different sizes, weights and
   colours because each came from its own one-off rule. */
.section-action,
.btn,
button.section-refresh,
.recipe-actions button,
.assign-bar button,
.tc-add-row button,
.preset-bar button,
.bucket-picker button,
.library-actions button,
.preview-cache-head button,
.library-row-delete,
.train-split-config button,
.compose-run {
    display: inline-flex; align-items: center; justify-content: center; gap: 0.35rem;
    min-height: 2rem; box-sizing: border-box;
    border: 1px solid var(--bd-2); background: var(--surface); color: var(--tx);
    border-radius: 6px; cursor: pointer; white-space: nowrap;
    font-family: inherit; font-size: 0.78rem; line-height: 1; font-weight: 700;
    padding: 0.42rem 0.7rem;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.section-action:hover,
.btn:hover,
button.section-refresh:hover,
.recipe-actions button:hover,
.assign-bar button:hover,
.tc-add-row button:hover,
.preset-bar button:hover,
.bucket-picker button:hover,
.library-actions button:hover,
.preview-cache-head button:hover,
.train-split-config button:hover,
.compose-run:hover { background: var(--blue-tint); border-color: var(--blue-soft-2); color: var(--blue-solid); }

.section-action:disabled,
.btn:disabled,
button.section-refresh:disabled,
.recipe-actions button:disabled,
.assign-bar button:disabled,
.tc-add-row button:disabled,
.preset-bar button:disabled,
.bucket-picker button:disabled,
.library-actions button:disabled,
.preview-cache-head button:disabled,
.library-row-delete:disabled,
.train-split-config button:disabled,
.compose-run:disabled { opacity: 0.55; cursor: default; }
/* Busy — the action is running, as opposed to unavailable. */
.section-action.spinning, .btn.spinning, button.section-refresh.spinning, .compose-run:disabled { cursor: wait; }

/* Small: section headers and inline toolbars, where the button annotates a
   panel rather than being the panel's action. Placement decides the size, so
   two buttons sitting side by side in a header always match — regardless of
   which family each came from. */
.section-action,
.btn.small,
.section-head button.section-refresh,
.preset-bar button,
.preview-cache-head button,
.compose-bar .rescan-chip,
.train-split-config button,
.library-row-delete {
    min-height: 1.7rem; font-size: 0.74rem; padding: 0.26rem 0.55rem;
}
.section-head { gap: 0.4rem; }

/* Tones. */
.btn.primary,
.recipe-actions button.primary,
.assign-bar button.primary,
.library-actions button.primary,
.compose-run { background: var(--inv-bg); border-color: var(--inv-bd); color: var(--inv-fg); }
.btn.primary:hover,
.recipe-actions button.primary:hover,
.assign-bar button.primary:hover,
.library-actions button.primary:hover,
.compose-run:hover { background: var(--inv-bg-strong); border-color: var(--inv-bd-strong); color: var(--inv-fg); }

.section-action.danger,
.btn.danger,
.recipe-actions button.danger,
.assign-bar button.danger,
.library-actions button.danger,
.library-row-delete { background: var(--red-tint-4); border-color: var(--red-soft-2); color: var(--red-solid); }
.section-action.danger:hover,
.btn.danger:hover,
.recipe-actions button.danger:hover,
.assign-bar button.danger:hover,
.library-actions button.danger:hover,
.library-row-delete:hover { background: var(--red-tint-5); border-color: var(--red-soft); color: var(--red-solid); }

.btn.accent,
.tc-add-row button,
.bucket-picker button,
.compose-bar .rescan-chip.all { background: var(--blue-tint); border-color: var(--blue-soft); color: var(--blue-solid); }
.btn.accent:hover,
.tc-add-row button:hover,
.bucket-picker button:hover,
.compose-bar .rescan-chip.all:hover { background: var(--blue-tint-3); border-color: var(--blue-soft-2); }

.btn.warn,
.library-actions button.warn { background: var(--amber-soft); border-color: var(--amber-solid); color: var(--amber-deep-3); }
/* --amber-mid is an accent/border tone (it flips light in the dark theme), so
   the hover fill uses the tinted-surface tone that --amber-deep-3 is legible on
   in both themes. */
.btn.warn:hover,
.library-actions button.warn:hover { background: var(--amber-soft-2); border-color: var(--amber-solid-3); color: var(--amber-deep-3); }

/* The ↻ glyph is shared by every rescan-style button, and spins while busy. */
.refresh-icon { display: inline-block; font-size: 0.92rem; line-height: 1; transform-origin: 50% 50%; }
button.section-refresh.spinning, .section-action.spinning { color: var(--blue-solid); border-color: var(--blue-soft-2); background: var(--blue-tint); }
button.section-refresh.spinning .refresh-icon, .section-action.spinning .refresh-icon, .btn.spinning .refresh-icon { animation: sed-spin 0.8s linear infinite; }
@keyframes sed-spin { to { transform: rotate(360deg); } }

/* Icon-only buttons: one square, one size, everywhere. */
.dd-x, .lib-row-x {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; padding: 0; line-height: 1;
    border: 1px solid var(--bd); background: var(--surface); color: var(--tx-faint);
    border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 700;
}
.dd-x:hover, .section-status { color: var(--tx-muted); font-size: 0.78rem; font-weight: 600; margin-left: 0.35rem; }
.section-status.err { color: var(--red-mid); }
.section-status.ok { color: var(--green-solid); }
.section-status.warn { color: var(--amber-solid-2); }"""

# Delegated [data-tip] handling for the single `.hover-tip` element.
TOOLTIP_JS = r"""// Single smooth hover tooltip
const hoverTip = document.getElementById("hover-tip");
let tipTimer = null;
function placeTip(ev) {
    if (!hoverTip) return;
    const pad = 12;
    hoverTip.style.left = "0px";
    hoverTip.style.top = "0px";
    const rect = hoverTip.getBoundingClientRect();
    let x = ev.clientX + 14;
    let y = ev.clientY + 14;
    if (x + rect.width + pad > window.innerWidth) x = ev.clientX - rect.width - 14;
    if (y + rect.height + pad > window.innerHeight) y = ev.clientY - rect.height - 14;
    hoverTip.style.left = Math.max(pad, x) + "px";
    hoverTip.style.top = Math.max(pad, y) + "px";
}
// Tooltips carry real explanations now — which threshold a metric depends on,
// then the formula, then the prose — and a wall of one-line text hid exactly the
// part that mattered. So a tiny markup: **bold**, *italic*, and newlines.
//
// The text is escaped FIRST and the markers applied to the escaped string, so a
// tip built from user data (a model id, a run description) cannot inject markup
// no matter what it contains. The worst a literal "**" in a description can do
// is render bold.
function tipMarkup(raw) {
    const esc = String(raw)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return esc
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
        .replace(/\*([^*\n]+)\*/g, "<i>$1</i>")
        .replace(/\n/g, "<br>");
}
document.addEventListener("mouseover", ev => {
    const el = ev.target.closest("[data-tip]");
    if (!el || !el.dataset.tip) return;
    clearTimeout(tipTimer);
    hoverTip.innerHTML = tipMarkup(el.dataset.tip);
    placeTip(ev);
    tipTimer = setTimeout(() => hoverTip.classList.add("show"), 120);
});
document.addEventListener("mousemove", ev => {
    const el = ev.target.closest("[data-tip]");
    if (el) placeTip(ev);
});
document.addEventListener("mouseout", ev => {
    const el = ev.target.closest("[data-tip]");
    if (el) {
        clearTimeout(tipTimer);
        hoverTip.classList.remove("show");
    }
});"""

# `toast(msg, cls)` — appends to `#toast-stack`.
TOAST_JS = r"""// --- toast stack ---
function toast(msg, cls) {
    const stack = document.getElementById("toast-stack");
    if (!stack) return;              // non-interactive page: no bar
    const el = document.createElement("div");
    el.className = "toast" + (cls ? " " + cls : "");
    el.textContent = msg;
    const remove = () => { el.classList.add("hide"); setTimeout(() => el.remove(), 260); };
    el.addEventListener("click", remove);
    stack.appendChild(el);
    setTimeout(remove, cls === "err" ? 8000 : 5000);
}"""

# Markup for the two singleton elements the shell JS expects on every page.
TOOLTIP_ELEMENT = '<div class="hover-tip" id="hover-tip"></div>'
TOAST_ELEMENT = '<div class="toast-stack" id="toast-stack" aria-live="polite"></div>'


def shell_css() -> str:
    """Every shared stylesheet block, in cascade order.

    Convenience for a page that wants the whole shell; ``inspect`` injects the
    blocks individually because its own rules are interleaved between them.
    """
    return "\n".join([
        THEME_TOKENS_CSS,
        BASE_TYPOGRAPHY_CSS,
        TOOLTIP_CSS,
        CONTROL_CSS,
        TOOLBAR_CSS,
        TOOLBAR_BUTTON_CSS,
        POPOVER_CSS,
        TOAST_CSS,
        KBD_CSS,
        PILL_CSS,
        BUTTON_SYSTEM_CSS,
    ])


def shell_js() -> str:
    """Tooltip + toast behaviour. Theme toggling is per-page: inspect also has
    charts to repaint, so it keeps its own `applyTheme`."""
    return "\n".join([TOOLTIP_JS, TOAST_JS])
