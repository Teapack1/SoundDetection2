"""The shared UI shell.

The inspect UI and the recording console render the same tokens and shell
components. These tests pin that they are actually shared, and that pulling
them out of inspect's template did not change what inspect renders.
"""

from __future__ import annotations

import re

import pytest

SHELL_CONSTANTS = (
    "THEME_PREPAINT_JS", "FONT_LINK", "THEME_TOKENS_CSS", "BASE_TYPOGRAPHY_CSS",
    "TOOLTIP_CSS", "CONTROL_CSS", "TOOLBAR_CSS", "TOOLBAR_BUTTON_CSS",
    "POPOVER_CSS", "TOAST_CSS", "KBD_CSS", "PILL_CSS", "BUTTON_SYSTEM_CSS",
    "TOOLTIP_JS", "TOAST_JS",
)


def test_every_shell_constant_is_present_and_non_empty():
    from sed.data import ui_theme

    for name in SHELL_CONSTANTS:
        value = getattr(ui_theme, name)
        assert isinstance(value, str) and value.strip(), name


def test_tokens_define_both_themes():
    from sed.data.ui_theme import THEME_TOKENS_CSS

    assert ":root {" in THEME_TOKENS_CSS
    assert ':root[data-theme="dark"] {' in THEME_TOKENS_CSS
    # A colour defined only inside the dark block would leave the light page
    # with an undefined variable, so both sets must carry the same token names.
    def tokens(block: str) -> set[str]:
        return set(re.findall(r"^\s+(--[a-z0-9-]+):", block, flags=re.M))

    light = THEME_TOKENS_CSS.split(':root[data-theme="dark"]')[0]
    dark = THEME_TOKENS_CSS.split(':root[data-theme="dark"]', 1)[1]
    assert tokens(light) == tokens(dark)


def test_button_system_exposes_a_ui_neutral_btn_family():
    from sed.data.ui_theme import BUTTON_SYSTEM_CSS

    for selector in (".btn,", ".btn.primary,", ".btn.danger,", ".btn.small,"):
        assert selector in BUTTON_SYSTEM_CSS, selector


def test_shell_helpers_concatenate_in_cascade_order():
    from sed.data import ui_theme

    css = ui_theme.shell_css()
    assert css.index(ui_theme.THEME_TOKENS_CSS) < css.index(ui_theme.BASE_TYPOGRAPHY_CSS)
    assert ui_theme.BUTTON_SYSTEM_CSS in css
    js = ui_theme.shell_js()
    assert "hover-tip" in js
    assert "function toast(" in js


def test_inspect_template_fills_exactly_the_shell_placeholders():
    from sed.data.inspect import _HTML_TEMPLATE, ui_theme_context

    context = ui_theme_context()
    placeholders = set(re.findall(r"\{\{ ([a-z_]+)\|safe \}\}", _HTML_TEMPLATE))
    assert set(context) <= placeholders
    assert len(context) == len(SHELL_CONSTANTS)


def test_inspect_still_renders_the_shell_it_used_to_carry_inline():
    jinja2 = pytest.importorskip("jinja2")
    from sed.data.inspect import _HTML_TEMPLATE, ui_theme_context

    empty = {
        key: {} for key in (
            "training_selection_summary", "panels", "aug_panels", "frozen_panels",
            "mined_panels", "class_summary", "class_images", "frozen_context",
            "panel_subgroups", "panel_origin_counts", "dataset_origin_counts",
        )
    }
    html = jinja2.Template(
        _HTML_TEMPLATE, undefined=jinja2.ChainableUndefined
    ).render(
        **ui_theme_context(), project_id="x", interactive=True,
        library_items=[], model_classes=[], **empty,
    )
    for needle in (
        "--bg-page:    #fafafa",      # light tokens
        '--bg-page:    #0d1420',      # dark tokens
        ".toolbar {",
        ".hover-tip {",
        ".tabs button.active",
        "function toast(msg, cls)",
        'localStorage.getItem("sed-theme")',
        "family=Inter",
    ):
        assert needle in html, needle


def test_record_page_renders_the_same_shell():
    from sed.config import Config
    from sed.data.record_ui import render_record_html

    html = render_record_html(Config(), "demo", {"label": "cheers_quiet"})
    for needle in (
        "--bg-page:    #fafafa",
        ".toolbar {",
        ".hover-tip {",
        'id="hover-tip"',
        'id="toast-stack"',
        "function toast(msg, cls)",
        'localStorage.getItem("sed-theme")',
        "Recording console",
    ):
        assert needle in html, needle
    # The form prefill has to reach the page, or --serve's flags do nothing.
    assert "cheers_quiet" in html
