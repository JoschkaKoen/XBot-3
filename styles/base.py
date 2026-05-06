"""
Style module contract — every file in styles/ except this one and any
underscore-prefixed helpers is auto-discovered as a style.

A style module MUST define exactly these three top-level names:
    midjourney_suffix: str
    motion_guidance:   str
    def build_image_prompt(*, funny: bool, is_zit: bool, ctx: PromptContext) -> tuple[str, str]

The module's filename (without .py) is the style's canonical name.
Files starting with "_" are treated as helpers and skipped by the autoloader.

Style modules MUST be cheap to import — no file/network I/O, no model loading,
no global side effects at top level. reload_styles() re-imports every file in
styles/ at the start of every cycle, so anything heavy at module top will pay
its cost on every cycle. Put module-level constants (prompt strings, aesthetic
blocks) only.

Branching on `is_zit` (Z-Image-Turbo / Z-Image-Base vs Midjourney/Grok) lives
inside each module's build_image_prompt — that is where prompt differences
genuinely live. If a future provider category appears, every style file will
need a new branch.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    """
    Per-cycle inputs the style needs to assemble its image prompt.

    Built fresh inside generate_image._build_image_prompt() on every call —
    never cached — so reload_settings()-driven changes to config.SOURCE_LANGUAGE
    etc. take effect on the next cycle automatically.
    """

    example_en: str          # state["example_sentence_target"]
    example_de: str          # state["example_sentence_source"] (may be empty)
    full_tweet: str          # state["full_tweet"] (may be empty)
    aspect_hint: str         # e.g. "16:9" or "wide 832×480 landscape frame"
    rules: str               # the _RULES block (provider-aware, built in node)
    source_language: str     # snapshot of config.SOURCE_LANGUAGE
    target_language: str     # snapshot of config.TARGET_LANGUAGE
