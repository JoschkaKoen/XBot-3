"""
Auto-discovering style registry.

Every Python file in this package is a style; the file's stem (e.g. "disney"
in styles/disney.py) is the style's canonical name. Files starting with "_"
and the file "base" are skipped.

reload_styles() rescans the package and refreshes the registry. main.py calls
it at the start of every cycle (right after config.reload_settings()) so new
files dropped in styles/ and edits to existing files are picked up between
cycles.

DO NOT import style modules directly (e.g. `from styles.disney import
build_image_prompt`). reload_styles() uses importlib.reload(); a direct import
captured at import time will silently become a stale reference. Always go
through get_style(name) to fetch attributes per-cycle.
"""

import importlib
import pkgutil
import random
import sys

from .base import PromptContext  # re-exported for consumers

# Special token accepted in IMAGE_STYLE: not a registered style, but resolved
# at cycle time to a random pick from the registry.
RANDOM_TOKEN = "random"

_STYLES: dict[str, object] = {}  # style_name → module object


def _is_valid_style_module(mod) -> bool:
    return (
        isinstance(getattr(mod, "midjourney_suffix", None), str)
        and isinstance(getattr(mod, "motion_guidance", None), str)
        and callable(getattr(mod, "build_image_prompt", None))
    )


def reload_styles() -> None:
    """
    Rescan the package, refresh the registry. Idempotent.

    Picks up newly-added files (importlib.import_module) and edits to existing
    files (importlib.reload). Files starting with "_" or named "base" are
    skipped. Raises ValueError if any discovered file fails the contract.
    """
    new_registry: dict[str, object] = {}
    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(__path__):
        if mod_name.startswith("_") or mod_name == "base":
            continue
        full = f"{__name__}.{mod_name}"
        if full in sys.modules:
            mod = importlib.reload(sys.modules[full])
        else:
            mod = importlib.import_module(full)
        if not _is_valid_style_module(mod):
            raise ValueError(
                f"styles/{mod_name}.py does not satisfy the style contract "
                f"(must define midjourney_suffix: str, motion_guidance: str, "
                f"and build_image_prompt(*, funny, is_zit, ctx))"
            )
        new_registry[mod_name] = mod
    _STYLES.clear()
    _STYLES.update(new_registry)


def _ensure_loaded() -> None:
    if not _STYLES:
        reload_styles()


def get_style(name: str):
    """Return the style module for *name*. Raises ValueError if unknown."""
    _ensure_loaded()
    try:
        return _STYLES[name]
    except KeyError:
        raise ValueError(
            f"Unknown image_style: {name!r}. Available: {known_styles()}"
        )


def known_styles() -> list[str]:
    """Return the sorted list of registered style names."""
    _ensure_loaded()
    return sorted(_STYLES)


def pick_random_style() -> str:
    """Return one randomly-chosen registered style name."""
    available = known_styles()
    if not available:
        raise RuntimeError("No styles registered — styles/ is empty?")
    return random.choice(available)


def validate_cycle(cycle: list[str]) -> None:
    """
    Raise ValueError if any entry is neither a registered style nor RANDOM_TOKEN.

    main.py calls this at the start of every cycle (after reload_styles()).
    """
    _ensure_loaded()
    valid = set(_STYLES) | {RANDOM_TOKEN}
    unknown = [s for s in cycle if s not in valid]
    if unknown:
        raise ValueError(
            f"IMAGE_STYLE contains unknown styles: {unknown}. "
            f"Available: {known_styles()} (or {RANDOM_TOKEN!r})"
        )


__all__ = [
    "PromptContext",
    "RANDOM_TOKEN",
    "reload_styles",
    "get_style",
    "known_styles",
    "pick_random_style",
    "validate_cycle",
]
