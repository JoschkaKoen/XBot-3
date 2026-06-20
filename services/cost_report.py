"""Cost computation from accumulated token usage and AI pricing.

Pricing is merged from two sources (per model, RMB per 1M tokens):

1. A bundled, in-repo fallback — ``data/ai_pricing.json`` — so cost reports
   work on any machine even without the shared spreadsheet.
2. ``AI API costs.xlsx`` — the shared source-of-truth spreadsheet (lives in
   eXercise's repo). When present it **overrides** the bundled values per
   model; the bundled values fill in any model the spreadsheet lacks.

Spreadsheet search order:
1. ``$AI_COSTS_XLSX`` if set
2. ``~/Programming/eXercise/AI API costs.xlsx``
3. ``<XBot-3 repo root>/AI API costs.xlsx`` (legacy fallback)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("xbot.cost_report")

_pricing_cache: dict[str, tuple[float, float]] | None = None  # model → (input_rate, output_rate)
_pricing_sources: list[str] = []  # human-readable descriptions of sources actually loaded

_BUNDLED_PRICING = Path(__file__).parents[1] / "data" / "ai_pricing.json"


def _resolve_pricing_file() -> Path | None:
    """Find the AI API costs spreadsheet. Returns None if no candidate exists."""
    env = os.getenv("AI_COSTS_XLSX", "").strip()
    candidates = [
        Path(env) if env else None,
        Path.home() / "Programming" / "eXercise" / "AI API costs.xlsx",
        Path(__file__).parents[1] / "AI API costs.xlsx",
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


def _load_bundled() -> dict[str, tuple[float, float]]:
    """Load the in-repo fallback prices from data/ai_pricing.json ({} on any error)."""
    out: dict[str, tuple[float, float]] = {}
    try:
        if not _BUNDLED_PRICING.is_file():
            return out
        data = json.loads(_BUNDLED_PRICING.read_text(encoding="utf-8"))
        prices = data.get("rmb_per_1m_tokens", data.get("prices", {}))
        for model, v in prices.items():
            if isinstance(v, dict):
                out[str(model).strip()] = (float(v.get("input", 0) or 0), float(v.get("output", 0) or 0))
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                out[str(model).strip()] = (float(v[0] or 0), float(v[1] or 0))
    except Exception as exc:
        logger.warning("Bundled pricing %s unreadable: %s", _BUNDLED_PRICING, exc)
    return out


def _load_xlsx(path: Path) -> dict[str, tuple[float, float]]:
    """Parse the AI API costs spreadsheet ({} if missing/unreadable)."""
    result: dict[str, tuple[float, float]] = {}
    try:
        import openpyxl  # noqa: PLC0415
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = iter(ws.rows)
        headers = [str(c.value).strip() if c.value else "" for c in next(rows)]
        model_col = next((i for i, h in enumerate(headers) if "model" in h.lower()), None)
        inp_col   = next((i for i, h in enumerate(headers) if "input" in h.lower()), None)
        out_col   = next((i for i, h in enumerate(headers) if "output" in h.lower()), None)
        if None not in (model_col, inp_col, out_col):
            for row in rows:
                model = str(row[model_col].value or "").strip()
                if not model:
                    continue
                try:
                    inp = float(row[inp_col].value or 0)
                    out = float(row[out_col].value or 0)
                except (TypeError, ValueError):
                    inp, out = 0.0, 0.0
                result[model] = (inp, out)
        wb.close()
    except Exception as exc:
        logger.warning("Pricing spreadsheet %s unreadable: %s", path, exc)
    return result


def _load_pricing() -> dict[str, tuple[float, float]]:
    """Merge bundled + spreadsheet pricing (cached after first call).

    Bundled values are loaded first; the spreadsheet (when found) overrides
    them per model. If neither source yields rows, returns {} and logs a clear
    warning so the resulting ¥0.00 cost report is explained rather than silent.
    """
    global _pricing_cache, _pricing_sources
    if _pricing_cache is not None:
        return _pricing_cache

    result: dict[str, tuple[float, float]] = {}
    sources: list[str] = []

    bundled = _load_bundled()
    if bundled:
        result.update(bundled)
        sources.append(f"{_BUNDLED_PRICING.name} ({len(bundled)} models)")

    xlsx_path = _resolve_pricing_file()
    if xlsx_path is not None:
        xlsx = _load_xlsx(xlsx_path)
        if xlsx:
            result.update(xlsx)  # spreadsheet overrides bundled
            sources.append(f"{xlsx_path} ({len(xlsx)} models)")

    _pricing_sources = sources
    if sources:
        logger.info("AI pricing loaded from: %s", " ; ".join(sources))
    else:
        logger.warning(
            "No AI pricing source found — costs will show ¥0. Add prices to "
            "%s or set AI_COSTS_XLSX (spreadsheet not found; AI_COSTS_XLSX=%r).",
            _BUNDLED_PRICING, os.getenv("AI_COSTS_XLSX", "") or "(unset)",
        )
    _pricing_cache = result
    return result


def has_pricing() -> bool:
    """True if any pricing source loaded at least one model."""
    return bool(_load_pricing())


def pricing_sources_desc() -> str:
    """Human-readable description of the loaded pricing source(s), '' if none."""
    _load_pricing()
    return " ; ".join(_pricing_sources)


def compute_cost(
    usage: dict[str, dict[str, int]],
) -> tuple[float, dict[str, dict]]:
    """Return (total_rmb, per_model_breakdown).

    breakdown: model → {"input_tokens": N, "output_tokens": N,
                        "thinking_tokens": N, "cost_rmb": X, "priced": bool}
    Rates come from the merged pricing sources (RMB per 1M tokens); a model with
    no rate gets 0.0 and ``"priced": False`` so callers can flag it.

    ``output_tokens`` is the total billed output (visible + thinking) and is
    multiplied by the output rate. ``thinking_tokens`` is the thinking portion
    of ``output_tokens`` and is informational — not double-counted in the cost.
    """
    pricing = _load_pricing()
    breakdown: dict[str, dict] = {}
    total = 0.0
    for model, counts in usage.items():
        priced = model in pricing
        inp_rate, out_rate = pricing.get(model, (0.0, 0.0))
        in_tokens = counts.get("input", 0)
        out_tokens = counts.get("output", 0)
        cost = in_tokens / 1_000_000 * inp_rate + out_tokens / 1_000_000 * out_rate
        total += cost
        breakdown[model] = {
            "input_tokens":    in_tokens,
            "output_tokens":   out_tokens,
            "thinking_tokens": counts.get("thinking", 0),
            "cost_rmb":        round(cost, 6),
            "priced":          priced,
        }
    return round(total, 6), breakdown
