"""Cost computation from accumulated token usage and AI API costs.xlsx.

Ported from eXercise/xscore/shared/cost_report.py with one change:
``_PRICING_FILE`` resolves via env-var search order so XBot-3 and eXercise
can share a single source-of-truth spreadsheet (lives in eXercise's repo).

Search order:
1. ``$AI_COSTS_XLSX`` if set
2. ``~/Programming/eXercise/AI API costs.xlsx``
3. ``<XBot-3 repo root>/AI API costs.xlsx`` (legacy fallback)
"""
from __future__ import annotations

import os
from pathlib import Path

_pricing_cache: dict[str, tuple[float, float]] | None = None  # model → (input_rate, output_rate)


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


def _load_pricing() -> dict[str, tuple[float, float]]:
    """Load pricing from AI API costs.xlsx (cached after first call).

    Falls back silently to {} if the file is missing or unreadable —
    cost reports will then show ¥0.00 with a "prices not found" hint.
    """
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache
    result: dict[str, tuple[float, float]] = {}
    path = _resolve_pricing_file()
    if path is None:
        _pricing_cache = result
        return result
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
    except Exception:
        pass  # file unreadable → all costs are 0
    _pricing_cache = result
    return result


def compute_cost(
    usage: dict[str, dict[str, int]],
) -> tuple[float, dict[str, dict]]:
    """Return (total_rmb, per_model_breakdown).

    breakdown: model → {"input_tokens": N, "output_tokens": N,
                        "thinking_tokens": N, "cost_rmb": X}
    Rates come from AI API costs.xlsx (RMB per 1M tokens); 0.0 if model not listed.

    ``output_tokens`` is the total billed output (visible + thinking) and is
    multiplied by the output rate. ``thinking_tokens`` is the thinking portion
    of ``output_tokens`` and is informational — not double-counted in the cost.
    """
    pricing = _load_pricing()
    breakdown: dict[str, dict] = {}
    total = 0.0
    for model, counts in usage.items():
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
        }
    return round(total, 6), breakdown
