from __future__ import annotations

import numpy as np
import pandas as pd

from ..i18n import HE_MONTHS, lang  # noqa: F401  (HE_MONTHS re-exported)
from ..model import Evidence, SeriesPoint, natural_sign


def ils(value: float) -> str:
    sign = "-" if value < 0 else ""
    if lang() == "en":
        return f"{sign}₪{abs(value):,.0f}"
    return f"{sign}{abs(value):,.0f} ₪"


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def period_label(p: pd.Period) -> str:
    return f"{p.month:02d}/{p.year}"


def accounts(ledger: pd.DataFrame) -> list[dict]:
    """One dict per account: code, name, kind, group, and its series by period."""
    out = []
    for code, g in ledger.groupby("account", sort=True):
        g = g.sort_values("period")
        idx = pd.PeriodIndex(g["period"], freq="M")
        kind = g["kind"].iloc[0]
        sign = natural_sign(kind)
        out.append({
            "code": code,
            "name": g["name"].iloc[0],
            "kind": kind,
            "group": g["group"].iloc[0],
            "movement": pd.Series(g["movement"].to_numpy(float), index=idx),
            "balance": pd.Series(g["balance"].to_numpy(float), index=idx),
            # Natural sign: revenue and liabilities shown as positive numbers.
            "activity": pd.Series(g["movement"].to_numpy(float) * sign, index=idx),
            "natural_balance": pd.Series(g["balance"].to_numpy(float) * sign, index=idx),
        })
    return out


def robust_scale(values: np.ndarray, level: float, min_amount: float) -> float:
    """Spread of `values` that ignores outliers, with floors so flat series do not explode."""
    values = values[~np.isnan(values)]
    mad = float(np.median(np.abs(values - np.median(values)))) * 1.4826 if len(values) else 0.0
    return max(mad, 0.05 * abs(level), min_amount / 4, 1.0)


def local_residuals(x: np.ndarray, half_window: int = 6) -> np.ndarray:
    """x minus the median of its neighbours (±half_window months, itself excluded)."""
    n = len(x)
    res = np.empty(n)
    for t in range(n):
        lo, hi = max(0, t - half_window), min(n, t + half_window + 1)
        neighbours = np.concatenate([x[lo:t], x[t + 1:hi]])
        res[t] = x[t] - np.median(neighbours)
    return res


def evidence(label: str, series: pd.Series, highlight=(), reference=None, reference_label=None) -> Evidence:
    return Evidence(
        label=label,
        series=[SeriesPoint(period_label(p), round(float(v), 2)) for p, v in series.items()],
        highlight=[period_label(p) for p in highlight],
        reference=None if reference is None else round(float(reference), 2),
        reference_label=reference_label,
    )
