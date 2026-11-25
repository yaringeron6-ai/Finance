"""Correlations between accounts, and alerts when a stable relationship breaks.

Two P&L accounts are related when their month-to-month changes move together
(so a shared growth trend alone does not count) and the ratio between them is
stable. An alert fires when the ratio in the most recent months leaves its
historical range, e.g. cost of sales jumping from 60% to 68% of revenue.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..model import ALERT, PNL_KINDS, REVENUE, Finding
from .common import accounts, evidence, ils, pct, period_label


def find_relations(ledger, recent: int = 3, min_history: int = 12, min_corr: float = 0.7,
                   max_cv: float = 0.15, z: float = 3.0, min_amount: float = 2_000):
    """Return (findings, relationships)."""
    candidates = []
    for acc in accounts(ledger):
        if acc["kind"] not in PNL_KINDS:
            continue
        values = acc["activity"].to_numpy()
        hist = values[:-recent]
        diffs = np.diff(hist)
        if len(hist) < min_history or np.count_nonzero(np.abs(diffs) > 1) < 6 or (hist <= 0).any():
            continue
        candidates.append((acc, hist, diffs))

    findings, relationships = [], []
    for (a, ha, da), (b, hb, db) in combinations(candidates, 2):
        corr = float(np.corrcoef(da, db)[0, 1])
        if not np.isfinite(corr) or corr < min_corr:
            continue
        # x is the driver: revenue when present, otherwise the larger account.
        if b["kind"] == REVENUE and a["kind"] != REVENUE or (
                a["kind"] != REVENUE and b["kind"] != REVENUE and hb.mean() > ha.mean()):
            (a, ha), (b, hb) = (b, hb), (a, ha)
        x_acc, y_acc, hx, hy = a, b, ha, hb
        ratio_hist = hy / hx
        mean, std = float(ratio_hist.mean()), float(ratio_hist.std(ddof=1))
        if mean <= 0 or std / mean > max_cv:
            continue
        relationships.append({
            "driver": x_acc["code"], "driver_name": x_acc["name"],
            "account": y_acc["code"], "account_name": y_acc["name"],
            "correlation": round(corr, 2), "ratio": round(mean, 4),
        })

        x_recent = x_acc["activity"].to_numpy()[-recent:]
        y_recent = y_acc["activity"].to_numpy()[-recent:]
        if (x_recent <= 0).any():
            continue
        ratio_recent = y_recent / x_recent
        score = (ratio_recent.mean() - mean) / max(std, 1e-9)
        excess = float(((ratio_recent - mean) * x_recent).sum())
        if abs(score) < z or abs(excess) < min_amount:
            continue

        ratio_series = y_acc["activity"] / x_acc["activity"] * 100
        recent_periods = list(ratio_series.index[-recent:])
        direction = "יותר" if excess > 0 else "פחות"
        findings.append(Finding(
            section=ALERT,
            kind="relation_break",
            account=y_acc["code"],
            account_name=y_acc["name"],
            title=(f"{y_acc['name']}: {pct(ratio_recent.mean())} מ{x_acc['name']} "
                   f"לעומת {pct(mean)} בדרך כלל"),
            reason=(f"ב-{len(hx)} חודשים {y_acc['name']} נע יחד עם {x_acc['name']} (מתאם {corr:.2f}) "
                    f"ועמד בממוצע על {pct(mean)} ממנו. ב-{recent} החודשים האחרונים היחס היה {pct(ratio_recent.mean())}, "
                    f"כ-{ils(abs(excess))} {direction} ממה שהיחס הרגיל היה נותן."),
            check="לבדוק אם השתנו מחירי ספקים, תמהיל המוצרים או סיווג התנועות בין שני הכרטיסים.",
            amount=abs(excess),
            periods=[period_label(p) for p in recent_periods],
            evidence=evidence(f"% מ{x_acc['name']}", ratio_series, highlight=recent_periods,
                              reference=mean * 100, reference_label="יחס רגיל"),
            details={"driver": x_acc["code"], "driver_name": x_acc["name"], "correlation": round(corr, 2),
                     "ratio_hist": round(mean, 4), "ratio_recent": round(float(ratio_recent.mean()), 4),
                     "score": round(float(score), 1)},
        ))
    return findings, relationships
