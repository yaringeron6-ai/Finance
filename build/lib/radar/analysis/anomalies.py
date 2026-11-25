"""Alerts: a recent month that is far from the account's own last 12 months."""

from __future__ import annotations

import numpy as np

from ..model import ALERT, PNL_KINDS, Finding
from .common import accounts, evidence, ils, period_label, robust_scale


def find_anomalies(ledger, recent: int = 3, lookback: int = 12, z: float = 4.0,
                   min_amount: float = 2_000) -> list[Finding]:
    findings = []
    for acc in accounts(ledger):
        if acc["kind"] not in PNL_KINDS:
            continue
        x = acc["activity"]
        values = x.to_numpy()
        n = len(values)
        best = None
        for t in range(max(lookback, n - recent), n):
            history = values[t - lookback:t]
            baseline = float(np.median(history))
            scale = robust_scale(history, baseline, min_amount)
            deviation = values[t] - baseline
            score = abs(deviation) / scale
            if score >= z and abs(deviation) >= min_amount and (best is None or score > best[1]):
                best = (t, score, baseline, deviation)
        if best is None:
            continue
        t, score, baseline, deviation = best
        period = x.index[t]
        change = deviation / baseline if baseline else np.inf
        direction = "מעל ה" if deviation > 0 else "מתחת ל"
        findings.append(Finding(
            section=ALERT,
            kind="anomaly",
            account=acc["code"],
            account_name=acc["name"],
            title=f"{acc['name']}: {period_label(period)} חריג, {ils(values[t])} לעומת {ils(baseline)} בדרך כלל",
            reason=(f"הסכום ב-{period_label(period)} היה {abs(change) * 100:.0f}% {direction}חציון של "
                    f"{lookback} החודשים שלפניו, הרבה מעבר לתנודה הרגילה של הכרטיס."),
            check="לפתוח את תנועות החודש ולבדוק אם זה חיוב חד-פעמי, רישום בכרטיס הלא נכון או שינוי אמיתי.",
            amount=abs(deviation),
            periods=[period_label(period)],
            evidence=evidence("סכום חודשי", x.iloc[-18:], highlight=[period],
                              reference=baseline, reference_label="רמה רגילה"),
            details={"score": round(score, 1), "baseline": round(baseline, 2)},
        ))
    return findings
