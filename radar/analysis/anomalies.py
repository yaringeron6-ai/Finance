"""Alerts: a recent month that is far from the account's own last 12 months."""

from __future__ import annotations

import numpy as np

from ..i18n import L
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
        period = period_label(x.index[t])
        change = abs(deviation / baseline * 100) if baseline else float("inf")
        above = deviation > 0
        findings.append(Finding(
            section=ALERT,
            kind="anomaly",
            account=acc["code"],
            account_name=acc["name"],
            title=L(f"{acc['name']}: {period} חריג, {ils(values[t])} לעומת {ils(baseline)} בדרך כלל",
                    f"{acc['name']}: {period} is unusual, {ils(values[t])} against a usual {ils(baseline)}"),
            reason=L(f"הסכום ב-{period} היה {change:.0f}% {'מעל ה' if above else 'מתחת ל'}חציון של "
                     f"{lookback} החודשים שלפניו, הרבה מעבר לתנודה הרגילה של הכרטיס.",
                     f"{period} was {change:.0f}% {'above' if above else 'below'} the median of the "
                     f"{lookback} months before it, far beyond the account's normal swings."),
            check=L("לפתוח את תנועות החודש ולבדוק אם זה חיוב חד-פעמי, רישום בכרטיס הלא נכון או שינוי אמיתי.",
                    "Open the month's entries and see whether it is a one-off charge, a posting to the wrong "
                    "account, or a real change."),
            amount=abs(deviation),
            periods=[period],
            evidence=evidence(L("סכום חודשי", "Monthly amount"), x.iloc[-18:], highlight=[x.index[t]],
                              reference=baseline, reference_label=L("רמה רגילה", "Usual level")),
            details={"score": round(score, 1), "baseline": round(baseline, 2)},
        ))
    return findings
