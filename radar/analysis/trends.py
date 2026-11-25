"""Alerts: an expense climbing (or revenue falling) several months in a row.

When a monthly target exists for the account, the alert also says when the
current pace crosses it.
"""

from __future__ import annotations

import math

import numpy as np

from ..model import ALERT, COGS, EXPENSE, REVENUE, Finding
from .common import accounts, evidence, ils, period_label


def _run_length(values: np.ndarray, rising: bool) -> int:
    run = 0
    for t in range(len(values) - 1, 0, -1):
        step = values[t] - values[t - 1]
        if (step > 0) if rising else (step < 0):
            run += 1
        else:
            break
    return run


def find_trends(ledger, targets: dict[str, float] | None = None, min_run: int = 3,
                min_change: float = 0.15, horizon: int = 3, min_amount: float = 2_000) -> list[Finding]:
    targets = targets or {}
    findings = []
    for acc in accounts(ledger):
        if acc["kind"] not in (REVENUE, COGS, EXPENSE):
            continue
        rising = acc["kind"] != REVENUE  # costs going up, revenue going down
        x = acc["activity"]
        values = x.to_numpy()
        n = len(values)
        run = _run_length(values, rising)
        if run < min_run:
            continue
        start = n - 1 - run  # the month before the first step of the run
        history = values[max(0, start - 11):start + 1]
        baseline = float(np.median(history))
        if baseline <= 0:
            continue
        last = values[-1]
        change = last / baseline - 1
        if (change if rising else -change) < min_change:
            continue

        slope = float(np.polyfit(np.arange(run + 1), values[start:], 1)[0])
        projected = [last + slope * k for k in range(1, horizon + 1)]
        at_stake = sum(abs(p - baseline) for p in projected)
        if at_stake < min_amount:
            continue

        verb = "עלה" if rising else "ירד"
        reason = (f"{verb} {run} חודשים ברציפות, מ-{ils(values[start])} ל-{ils(last)}. "
                  f"זה {abs(change) * 100:.0f}% {'מעל ה' if rising else 'מתחת ל'}רמה הרגילה של השנה שקדמה ({ils(baseline)}).")
        target = targets.get(acc["code"])
        reference, reference_label = baseline, "רמה רגילה"
        details = {"run": run, "baseline": round(baseline, 2), "slope": round(slope, 2),
                   "projected": [round(p, 2) for p in projected]}
        if target:
            reference, reference_label = target, "יעד חודשי"
            crossed = last >= target if rising else last <= target
            if crossed:
                reason += f" הכרטיס כבר {'מעל ה' if rising else 'מתחת ל'}יעד החודשי ({ils(target)})."
                details["months_to_target"] = 0
            elif (slope > 0) == rising and slope != 0:
                months = math.ceil((target - last) / slope)
                when = "כלומר ברבעון הקרוב" if months <= 3 else f"בעוד כ-{months} חודשים"
                reason += f" בקצב הזה הוא יחצה את היעד החודשי ({ils(target)}) בעוד {months} חודשים, {when}."
                details["months_to_target"] = months
        reason += f" אם הקצב יימשך, {horizon} החודשים הבאים יעלו כ-{ils(at_stake)} מעבר לרגיל." if rising else \
            f" אם הקצב יימשך, {horizon} החודשים הבאים יכניסו כ-{ils(at_stake)} פחות מהרגיל."

        run_periods = list(x.index[start + 1:])
        findings.append(Finding(
            section=ALERT,
            kind="trend",
            account=acc["code"],
            account_name=acc["name"],
            title=f"{acc['name']}: {verb} {run} חודשים ברציפות",
            reason=reason,
            check=("לבדוק מה מניע את העלייה (ספק, קמפיין, מחיר) והאם היא מתוכננת." if rising
                   else "לבדוק אם הירידה נובעת מלקוח מסוים, ממחיר או מתזמון חשבוניות."),
            amount=at_stake,
            periods=[period_label(p) for p in run_periods],
            evidence=evidence("סכום חודשי", x.iloc[-18:], highlight=run_periods,
                              reference=reference, reference_label=reference_label),
            details=details,
        ))
    return findings
