"""Alerts: an expense climbing (or revenue falling) several months in a row.

When a monthly target exists for the account, the alert also says when the
current pace crosses it.
"""

from __future__ import annotations

import math

import numpy as np

from ..i18n import L
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

        pct = f"{abs(change) * 100:.0f}%"
        if rising:
            reason = L(f"עלה {run} חודשים ברציפות, מ-{ils(values[start])} ל-{ils(last)}. "
                       f"זה {pct} מעל הרמה הרגילה של השנה שקדמה ({ils(baseline)}).",
                       f"Up {run} months in a row, from {ils(values[start])} to {ils(last)}. "
                       f"That is {pct} above the usual level of the year before ({ils(baseline)}).")
        else:
            reason = L(f"ירד {run} חודשים ברציפות, מ-{ils(values[start])} ל-{ils(last)}. "
                       f"זה {pct} מתחת לרמה הרגילה של השנה שקדמה ({ils(baseline)}).",
                       f"Down {run} months in a row, from {ils(values[start])} to {ils(last)}. "
                       f"That is {pct} below the usual level of the year before ({ils(baseline)}).")
        target = targets.get(acc["code"])
        reference, reference_label = baseline, L("רמה רגילה", "Usual level")
        details = {"run": run, "baseline": round(baseline, 2), "slope": round(slope, 2),
                   "projected": [round(p, 2) for p in projected]}
        if target:
            reference, reference_label = target, L("יעד חודשי", "Monthly target")
            crossed = last >= target if rising else last <= target
            if crossed:
                reason += L(f" הכרטיס כבר {'מעל ה' if rising else 'מתחת ל'}יעד החודשי ({ils(target)}).",
                            f" It is already {'above' if rising else 'below'} its monthly target ({ils(target)}).")
                details["months_to_target"] = 0
            elif (slope > 0) == rising and slope != 0:
                months = math.ceil((target - last) / slope)
                soon = months <= 3
                reason += L(f" בקצב הזה הוא יחצה את היעד החודשי ({ils(target)}) בעוד {months} חודשים"
                            f"{', כלומר ברבעון הקרוב' if soon else ''}.",
                            f" At this pace it crosses its monthly target ({ils(target)}) in {months} months"
                            f"{', within the next quarter' if soon else ''}.")
                details["months_to_target"] = months
        if rising:
            reason += L(f" אם הקצב יימשך, {horizon} החודשים הבאים יעלו כ-{ils(at_stake)} מעבר לרגיל.",
                        f" If the pace holds, the next {horizon} months cost about {ils(at_stake)} more than usual.")
        else:
            reason += L(f" אם הקצב יימשך, {horizon} החודשים הבאים יכניסו כ-{ils(at_stake)} פחות מהרגיל.",
                        f" If the pace holds, the next {horizon} months bring in about {ils(at_stake)} less than usual.")

        run_periods = list(x.index[start + 1:])
        findings.append(Finding(
            section=ALERT,
            kind="trend",
            account=acc["code"],
            account_name=acc["name"],
            title=(L(f"{acc['name']}: עלה {run} חודשים ברציפות", f"{acc['name']}: up {run} months in a row") if rising
                   else L(f"{acc['name']}: ירד {run} חודשים ברציפות", f"{acc['name']}: down {run} months in a row")),
            reason=reason,
            check=(L("לבדוק מה מניע את העלייה (ספק, קמפיין, מחיר) והאם היא מתוכננת.",
                     "Find what drives the rise (a supplier, a campaign, a price) and whether it was planned.")
                   if rising else
                   L("לבדוק אם הירידה נובעת מלקוח מסוים, ממחיר או מתזמון חשבוניות.",
                     "Check whether the drop comes from one customer, from pricing or from invoice timing.")),
            amount=at_stake,
            periods=[period_label(p) for p in run_periods],
            evidence=evidence(L("סכום חודשי", "Monthly amount"), x.iloc[-18:], highlight=run_periods,
                              reference=reference, reference_label=reference_label),
            details=details,
        ))
    return findings
