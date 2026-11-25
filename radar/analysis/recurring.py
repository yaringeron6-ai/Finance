"""Question 3 of the historical scan: does the same gap come back in the same month every year?

A spike is a month that sits far from its neighbouring months. When spikes land
on the same calendar month in most of the years available, that points to a
process (a closing step, an annual charge) rather than chance.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..i18n import L, in_month, month_name
from ..model import CLEARING, HISTORICAL, PNL_KINDS, SHOULD_CLEAR_GROUPS, Finding
from .common import accounts, evidence, ils, local_residuals, robust_scale


def find_recurring(ledger, z: float = 4.0, min_years: int = 2, min_share: float = 0.6,
                   min_amount: float = 2_000) -> list[Finding]:
    findings = []
    for acc in accounts(ledger):
        is_pnl = acc["kind"] in PNL_KINDS
        if not is_pnl and acc["group"] not in SHOULD_CLEAR_GROUPS:
            continue
        # Clearing-type accounts: month-end balances (they should be zero); P&L: the month's activity.
        series = acc["activity"] if is_pnl else acc["balance"] * (1 if acc["balance"].iloc[-1] >= 0 else -1)
        x = series.to_numpy()
        res = local_residuals(x)
        scale = robust_scale(res, float(np.median(np.abs(x))), min_amount)
        flagged = (np.abs(res) / scale >= z) & (np.abs(res) >= min_amount)

        years_by_month: dict[int, list[int]] = defaultdict(list)
        excess_by_month: dict[int, list[float]] = defaultdict(list)
        for p, f, r in zip(series.index, flagged, res):
            if f:
                years_by_month[p.month].append(p.year)
                excess_by_month[p.month].append(float(r))
        available = defaultdict(set)
        for p in series.index:
            available[p.month].add(p.year)

        recurring = [m for m, ys in years_by_month.items()
                     if len(set(ys)) >= min_years and len(set(ys)) / len(available[m]) >= min_share
                     and len({np.sign(e) for e in excess_by_month[m]}) == 1]
        if not recurring:
            continue
        recurring.sort()
        main = max(recurring, key=lambda m: sum(abs(e) for e in excess_by_month[m]))
        main_total = sum(abs(e) for e in excess_by_month[main])
        main_avg = ils(main_total / len(excess_by_month[main]))
        years = sorted(set(years_by_month[main]))
        years_text = ", ".join(map(str, years))
        months_text = L(" ו", " and ").join(month_name(m) for m in recurring)

        if acc["group"] == CLEARING:
            title = L(f"{acc['name']}: נשאר פתוח בסוף {month_name(main)} כל שנה",
                      f"{acc['name']}: left open at the end of {month_name(main)} every year")
            reason = L(f"ב-{len(years)} שנים ({years_text}) החשבון נסגר {in_month(main)} עם יתרה של כ-{main_avg} "
                       f"שהתאפסה רק אחר כך. הדפוס חוזר {in_month(main)}, ולכן זה נראה כמו שלב בתהליך הסגירה ולא כמו מקרה.",
                       f"In {len(years)} years ({years_text}) the account closed {in_month(main)} with about {main_avg} "
                       f"that was cleared only later. The same month every year points to a step in the closing "
                       f"process rather than chance.")
            check = L(f"לעבור על תהליך הסגירה של {month_name(main)}: מה נרשם בחשבון המעבר ולמה הוא מתאפס רק בחודש שאחרי.",
                      f"Walk through the {month_name(main)} close: what goes into the clearing account and why it "
                      f"clears only the month after.")
            shown, label = acc["balance"], L("יתרת סוף חודש", "Month-end balance")
        else:
            high = np.mean(excess_by_month[main]) > 0
            title = L(f"{acc['name']}: חריגה חוזרת ב{months_text} בכל שנה",
                      f"{acc['name']}: the same spike in {months_text} every year")
            reason = L(f"ב-{len(years)} שנים ({years_text}) הסכום {in_month(main)} היה "
                       f"{'גבוה' if high else 'נמוך'} בכ-{main_avg} מהחודשים שסביבו.",
                       f"In {len(years)} years ({years_text}) the amount {in_month(main)} was about {main_avg} "
                       f"{'above' if high else 'below'} the months around it.")
            check = L(f"לבדוק מה נרשם בכרטיס {in_month(main)} בכל שנה. אם זה חיוב שנתי ידוע, אפשר לסמן כצפוי.",
                      f"Check what is booked to the account {in_month(main)} each year. If it is a known "
                      f"annual charge, mark it as expected.")
            shown, label = acc["activity"], L("סכום חודשי", "Monthly amount")

        highlight = [p for p, f in zip(series.index, flagged) if f and p.month in recurring]
        findings.append(Finding(
            section=HISTORICAL,
            kind="recurring",
            account=acc["code"],
            account_name=acc["name"],
            title=title,
            reason=reason,
            check=check,
            amount=main_total,
            periods=[f"{p.month:02d}/{p.year}" for p in highlight],
            evidence=evidence(label, shown, highlight=highlight),
            details={"months": recurring, "years": years,
                     "average_excess": round(main_total / len(excess_by_month[main]), 2)},
        ))
    return findings
