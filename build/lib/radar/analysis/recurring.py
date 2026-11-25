"""Question 3 of the historical scan: does the same gap come back in the same month every year?

A spike is a month that sits far from its neighbouring months. When spikes land
on the same calendar month in most of the years available, that points to a
process (a closing step, an annual charge) rather than chance.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..model import CLEARING, HISTORICAL, PNL_KINDS, SHOULD_CLEAR_GROUPS, Finding
from .common import HE_MONTHS, accounts, evidence, ils, local_residuals, robust_scale


def find_recurring(ledger, z: float = 4.0, min_years: int = 2, min_share: float = 0.6,
                   min_amount: float = 2_000) -> list[Finding]:
    findings = []
    for acc in accounts(ledger):
        is_pnl = acc["kind"] in PNL_KINDS
        if not is_pnl and acc["group"] not in SHOULD_CLEAR_GROUPS:
            continue
        # Clearing accounts: look at month-end balances (they should be zero);
        # P&L accounts: at the month's activity.
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
        main_avg = main_total / len(excess_by_month[main])
        years = sorted(set(years_by_month[main]))
        months_text = " ו".join(HE_MONTHS[m - 1] for m in recurring)

        if acc["group"] == CLEARING:
            title = f"{acc['name']}: נשאר פתוח בסוף {HE_MONTHS[main - 1]} כל שנה"
            reason = (f"ב-{len(years)} שנים ({', '.join(map(str, years))}) החשבון נסגר ב{HE_MONTHS[main - 1]} "
                      f"עם יתרה של כ-{ils(main_avg)} שהתאפסה רק אחר כך. "
                      f"הדפוס חוזר ב{months_text}, ולכן זה נראה כמו שלב בתהליך הסגירה ולא כמו מקרה.")
            check = f"לעבור על תהליך הסגירה של {HE_MONTHS[main - 1]}: מה נרשם בחשבון המעבר ולמה הוא מתאפס רק בחודש שאחרי."
            shown, label = acc["balance"], "יתרת סוף חודש"
        else:
            direction = "גבוה" if np.mean(excess_by_month[main]) > 0 else "נמוך"
            title = f"{acc['name']}: חריגה חוזרת ב{months_text} בכל שנה"
            reason = (f"ב-{len(years)} שנים ({', '.join(map(str, years))}) הסכום ב{HE_MONTHS[main - 1]} "
                      f"היה {direction} בכ-{ils(main_avg)} מהחודשים שסביבו.")
            check = (f"לבדוק מה נרשם בכרטיס ב{HE_MONTHS[main - 1]} בכל שנה. "
                     f"אם זה חיוב שנתי ידוע, אפשר לסמן כצפוי.")
            shown, label = acc["activity"], "סכום חודשי"

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
            details={"months": recurring, "years": years, "average_excess": round(main_avg, 2)},
        ))
    return findings
