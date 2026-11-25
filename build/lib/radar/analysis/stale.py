"""Question 1 and 2 of the historical scan: balances that were opened and never closed.

Applies to accounts that are supposed to clear: clearing/suspense accounts,
advances and one-off provisions. For each, the floor of the balance over the
last `min_months` months is the amount that has been sitting there at least
that long. When the balance revolves (new advances replace old ones), the floor
can include current items too, so the finding says what it measured and asks
for a breakdown rather than calling it an error.
"""

from __future__ import annotations

import numpy as np

from ..model import ADVANCE, CLEARING, HISTORICAL, PROVISION, SHOULD_CLEAR_GROUPS, Finding
from .common import accounts, evidence, ils, period_label


def find_stale(ledger, min_months: int = 12, min_amount: float = 2_000) -> list[Finding]:
    findings = []
    for acc in accounts(ledger):
        if acc["group"] not in SHOULD_CLEAR_GROUPS:
            continue
        balance = acc["balance"]
        if len(balance) < min_months or abs(balance.iloc[-1]) < min_amount:
            continue
        sign = np.sign(balance.iloc[-1])
        oriented = balance * sign
        floor = float(oriented.iloc[-min_months:].min())
        if floor < min_amount:
            continue

        # How far back has the balance stayed at or above (about) the floor?
        values = oriented.to_numpy()
        start = len(values) - 1
        while start > 0 and values[start - 1] >= floor * 0.95:
            start -= 1
        since_start_of_data = start == 0
        start_period = oriented.index[start]
        months_open = len(values) - start
        since = period_label(start_period)
        later_moves = acc["movement"].iloc[start + 1:]
        untouched = bool((later_moves.abs() < 1).all())

        group = acc["group"]
        if group == CLEARING:
            reason = (f"חשבון מעבר אמור להתאפס בסוף כל חודש. "
                      f"מאז {since} היתרה לא ירדה מתחת ל-{ils(floor)}.")
            check = f"לפרט את תנועות הכרטיס מ-{since} ולזהות מה לא הועבר לכרטיס היעד."
        elif group == ADVANCE:
            reason = (f"מקדמות אמורות להיסגר מול חשבונית. מאז {since} היתרה לא ירדה מתחת ל-{ils(floor)}, "
                      f"כלומר סכום כזה פתוח בכרטיס {months_open} חודשים ברציפות.")
            check = "לפרט את היתרה לפי ספק ולבדוק אילו מקדמות פתוחות יותר משנה ולמה לא התקבלה חשבונית."
        elif group == PROVISION and untouched:
            reason = f"ההפרשה נרשמה ב-{since} ומאז לא נוצלה, לא עודכנה ולא בוטלה."
            check = "לבדוק עם הגורם המטפל אם ההפרשה עדיין נדרשת ובאיזה סכום."
        else:
            reason = f"מאז {since} היתרה לא ירדה מתחת ל-{ils(floor)}."
            check = "לבדוק עם הגורם המטפל אם ההפרשה עדיין נדרשת ובאיזה סכום."
        if since_start_of_data:
            reason += " היתרה הזאת קיימת כבר מתחילת הנתונים שהועלו, כך שייתכן שהיא ותיקה יותר."

        findings.append(Finding(
            section=HISTORICAL,
            kind="stale_balance",
            account=acc["code"],
            account_name=acc["name"],
            title=f"{acc['name']}: {ils(floor)} פתוחים {months_open} חודשים",
            reason=reason,
            check=check,
            amount=floor,
            periods=[since],
            evidence=evidence("יתרת סוף חודש", oriented, highlight=oriented.index[start:],
                              reference=floor, reference_label="סכום שלא זז"),
            details={"months_open": months_open, "since": since, "group": group},
        ))
    return findings
