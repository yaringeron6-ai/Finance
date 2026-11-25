"""Question 1 and 2 of the historical scan: balances that were opened and never closed.

Applies to accounts that are supposed to clear: clearing/suspense accounts,
advances and one-off provisions. For each, the floor of the balance over the
last `min_months` months is the amount that has been sitting there at least
that long. When the balance revolves (new advances replace old ones), the floor
can include current items too, so the finding says what it measured and asks
for a breakdown rather than calling it an error. With ledger lines uploaded,
`open_items` replaces this estimate with the actual lines.
"""

from __future__ import annotations

import numpy as np

from ..i18n import L
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
        months_open = len(values) - start
        since = period_label(oriented.index[start])
        later_moves = acc["movement"].iloc[start + 1:]
        untouched = bool((later_moves.abs() < 1).all())
        amount = ils(floor)

        group = acc["group"]
        provision_check = L("לבדוק עם הגורם המטפל אם ההפרשה עדיין נדרשת ובאיזה סכום.",
                            "Check with whoever owns it whether the provision is still needed, and at what amount.")
        if group == CLEARING:
            reason = L(f"חשבון מעבר אמור להתאפס בסוף כל חודש. מאז {since} היתרה לא ירדה מתחת ל-{amount}.",
                       f"A clearing account should be zero at every month end. Since {since} the balance "
                       f"has not gone below {amount}.")
            check = L(f"לפרט את תנועות הכרטיס מ-{since} ולזהות מה לא הועבר לכרטיס היעד.",
                      f"List the account's entries since {since} and find what was never moved to its target account.")
        elif group == ADVANCE:
            reason = L(f"מקדמות אמורות להיסגר מול חשבונית. מאז {since} היתרה לא ירדה מתחת ל-{amount}, "
                       f"כלומר סכום כזה פתוח בכרטיס {months_open} חודשים ברציפות.",
                       f"Advances should be closed against an invoice. Since {since} the balance has not gone "
                       f"below {amount}, so that much has been open for {months_open} months in a row.")
            check = L("לפרט את היתרה לפי ספק ולבדוק אילו מקדמות פתוחות יותר משנה ולמה לא התקבלה חשבונית.",
                      "Break the balance down by supplier and find which advances are over a year old "
                      "and why no invoice arrived.")
        elif group == PROVISION and untouched:
            reason = L(f"ההפרשה נרשמה ב-{since} ומאז לא נוצלה, לא עודכנה ולא בוטלה.",
                       f"The provision was booked in {since} and has not been used, updated or released since.")
            check = provision_check
        else:
            reason = L(f"מאז {since} היתרה לא ירדה מתחת ל-{amount}.",
                       f"Since {since} the balance has not gone below {amount}.")
            check = provision_check
        if since_start_of_data:
            reason += L(" היתרה הזאת קיימת כבר מתחילת הנתונים שהועלו, כך שייתכן שהיא ותיקה יותר.",
                        " It is there from the first month of the data, so it may be older still.")

        findings.append(Finding(
            section=HISTORICAL,
            kind="stale_balance",
            account=acc["code"],
            account_name=acc["name"],
            title=L(f"{acc['name']}: {amount} פתוחים {months_open} חודשים",
                    f"{acc['name']}: {amount} open for {months_open} months"),
            reason=reason,
            check=check,
            amount=floor,
            periods=[since],
            evidence=evidence(L("יתרת סוף חודש", "Month-end balance"), oriented, highlight=oriented.index[start:],
                              reference=floor, reference_label=L("סכום שלא זז", "Amount not moving")),
            details={"months_open": months_open, "since": since, "group": group},
        ))
    return findings
