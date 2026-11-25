"""Which lines in an account are still open, and how old they are.

Lines that cancel each other are matched in three passes:
  1. same reference (אסמכתא), e.g. an invoice and its receipt;
  2. same amount with opposite sign, earliest first;
  3. first in, first out for whatever is left.
What remains unmatched makes up the balance, line by line, each with its date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..i18n import L
from ..model import (
    ADVANCE, CLEARING, CUSTOMER, HISTORICAL, OPEN_ITEM_GROUPS, PROVISION, SUPPLIER, Finding,
)
from .common import evidence, ils, period_label

AGING_BUCKETS = [(0, 30, "0–30"), (31, 60, "31–60"), (61, 90, "61–90"),
                 (91, 180, "91–180"), (181, 365, "181–365"), (366, 10**6, "365+")]


def _consume(open_amt: np.ndarray, i: int, j: int) -> None:
    """Net line i against line j (opposite signs) as far as both go."""
    take = min(abs(open_amt[i]), abs(open_amt[j]))
    open_amt[i] -= np.sign(open_amt[i]) * take
    open_amt[j] -= np.sign(open_amt[j]) * take


def match_account(lines: pd.DataFrame) -> pd.DataFrame:
    """Return the account's lines with an `open` column: the part of each line not yet matched."""
    lines = lines.sort_values("date").reset_index(drop=True)
    amt = lines["amount"].to_numpy(float).copy()
    refs = lines["reference"].fillna("").astype(str).to_numpy()
    eps = 0.5

    # 1. By reference.
    by_ref: dict[str, list[int]] = {}
    for i, r in enumerate(refs):
        if r:
            by_ref.setdefault(r, []).append(i)
    for idx in by_ref.values():
        if len(idx) < 2:
            continue
        pos = [i for i in idx if amt[i] > eps]
        neg = [i for i in idx if amt[i] < -eps]
        for i in pos:
            for j in neg:
                if abs(amt[i]) > eps and abs(amt[j]) > eps:
                    _consume(amt, i, j)

    # 2. Same amount, opposite sign.
    open_idx = [i for i in range(len(amt)) if abs(amt[i]) > eps]
    by_size: dict[int, list[int]] = {}
    for i in open_idx:
        by_size.setdefault(int(round(abs(amt[i]))), []).append(i)
    for idx in by_size.values():
        pos = [i for i in idx if amt[i] > eps]
        neg = [i for i in idx if amt[i] < -eps]
        for i, j in zip(pos, neg):
            if abs(abs(amt[i]) - abs(amt[j])) <= eps:
                _consume(amt, i, j)

    # 3. FIFO for the rest.
    pos = [i for i in range(len(amt)) if amt[i] > eps]
    neg = [i for i in range(len(amt)) if amt[i] < -eps]
    pi = ni = 0
    while pi < len(pos) and ni < len(neg):
        i, j = pos[pi], neg[ni]
        _consume(amt, i, j)
        if abs(amt[i]) <= eps:
            pi += 1
        if abs(amt[j]) <= eps:
            ni += 1

    out = lines.copy()
    out["open"] = np.where(np.abs(amt) > eps, amt, 0.0)
    return out


def open_items(lines: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Open lines of every account where items should be matched, with their age in days."""
    if lines is None or lines.empty:
        return pd.DataFrame()
    as_of = as_of or (lines["date"].max() + pd.offsets.MonthEnd(0))
    parts = []
    for _, g in lines[lines["group"].isin(OPEN_ITEM_GROUPS)].groupby("account"):
        m = match_account(g)
        parts.append(m[m["open"] != 0])
    if not parts:
        return pd.DataFrame()
    items = pd.concat(parts, ignore_index=True)
    items["age_days"] = (as_of - items["date"]).dt.days
    return items


def aging(items: pd.DataFrame, ledger: pd.DataFrame) -> list[dict]:
    """Aging buckets per customer / supplier account, in the account's natural sign."""
    if items.empty:
        return []
    rows = []
    for account, g in items[items["group"].isin((CUSTOMER, SUPPLIER))].groupby("account"):
        sign = -1 if g["kind"].iloc[0] in ("liability", "equity", "revenue") else 1
        buckets = {}
        for lo, hi, label in AGING_BUCKETS:
            sel = g[(g["age_days"] >= lo) & (g["age_days"] <= hi)]
            buckets[label] = round(float(sel["open"].sum() * sign), 2)
        rows.append({"account": account, "name": g["name"].iloc[0], "group": g["group"].iloc[0],
                     "total": round(float(g["open"].sum() * sign), 2), "buckets": buckets})
    return sorted(rows, key=lambda r: -abs(r["buckets"]["365+"]))


def _item_dict(row) -> dict:
    return {"date": row["date"].strftime("%d/%m/%Y"), "reference": row["reference"],
            "description": row["description"], "amount": round(float(row["open"]), 2),
            "age_days": int(row["age_days"])}


def find_old_items(items: pd.DataFrame, ledger: pd.DataFrame, min_days: int = 365,
                   min_amount: float = 2_000) -> list[Finding]:
    """One finding per account that holds lines open for more than `min_days`."""
    if items.empty:
        return []
    findings = []
    old = items[items["age_days"] > min_days]
    for account, g in old.groupby("account"):
        total = float(g["open"].sum())
        if abs(total) < min_amount:
            continue
        group, name = g["group"].iloc[0], g["name"].iloc[0]
        g = g.reindex(g["open"].abs().sort_values(ascending=False).index)
        oldest = g["date"].min()
        n = len(g)
        what = {CUSTOMER: L("חשבוניות לקוח", "customer invoices"), SUPPLIER: L("חשבוניות ספק", "supplier invoices"),
                ADVANCE: L("מקדמות", "advances"), CLEARING: L("רישומים", "entries"),
                PROVISION: L("רישומי הפרשה", "provision entries")}.get(group, L("רישומים", "entries"))
        count_text = L("פריט אחד", "1 item") if n == 1 else L(f"{n} פריטים", f"{n} items")
        top = g.iloc[0]
        top_text = f"{top['description'] or top['reference']} ({top['date'].strftime('%d/%m/%Y')}, {ils(abs(top['open']))})"
        if n == 1:
            reason = L(f"פריט אחד פתוח יותר משנה: {top_text}.", f"One item has been open for over a year: {top_text}.")
        else:
            reason = L(f"{n} פריטים ({what}) פתוחים יותר משנה, בסך {ils(abs(total))}. "
                       f"הוותיק ביותר מ-{oldest.strftime('%m/%Y')}. הגדול ביותר: {top_text}.",
                       f"{n} {what} have been open for over a year, {ils(abs(total))} in total. "
                       f"The oldest is from {oldest.strftime('%m/%Y')}. The largest: {top_text}.")
        check = {
            CUSTOMER: L("לבדוק מול הלקוח אם החשבונית שולמה ולא הותאמה, או שצריך לטפל בגבייה.",
                        "Check with the customer whether it was paid and not matched, or needs collecting."),
            SUPPLIER: L("לבדוק מול כרטסת הספק אם החשבונית שולמה ולא הותאמה, או שהיא שגויה.",
                        "Check against the supplier's statement whether it was paid and not matched, or is wrong."),
            ADVANCE: L("לבדוק מול כל ספק אם התקבלה חשבונית שלא נרשמה כנגד המקדמה.",
                       "Check with each supplier whether an invoice arrived that was never set against the advance."),
            CLEARING: L("לאתר לאיזה כרטיס היה אמור לעבור כל רישום ולהשלים את ההעברה.",
                        "Find the account each entry should have gone to and complete the transfer."),
            PROVISION: L("לבדוק עם הגורם המטפל אם ההפרשה עדיין נדרשת ובאיזה סכום.",
                         "Check with whoever owns it whether the provision is still needed, and at what amount."),
        }.get(group, L("לבדוק את הרישומים הפתוחים.", "Review the open entries."))
        acc_ledger = ledger[ledger["account"] == account].set_index("period")["balance"]
        sign = -1 if g["kind"].iloc[0] in ("liability", "equity", "revenue") else 1
        findings.append(Finding(
            section=HISTORICAL,
            kind="old_items",
            account=account,
            account_name=name,
            title=L(f"{name}: {ils(abs(total))} פתוחים מעל שנה ({count_text})",
                    f"{name}: {ils(abs(total))} open for over a year ({count_text})"),
            reason=reason,
            check=check,
            amount=abs(total),
            periods=sorted({period_label(d.to_period('M')) for d in g["date"]}),
            evidence=evidence(L("יתרת סוף חודש", "Month-end balance"), acc_ledger * sign,
                              highlight=sorted({d.to_period("M") for d in g["date"]})),
            details={"items": [_item_dict(r) for _, r in g.head(20).iterrows()], "count": n,
                     "oldest": oldest.strftime("%d/%m/%Y")},
        ))
    return findings
