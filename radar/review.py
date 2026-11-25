"""Balance-sheet review: every balance-sheet account, months side by side, with a status.

The controller's monthly file today is a sheet of balances month by month. This
builds the same sheet and fills in the first pass of the review:

    ok       nothing to look at
    explain  something changed or looks odd and needs a one-line explanation
    issue    an item is stuck or the balance cannot be right as it stands

Only `explain` and `issue` rows need the controller's attention.
"""

from __future__ import annotations

import io

import pandas as pd

from .analysis.common import ils, period_label
from .i18n import L, lang
from .model import ACCRUAL, BANK, CLEARING, EQUITY, PNL_KINDS, natural_sign

OK, EXPLAIN, ISSUE = "ok", "explain", "issue"
def status_label(status: str) -> str:
    return {OK: L("תקין", "OK"), EXPLAIN: L("נדרש הסבר", "Explain"), ISSUE: L("לטיפול", "Action")}[status]
_RANK = {OK: 0, EXPLAIN: 1, ISSUE: 2}


def build_review(ledger: pd.DataFrame, findings: list[dict], months: int = 6,
                 flux_amount: float = 50_000, flux_share: float = 0.2, min_amount: float = 2_000) -> dict:
    bs = ledger[~ledger["kind"].isin(PNL_KINDS)]
    periods = sorted(bs["period"].unique())
    shown = periods[-months:]
    by_account: dict[str, list[dict]] = {}
    for f in findings:
        by_account.setdefault(f["account"], []).append(f)

    rows = []
    for account, g in bs.groupby("account", sort=True):
        g = g.set_index("period").sort_index()
        kind, group, name = g["kind"].iloc[0], g["group"].iloc[0], g["name"].iloc[0]
        sign = natural_sign(kind)
        bal = g["balance"] * sign
        last, prev = float(bal.iloc[-1]), float(bal.iloc[-2]) if len(bal) > 1 else 0.0
        change = last - prev
        status, notes = OK, []

        def flag(level: str, note: str):
            nonlocal status
            notes.append(note)
            if _RANK[level] > _RANK[status]:
                status = level

        for f in by_account.get(account, []):
            if f["kind"] in ("stale_balance", "old_items"):
                flag(ISSUE, f["title"].split(": ", 1)[-1])
            elif f["kind"] == "recurring":
                flag(EXPLAIN, f["title"].split(": ", 1)[-1])
        if group == CLEARING and abs(last) >= min_amount:
            flag(EXPLAIN, L(f"חשבון מעבר עם יתרה של {ils(last)} בסוף החודש",
                             f"Clearing account holds {ils(last)} at month end"))
        if last <= -min_amount and group != CLEARING:
            flag(EXPLAIN, L(f"יתרה בסימן הפוך ({'זכות' if sign > 0 else 'חובה'})",
                             f"Balance on the wrong side ({'credit' if sign > 0 else 'debit'})"))
        if kind == EQUITY and abs(change) >= min_amount:
            flag(EXPLAIN, L(f"תנועה בהון של {ils(change)} החודש", f"Equity moved by {ils(change)} this month"))
        elif group not in (BANK, ACCRUAL) and abs(change) >= max(flux_amount, flux_share * abs(prev)):
            share = f" ({change / prev * 100:+.0f}%)" if prev else ""
            flag(EXPLAIN, L(f"שינוי של {ils(change)}{share} מהחודש הקודם", f"Changed by {ils(change)}{share} from last month"))
        if group == BANK:
            notes.append(L("להתאים מול דף הבנק", "Reconcile with the bank statement"))

        rows.append({
            "account": account, "name": name, "kind": kind, "group": group,
            "balances": {period_label(p): round(float(bal.get(p, 0.0)), 2) for p in shown},
            "change": round(change, 2), "status": status, "status_label": status_label(status),
            "notes": notes,
        })
    rows.sort(key=lambda r: (-_RANK[r["status"]], r["account"]))
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in (OK, EXPLAIN, ISSUE)}
    return {"periods": [period_label(p) for p in shown], "rows": rows, "counts": counts}


def review_to_xlsx(ledger: pd.DataFrame, review: dict, company: str | None = None) -> bytes:
    """The review as an Excel file: all months side by side, status, notes, and empty sign-off columns."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    bs = ledger[~ledger["kind"].isin(PNL_KINDS)]
    periods = sorted(bs["period"].unique())
    wb = Workbook()
    ws = wb.active
    ws.title = L("סקירת מאזן", "Balance sheet review")
    ws.sheet_view.rightToLeft = lang() == "he"
    title = L("סקירת מאזן", "Balance sheet review") + f" {period_label(periods[-1])}" + (f" · {company}" if company else "")
    ws.append([title])
    ws["A1"].font = Font(bold=True, size=14)
    header = [L("כרטיס", "Account"), L("שם כרטיס", "Name")] + [period_label(p) for p in periods] + \
             [L("שינוי מהחודש הקודם", "Change from last month"), L("סטטוס", "Status"), L("ממצאים", "Findings"),
              L("הסבר", "Explanation"), L("נבדק על ידי", "Reviewed by")]
    ws.append([])
    ws.append(header)
    for cell in ws[3]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="EDEAE3")
    fills = {OK: "E3F1E7", EXPLAIN: "FFF2CC", ISSUE: "F8D7CC"}
    for r in review["rows"]:
        g = bs[bs["account"] == r["account"]].set_index("period")["balance"] * natural_sign(r["kind"])
        ws.append([r["account"], r["name"]] + [round(float(g.get(p, 0.0)), 0) for p in periods] +
                  [round(r["change"], 0), r["status_label"], "; ".join(r["notes"]), "", ""])
        ws.cell(ws.max_row, len(periods) + 4).fill = PatternFill("solid", fgColor=fills[r["status"]])
    for col in range(3, len(periods) + 4):
        for row in ws.iter_rows(min_row=4, min_col=col, max_col=col):
            row[0].number_format = "#,##0;-#,##0"
    widths = {1: 10, 2: 26, len(periods) + 3: 16, len(periods) + 4: 12, len(periods) + 5: 50,
              len(periods) + 6: 30, len(periods) + 7: 14}
    for i in range(1, len(header) + 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(i, 12)
    for row in ws.iter_rows(min_row=4, min_col=len(periods) + 5, max_col=len(periods) + 5):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = ws.cell(4, 3)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
