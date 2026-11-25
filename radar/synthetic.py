"""A made-up company with 36 months of ledger lines (כרטסת) and planted issues.

Used for the demo and for tests, so the engine can be shown and checked
without any client's data. Everything here is fictional. The monthly trial
balance is derived from the lines, so both upload paths see the same books.

Planted issues (what the radar should find):
  historical
    2400 הפרשה לתביעה משפטית   150,000 opened 01/2024, never used or released
    1300 מקדמות לספקים          85,000 advance from 05/2024 never cleared
    2200 חשבון מעבר             23,400 left in 09/2024 and never cleared
    2200 חשבון מעבר             every December closes with a balance, cleared in January
    1200 לקוחות                 one invoice from 03/2025 never paid
    6600 עמלות בנק              an extra charge every March
  alerts (latest months)
    6300 פרסום ושיווק           four months of growth, heading through its monthly target
    6400 אחזקה ותיקונים         a one-month spike in the last month
    5000 עלות המכר              share of revenue jumped from ~60% to ~68%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPANY_NAME = "חברה לדוגמה בע\"מ"
START = pd.Period("2023-11", freq="M")
MONTHS = 36
DEMO_TARGETS = {"6300": 90_000.0}

ACCOUNTS = {
    "1100": "בנק הפועלים - עו\"ש",
    "1200": "לקוחות",
    "1300": "מקדמות לספקים",
    "2100": "ספקים",
    "2200": "חשבון מעבר",
    "2300": "הפרשה לחופשה",
    "2400": "הפרשה לתביעה משפטית",
    "3000": "הון מניות",
    "4000": "הכנסות ממכירות",
    "5000": "עלות המכר",
    "6100": "הוצאות שכר",
    "6110": "הפרשות סוציאליות",
    "6200": "שכירות",
    "6300": "פרסום ושיווק",
    "6400": "אחזקה ותיקונים",
    "6500": "ביטוח",
    "6600": "עמלות בנק",
}
CUSTOMERS = ["אורן תעשיות בע\"מ", "גל-טק מערכות בע\"מ", "שקד מזון בע\"מ", "נתיב לוגיסטיקה בע\"מ"]
SUPPLIERS = ["אלומיניום הצפון בע\"מ", "פלסט-אור בע\"מ", "מתכות הדרום בע\"מ"]


def _day(p: pd.Period, day: int) -> pd.Timestamp:
    return pd.Timestamp(year=p.year, month=p.month, day=min(day, p.days_in_month))


def demo_transactions(seed: int = 7) -> pd.DataFrame:
    """Ledger lines in the layout of a כרטסת export, with Hebrew headers."""
    rng = np.random.default_rng(seed)
    periods = pd.period_range(START, periods=MONTHS, freq="M")
    end = periods[-1].end_time.normalize()
    t = np.arange(MONTHS)
    months = np.array([p.month for p in periods])
    years = np.array([p.year for p in periods])
    noise = lambda sd: 1 + rng.normal(0, sd, MONTHS)  # noqa: E731
    lines: list[dict] = []

    def add(account, date, amount, ref="", desc=""):
        if date <= end and abs(amount) >= 0.5:
            lines.append({"account": account, "date": date, "reference": ref, "description": desc,
                          "amount": round(float(amount), 0)})

    # P&L, natural sign.
    revenue = 1_200_000 * 1.01 ** t * (1 + 0.04 * np.sin(2 * np.pi * months / 12)) * noise(0.03)
    cogs_ratio = np.full(MONTHS, 0.60)
    cogs_ratio[-3:] = [0.67, 0.68, 0.69]
    cogs = revenue * cogs_ratio * noise(0.015)
    salary = 280_000 * 1.04 ** (years - years[0]) * noise(0.01)
    social = salary * 0.22 * noise(0.01)
    rent = np.where(periods >= pd.Period("2025-01", freq="M"), 47_000.0, 45_000.0)
    marketing = 60_000 * noise(0.04)
    marketing[-5:] = [59_000, 61_200, 66_500, 72_300, 79_100]
    maintenance = 15_000 * noise(0.06)
    maintenance[-1] = 61_500
    insurance = np.full(MONTHS, 8_000.0)
    bank_fees = 3_000 * noise(0.05) + np.where(months == 3, 12_000, 0)
    vac_accrued, vac_used = 12_000 * noise(0.05), 10_500 * noise(0.2)

    inv_no = 10_000
    for i, p in enumerate(periods):
        tag = f"{p.month:02d}/{p.year}"
        last = _day(p, 31)
        # Sales invoices: revenue credited, customers debited; paid 45–75 days later.
        shares = rng.dirichlet(np.ones(len(CUSTOMERS)) * 4)
        for k, (cust, share) in enumerate(zip(CUSTOMERS, shares)):
            inv_no += 1
            amount = round(revenue[i] * share, 0)
            date = _day(p, 5 + 7 * k)
            ref = f"INV-{inv_no}"
            add("4000", date, -amount, ref, f"חשבונית {inv_no} - {cust}")
            add("1200", date, amount, ref, f"חשבונית {inv_no} - {cust}")
            add("1200", date + pd.Timedelta(days=int(rng.integers(55, 66))), -amount, ref, f"קבלה - {cust}")
        # Supplier invoices for cost of sales; paid about 30–45 days later.
        for k, (sup, share) in enumerate(zip(SUPPLIERS, rng.dirichlet(np.ones(len(SUPPLIERS)) * 4))):
            amount = round(cogs[i] * share, 0)
            date = _day(p, 8 + 8 * k)
            ref = f"SI-{p.year}{p.month:02d}-{k + 1}"
            add("5000", date, amount, ref, f"חשבונית ספק - {sup}")
            add("2100", date, -amount, ref, f"חשבונית ספק - {sup}")
            add("2100", date + pd.Timedelta(days=int(rng.integers(30, 46))), amount, ref, f"תשלום - {sup}")
        # Monthly expenses posted at month end.
        for code, values, desc in [
            ("6100", salary, "שכר חודש"), ("6110", social, "הפרשות סוציאליות"), ("6200", rent, "שכירות"),
            ("6300", marketing, "פרסום ושיווק"), ("6400", maintenance, "אחזקה ותיקונים"),
            ("6500", insurance, "ביטוח"), ("6600", bank_fees, "עמלות בנק"),
        ]:
            add(code, last, values[i], f"JV-{p.year}{p.month:02d}", f"{desc} {tag}")
        add("2300", last, -vac_accrued[i], f"JV-{p.year}{p.month:02d}", f"צבירת חופשה {tag}")
        add("2300", last, vac_used[i], f"JV-{p.year}{p.month:02d}", f"ניצול חופשה {tag}")
        # Advances to suppliers: paid in some months, offset against the invoice next month.
        if rng.random() < 0.55:
            sup = SUPPLIERS[int(rng.integers(len(SUPPLIERS)))]
            amount = round(rng.uniform(20_000, 40_000), 0)
            ref = f"ADV-{p.year}{p.month:02d}"
            add("1300", _day(p, 10), amount, ref, f"מקדמה - {sup}")
            add("1300", _day(p + 1, 20), -amount, ref, f"קיזוז מקדמה מול חשבונית - {sup}")

    # Planted items.
    add("4000", pd.Timestamp("2025-03-17"), -42_120, "INV-25031", "חשבונית 25031 - גל-טק מערכות בע\"מ (שירות)")
    add("1200", pd.Timestamp("2025-03-17"), 42_120, "INV-25031", "חשבונית 25031 - גל-טק מערכות בע\"מ (שירות)")
    add("1300", pd.Timestamp("2024-05-14"), 85_000, "ADV-202405-B", "מקדמה - אלומיניום הצפון בע\"מ (הזמנה 4471)")
    for year, amount in zip((2023, 2024, 2025), (34_200, 37_100, 35_800)):
        add("2200", pd.Timestamp(f"{year}-12-31"), amount, f"CLR-{year}", "הפרשי סגירה שנתית - לבירור")
        add("2200", pd.Timestamp(f"{year + 1}-01-15"), -amount, f"CLR-{year}", "סגירת הפרשי סגירה שנתית")
    add("2200", pd.Timestamp("2024-09-26"), 23_400, "CLR-0924", "תשלום לא מזוהה - לבירור")
    add("2400", pd.Timestamp("2024-01-31"), -150_000, "PRV-2401", "הפרשה לתביעת ספק")

    # Opening balances.
    first = _day(periods[0], 1)
    add("1100", first, 400_000, "OPEN", "יתרת פתיחה")
    add("2300", first, -180_000, "OPEN", "יתרת פתיחה")
    add("3000", first, -100_000, "OPEN", "יתרת פתיחה")
    # Bank: the other side of everything, month by month, plus small differences.
    df = pd.DataFrame(lines)
    df["period"] = df["date"].dt.to_period("M")
    others = df[df["account"] != "1100"].groupby("period")["amount"].sum().reindex(periods, fill_value=0)
    for p, total in others.items():
        add("1100", _day(p, 31), -total + rng.normal(0, 5_000), f"BNK-{p.year}{p.month:02d}", "תנועות בנק החודש")

    df = pd.DataFrame(lines).sort_values(["account", "date"], kind="stable")
    return pd.DataFrame({
        "כרטיס": df["account"],
        "שם כרטיס": df["account"].map(ACCOUNTS),
        "תאריך": df["date"].dt.strftime("%d/%m/%Y"),
        "אסמכתא": df["reference"],
        "פרטים": df["description"],
        "חובה": df["amount"].clip(lower=0),
        "זכות": (-df["amount"]).clip(lower=0),
    }).reset_index(drop=True)


def demo_trial_balance(seed: int = 7) -> pd.DataFrame:
    """The monthly trial balance the demo lines add up to (long layout, Hebrew headers)."""
    from .transactions import lines_to_trial_balance, normalize_transactions

    result = normalize_transactions(demo_transactions(seed))
    return lines_to_trial_balance(result.lines, result.ledger)


__all__ = ["COMPANY_NAME", "DEMO_TARGETS", "demo_transactions", "demo_trial_balance"]
