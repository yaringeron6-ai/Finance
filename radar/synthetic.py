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
COMPANY_NAME_EN = "Example Co. Ltd."
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
ACCOUNTS_EN = {
    "1100": "Bank - current account", "1200": "Customers", "1300": "Advances to suppliers",
    "2100": "Suppliers", "2200": "Clearing account", "2300": "Vacation accrual",
    "2400": "Legal claim provision", "3000": "Share capital", "4000": "Sales revenue",
    "5000": "Cost of sales", "6100": "Salaries", "6110": "Social benefits", "6200": "Rent",
    "6300": "Advertising and marketing", "6400": "Maintenance and repairs", "6500": "Insurance",
    "6600": "Bank fees",
}
CUSTOMERS_EN = ["Oren Industries Ltd.", "Gal-Tech Systems Ltd.", "Shaked Foods Ltd.", "Nativ Logistics Ltd."]
SUPPLIERS_EN = ["North Aluminium Ltd.", "Plast-Or Ltd.", "South Metals Ltd."]
CUSTOMERS = ["אורן תעשיות בע\"מ", "גל-טק מערכות בע\"מ", "שקד מזון בע\"מ", "נתיב לוגיסטיקה בע\"מ"]
SUPPLIERS = ["אלומיניום הצפון בע\"מ", "פלסט-אור בע\"מ", "מתכות הדרום בע\"מ"]


def _day(p: pd.Period, day: int) -> pd.Timestamp:
    return pd.Timestamp(year=p.year, month=p.month, day=min(day, p.days_in_month))


def demo_transactions(seed: int = 7, lang: str = "he") -> pd.DataFrame:
    """Ledger lines in the layout of a כרטסת export (Hebrew headers, or English with lang="en")."""
    en = lang == "en"
    T = (lambda he, e: e) if en else (lambda he, e: he)  # noqa: E731
    customers, suppliers = (CUSTOMERS_EN, SUPPLIERS_EN) if en else (CUSTOMERS, SUPPLIERS)
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
        shares = rng.dirichlet(np.ones(len(customers)) * 4)
        for k, (cust, share) in enumerate(zip(customers, shares)):
            inv_no += 1
            amount = round(revenue[i] * share, 0)
            date = _day(p, 5 + 7 * k)
            ref = f"INV-{inv_no}"
            add("4000", date, -amount, ref, T(f"חשבונית {inv_no} - {cust}", f"Invoice {inv_no} - {cust}"))
            add("1200", date, amount, ref, T(f"חשבונית {inv_no} - {cust}", f"Invoice {inv_no} - {cust}"))
            add("1200", date + pd.Timedelta(days=int(rng.integers(55, 66))), -amount, ref, T(f"קבלה - {cust}", f"Receipt - {cust}"))
        # Supplier invoices for cost of sales; paid about 30–45 days later.
        for k, (sup, share) in enumerate(zip(suppliers, rng.dirichlet(np.ones(len(suppliers)) * 4))):
            amount = round(cogs[i] * share, 0)
            date = _day(p, 8 + 8 * k)
            ref = f"SI-{p.year}{p.month:02d}-{k + 1}"
            add("5000", date, amount, ref, T(f"חשבונית ספק - {sup}", f"Supplier invoice - {sup}"))
            add("2100", date, -amount, ref, T(f"חשבונית ספק - {sup}", f"Supplier invoice - {sup}"))
            add("2100", date + pd.Timedelta(days=int(rng.integers(30, 46))), amount, ref, T(f"תשלום - {sup}", f"Payment - {sup}"))
        # Monthly expenses posted at month end.
        for code, values, desc in [
            ("6100", salary, T("שכר חודש", "Payroll")), ("6110", social, T("הפרשות סוציאליות", "Social benefits")),
            ("6200", rent, T("שכירות", "Rent")), ("6300", marketing, T("פרסום ושיווק", "Marketing")),
            ("6400", maintenance, T("אחזקה ותיקונים", "Maintenance")),
            ("6500", insurance, T("ביטוח", "Insurance")), ("6600", bank_fees, T("עמלות בנק", "Bank fees")),
        ]:
            add(code, last, values[i], f"JV-{p.year}{p.month:02d}", f"{desc} {tag}")
        add("2300", last, -vac_accrued[i], f"JV-{p.year}{p.month:02d}", T(f"צבירת חופשה {tag}", f"Vacation accrued {tag}"))
        add("2300", last, vac_used[i], f"JV-{p.year}{p.month:02d}", T(f"ניצול חופשה {tag}", f"Vacation used {tag}"))
        # Advances to suppliers: paid in some months, offset against the invoice next month.
        if rng.random() < 0.55:
            sup = suppliers[int(rng.integers(len(suppliers)))]
            amount = round(rng.uniform(20_000, 40_000), 0)
            ref = f"ADV-{p.year}{p.month:02d}"
            add("1300", _day(p, 10), amount, ref, T(f"מקדמה - {sup}", f"Advance - {sup}"))
            add("1300", _day(p + 1, 20), -amount, ref, T(f"קיזוז מקדמה מול חשבונית - {sup}", f"Advance set against invoice - {sup}"))

    # Planted items.
    add("4000", pd.Timestamp("2025-03-17"), -42_120, "INV-25031", T("חשבונית 25031 - גל-טק מערכות בע\"מ (שירות)", "Invoice 25031 - Gal-Tech Systems Ltd. (service)"))
    add("1200", pd.Timestamp("2025-03-17"), 42_120, "INV-25031", T("חשבונית 25031 - גל-טק מערכות בע\"מ (שירות)", "Invoice 25031 - Gal-Tech Systems Ltd. (service)"))
    add("1300", pd.Timestamp("2024-05-14"), 85_000, "ADV-202405-B",
        T("מקדמה - אלומיניום הצפון בע\"מ (הזמנה 4471)", "Advance - North Aluminium Ltd. (order 4471)"))
    for year, amount in zip((2023, 2024, 2025), (34_200, 37_100, 35_800)):
        add("2200", pd.Timestamp(f"{year}-12-31"), amount, f"CLR-{year}", T("הפרשי סגירה שנתית - לבירור", "Year-end closing differences - to clarify"))
        add("2200", pd.Timestamp(f"{year + 1}-01-15"), -amount, f"CLR-{year}", T("סגירת הפרשי סגירה שנתית", "Clearing year-end differences"))
    add("2200", pd.Timestamp("2024-09-26"), 23_400, "CLR-0924", T("תשלום לא מזוהה - לבירור", "Unidentified payment - to clarify"))
    add("2400", pd.Timestamp("2024-01-31"), -150_000, "PRV-2401", T("הפרשה לתביעת ספק", "Provision for a supplier claim"))

    # Opening balances.
    first = _day(periods[0], 1)
    opening = T("יתרת פתיחה", "Opening balance")
    add("1100", first, 400_000, "OPEN", opening)
    add("2300", first, -180_000, "OPEN", opening)
    add("3000", first, -100_000, "OPEN", opening)
    # Bank: the other side of everything, month by month, plus small differences.
    df = pd.DataFrame(lines)
    df["period"] = df["date"].dt.to_period("M")
    others = df[df["account"] != "1100"].groupby("period")["amount"].sum().reindex(periods, fill_value=0)
    for p, total in others.items():
        add("1100", _day(p, 31), -total + rng.normal(0, 5_000), f"BNK-{p.year}{p.month:02d}", T("תנועות בנק החודש", "Bank movements for the month"))

    df = pd.DataFrame(lines).sort_values(["account", "date"], kind="stable")
    return pd.DataFrame({
        T("כרטיס", "Account"): df["account"],
        T("שם כרטיס", "Account name"): df["account"].map(ACCOUNTS_EN if en else ACCOUNTS),
        T("תאריך", "Date"): df["date"].dt.strftime("%d/%m/%Y"),
        T("אסמכתא", "Reference"): df["reference"],
        T("פרטים", "Description"): df["description"],
        T("חובה", "Debit"): df["amount"].clip(lower=0),
        T("זכות", "Credit"): (-df["amount"]).clip(lower=0),
    }).reset_index(drop=True)


def demo_trial_balance(seed: int = 7, lang: str = "he") -> pd.DataFrame:
    """The monthly trial balance the demo lines add up to (long layout, Hebrew headers)."""
    from .transactions import lines_to_trial_balance, normalize_transactions

    result = normalize_transactions(demo_transactions(seed, lang))
    return lines_to_trial_balance(result.lines, result.ledger)


__all__ = ["COMPANY_NAME", "COMPANY_NAME_EN", "DEMO_TARGETS", "demo_transactions", "demo_trial_balance"]
