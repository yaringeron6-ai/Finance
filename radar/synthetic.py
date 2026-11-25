"""A made-up company with 36 months of trial balances and planted issues.

Used for the demo and for tests, so the engine can be shown and checked
without any client's data. Everything here is fictional.

Planted issues (what the radar should find):
  historical
    2400 הפרשה לתביעה משפטית   150,000 opened 01/2024, never used or released
    1300 מקדמות לספקים          85,000 advance from 05/2024 never cleared
    2200 חשבון מעבר             23,400 left in 09/2024 and never cleared
    2200 חשבון מעבר             every December closes with a balance, cleared in January
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

ACCOUNTS = [
    # code, name, kind
    ("1100", "בנק הפועלים - עו\"ש", "asset"),
    ("1200", "לקוחות", "asset"),
    ("1300", "מקדמות לספקים", "asset"),
    ("2100", "ספקים", "liability"),
    ("2200", "חשבון מעבר", "asset"),
    ("2300", "הפרשה לחופשה", "liability"),
    ("2400", "הפרשה לתביעה משפטית", "liability"),
    ("3000", "הון מניות", "equity"),
    ("4000", "הכנסות ממכירות", "revenue"),
    ("5000", "עלות המכר", "cogs"),
    ("6100", "הוצאות שכר", "expense"),
    ("6110", "הפרשות סוציאליות", "expense"),
    ("6200", "שכירות", "expense"),
    ("6300", "פרסום ושיווק", "expense"),
    ("6400", "אחזקה ותיקונים", "expense"),
    ("6500", "ביטוח", "expense"),
    ("6600", "עמלות בנק", "expense"),
]


def _movements(seed: int) -> tuple[pd.PeriodIndex, dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    periods = pd.period_range(START, periods=MONTHS, freq="M")
    t = np.arange(MONTHS)
    months = np.array([p.month for p in periods])
    years = np.array([p.year for p in periods])
    noise = lambda sd: 1 + rng.normal(0, sd, MONTHS)  # noqa: E731
    mv: dict[str, np.ndarray] = {}
    opening: dict[str, float] = {}

    # P&L, amounts in natural sign (positive), converted to debit-positive below.
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

    mv["4000"] = -revenue
    mv["5000"] = cogs
    mv["6100"] = salary
    mv["6110"] = social
    mv["6200"] = rent
    mv["6300"] = marketing
    mv["6400"] = maintenance
    mv["6500"] = insurance
    mv["6600"] = bank_fees

    # Balance sheet, debit positive.
    customers = 1.4 * revenue
    opening["1200"] = customers[0] * 0.98
    mv["1200"] = np.diff(np.concatenate([[opening["1200"]], customers]))
    suppliers = 0.9 * cogs
    opening["2100"] = -suppliers[0] * 0.98
    mv["2100"] = -np.diff(np.concatenate([[-opening["2100"]], suppliers]))

    # Advances: paid in some months, cleared against the invoice next month.
    paid = np.where(rng.random(MONTHS) < 0.55, rng.uniform(20_000, 40_000, MONTHS), 0.0)
    advances = paid.copy()
    stuck_from = periods.get_loc(pd.Period("2024-05", freq="M"))
    advances[stuck_from:] += 85_000
    opening["1300"] = 0.0
    mv["1300"] = np.diff(np.concatenate([[0.0], advances]))

    # Clearing account: should be zero at every month end.
    clearing = np.zeros(MONTHS)
    for year, amount in zip((2023, 2024, 2025), (34_200, 37_100, 35_800)):
        clearing[(years == year) & (months == 12)] += amount
    clearing[periods >= pd.Period("2024-09", freq="M")] += 23_400
    opening["2200"] = 0.0
    mv["2200"] = np.diff(np.concatenate([[0.0], clearing]))

    # Vacation accrual: builds up and is used, expected to stay open.
    opening["2300"] = -180_000.0
    mv["2300"] = -(12_000 * noise(0.05)) + 10_500 * noise(0.2)

    # One-off legal provision, never used or released.
    legal = np.zeros(MONTHS)
    legal[periods.get_loc(pd.Period("2024-01", freq="M"))] = -150_000
    opening["2400"] = 0.0
    mv["2400"] = legal

    opening["3000"] = -100_000.0
    mv["3000"] = np.zeros(MONTHS)

    # Bank takes whatever keeps the books roughly balanced, plus noise.
    others = sum(mv.values())
    opening["1100"] = 400_000.0
    mv["1100"] = -others + rng.normal(0, 5_000, MONTHS)

    return periods, mv, opening


def demo_trial_balance(seed: int = 7) -> pd.DataFrame:
    """Monthly trial balance in the long layout with Hebrew headers, like an export."""
    periods, mv, opening = _movements(seed)
    years = np.array([p.year for p in periods])
    rows = []
    for code, name, kind in ACCOUNTS:
        movement = np.round(mv[code], 0)
        if kind in ("revenue", "cogs", "expense"):
            balance = pd.Series(movement).groupby(years).cumsum().to_numpy()  # year to date
        else:
            balance = opening.get(code, 0.0) + np.cumsum(movement)
        for p, m, b in zip(periods, movement, balance):
            rows.append({
                "כרטיס": code,
                "שם כרטיס": name,
                "חודש": f"{p.month:02d}/{p.year}",
                "חובה": max(m, 0.0),
                "זכות": max(-m, 0.0),
                "יתרה": round(float(b), 0),
            })
    return pd.DataFrame(rows)


__all__ = ["COMPANY_NAME", "DEMO_TARGETS", "demo_trial_balance"]
