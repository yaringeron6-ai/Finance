import io

import pandas as pd
import pytest

from radar.connectors import FileConnector
from radar.normalize import InputError, normalize, parse_period, to_number


@pytest.mark.parametrize("value, expected", [
    ("01/2024", "2024-01"), ("2024-01", "2024-01"), ("31/01/2024", "2024-01"),
    ("2024-01-31 00:00:00", "2024-01"), ("1/24", "2024-01"), (pd.Timestamp("2024-03-15"), "2024-03"),
])
def test_parse_period(value, expected):
    assert str(parse_period(value)) == expected


@pytest.mark.parametrize("value, expected", [
    ("1,234.50", 1234.5), ("(500)", -500), ("500-", -500), ("₪ 20", 20), (7, 7),
])
def test_to_number(value, expected):
    assert to_number(value) == expected


def test_wide_layout_with_hebrew_headers():
    months = [f"{m:02d}/2024" for m in range(1, 9)]
    raw = pd.DataFrame([
        ["6300", "פרסום ושיווק"] + [1000] * 8,
        ["4000", "הכנסות ממכירות"] + [-5000] * 8,
    ], columns=["מס' כרטיס", "שם חשבון"] + months)
    ledger = normalize(raw).ledger
    assert len(ledger) == 16
    assert set(ledger["kind"]) == {"expense", "revenue"}
    rev = ledger[ledger["account"] == "4000"]
    assert rev["balance"].iloc[-1] == -40_000  # year to date


def test_balance_only_derives_movement():
    rows = [{"account": "2200", "name": "חשבון מעבר", "month": f"2024-{m:02d}", "balance": b}
            for m, b in zip(range(1, 8), [0, 100, 100, 0, 0, 50, 50])]
    ledger = normalize(pd.DataFrame(rows)).ledger
    assert list(ledger["movement"]) == [0, 100, 0, -100, 0, 50, 0]
    assert ledger["group"].iloc[0] == "clearing"


def test_missing_account_column_is_explained():
    with pytest.raises(InputError, match="כרטיס"):
        normalize(pd.DataFrame({"x": [1], "month": ["2024-01"]}))


def test_excel_upload_round_trip():
    from radar.synthetic import demo_trial_balance

    buf = io.BytesIO()
    demo_trial_balance().to_excel(buf, index=False)
    result = FileConnector(buf.getvalue(), "tb.xlsx").load()
    assert result.ledger["account"].nunique() == 17
    assert result.ledger["period"].nunique() == 36
