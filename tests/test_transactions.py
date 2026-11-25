import pandas as pd
import pytest

from radar.analysis.open_items import match_account
from radar.engine import Settings, analyze
from radar.synthetic import DEMO_TARGETS, demo_transactions
from radar.transactions import is_transactions, normalize_transactions


def lines(rows):
    df = pd.DataFrame(rows, columns=["date", "reference", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["description"] = ""
    return df


def test_match_by_reference_then_amount_then_fifo():
    m = match_account(lines([
        ("2024-01-05", "A", 100), ("2024-01-10", "B", 250), ("2024-02-01", "C", 70),
        ("2024-02-10", "A", -100),   # same reference
        ("2024-02-15", "X", -250),   # same amount
        ("2024-03-01", "Y", -50),    # FIFO: eats 50 of C
    ]))
    open_ = m[m["open"] != 0]
    assert list(open_["reference"]) == ["C"]
    assert open_["open"].iloc[0] == pytest.approx(20)


@pytest.fixture(scope="module")
def result():
    return normalize_transactions(demo_transactions())


def test_detects_ledger_lines():
    assert is_transactions(demo_transactions())


def test_monthly_ledger_from_lines(result):
    ledger = result.ledger
    assert ledger["period"].nunique() == 36
    adv = ledger[ledger["account"] == "1300"].set_index("period")["balance"]
    assert adv.min() >= 0


def test_open_items_name_the_stuck_lines(result):
    rep = analyze(result.ledger, Settings(targets=DEMO_TARGETS), lines=result.lines)
    old = {f["account"]: f for f in rep["historical"] if f["kind"] == "old_items"}
    assert set(old) == {"1200", "1300", "2200", "2400"}
    assert old["1300"]["details"]["items"][0]["reference"] == "ADV-202405-B"
    assert old["1300"]["amount"] == pytest.approx(85_000)
    assert old["1200"]["details"]["items"][0]["reference"] == "INV-25031"
    assert old["2200"]["amount"] == pytest.approx(23_400)
    # Balance-level estimates are replaced by the line-level ones.
    assert not [f for f in rep["historical"] if f["kind"] == "stale_balance"]


def test_aging_and_review(result):
    rep = analyze(result.ledger, Settings(targets=DEMO_TARGETS), lines=result.lines)
    customers = next(a for a in rep["aging"] if a["account"] == "1200")
    assert customers["buckets"]["365+"] == pytest.approx(42_120)
    status = {r["account"]: r["status"] for r in rep["review"]["rows"]}
    assert status["1100"] == "ok" and status["2300"] == "ok" and status["3000"] == "ok"
    assert all(status[a] == "issue" for a in ("1200", "1300", "2200", "2400"))
