import pandas as pd
import pytest

from radar.engine import Settings, analyze
from radar.normalize import normalize
from radar.synthetic import DEMO_TARGETS, demo_trial_balance


@pytest.fixture(scope="module")
def report():
    ledger = normalize(demo_trial_balance()).ledger
    return analyze(ledger, Settings(targets=DEMO_TARGETS))


def found(report, section, kind):
    return {f["account"]: f for f in report[section] if f["kind"] == kind}


def test_stale_balances(report):
    stale = found(report, "historical", "stale_balance")
    assert set(stale) == {"2400", "1300", "2200"}
    assert stale["2400"]["amount"] == pytest.approx(150_000, abs=5)
    assert stale["2400"]["details"]["since"] == "01/2024"
    assert stale["1300"]["amount"] == pytest.approx(85_000, abs=50)
    assert stale["1300"]["details"]["since"] == "05/2024"
    assert stale["2200"]["amount"] == pytest.approx(23_400, abs=5)
    assert stale["2200"]["details"]["since"] == "09/2024"


def test_recurring_patterns(report):
    recurring = found(report, "historical", "recurring")
    assert set(recurring) == {"2200", "6600"}
    assert recurring["2200"]["details"]["months"] == [12]
    assert recurring["6600"]["details"]["months"] == [3]


def test_alerts(report):
    alerts = {f["account"]: f["kind"] for f in report["alerts"]}
    assert alerts == {"5000": "relation_break", "6300": "trend", "6400": "anomaly"}
    trend = next(f for f in report["alerts"] if f["account"] == "6300")
    assert trend["details"]["months_to_target"] <= 3


def test_relationships_found(report):
    pairs = {(r["driver"], r["account"]) for r in report["relationships"]}
    assert ("4000", "5000") in pairs
    assert ("6100", "6110") in pairs


def test_sorted_by_amount(report):
    for section in ("historical", "alerts"):
        amounts = [f["amount"] for f in report[section]]
        assert amounts == sorted(amounts, reverse=True)


def test_every_finding_is_marked_for_review(report):
    for f in report["historical"] + report["alerts"]:
        assert f["status"] == "לבדיקה"
        assert f["check"] and f["reason"]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_no_false_alarms_on_healthy_accounts(seed):
    """Across random seeds, the healthy accounts never get flagged."""
    healthy = {"1100", "1200", "2100", "2300", "3000", "4000", "6100", "6110", "6200", "6500"}
    rep = analyze(normalize(demo_trial_balance(seed)).ledger, Settings(targets=DEMO_TARGETS))
    flagged = {f["account"] for f in rep["historical"] + rep["alerts"]}
    assert not flagged & healthy, flagged & healthy
