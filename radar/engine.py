"""Run every check on a ledger and assemble the report the web app shows."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .analysis.anomalies import find_anomalies
from .analysis.common import period_label
from .analysis.open_items import aging, find_old_items, open_items
from .analysis.recurring import find_recurring
from .analysis.relations import find_relations
from .analysis.stale import find_stale
from .analysis.trends import find_trends
from .i18n import use_lang
from .model import OPEN_ITEM_GROUPS, Finding
from .review import build_review

# When one account triggers several alerts, keep the one that explains the most.
ALERT_PRIORITY = {"relation_break": 0, "trend": 1, "anomaly": 2}


@dataclass
class Settings:
    min_amount: float = 2_000       # ignore anything smaller, in ILS
    stale_months: int = 12          # "open for more than a year"
    recent_months: int = 3          # what counts as "recent" for alerts
    targets: dict[str, float] = field(default_factory=dict)  # account -> monthly target, natural sign


def _dedupe_alerts(alerts: list[Finding]) -> list[Finding]:
    best: dict[str, Finding] = {}
    for f in alerts:
        current = best.get(f.account)
        if current is None or ALERT_PRIORITY[f.kind] < ALERT_PRIORITY[current.kind]:
            best[f.account] = f
    return list(best.values())


def analyze(ledger: pd.DataFrame, settings: Settings | None = None, company: str | None = None,
            warnings: list[str] | None = None, lines: pd.DataFrame | None = None, lang: str = "he") -> dict:
    with use_lang(lang):
        report = _analyze(ledger, settings or Settings(), company, warnings, lines)
    report["lang"] = lang
    return report


def _analyze(ledger, s: Settings, company, warnings, lines) -> dict:
    stale = find_stale(ledger, min_months=s.stale_months, min_amount=s.min_amount)
    items = open_items(lines) if lines is not None else pd.DataFrame()
    aging_rows = aging(items, ledger)
    if lines is not None:
        # With the actual lines, the open items replace the balance-level estimate.
        groups = ledger.groupby("account")["group"].first()
        stale = [f for f in stale if groups.get(f.account) not in OPEN_ITEM_GROUPS]
        stale += find_old_items(items, ledger, min_days=s.stale_months * 30 + 5, min_amount=s.min_amount)
    historical = stale + find_recurring(ledger, min_amount=s.min_amount)
    relation_findings, relationships = find_relations(ledger, recent=s.recent_months, min_amount=s.min_amount)
    alerts = _dedupe_alerts(
        relation_findings
        + find_trends(ledger, targets=s.targets, min_amount=s.min_amount)
        + find_anomalies(ledger, recent=s.recent_months, min_amount=s.min_amount)
    )
    historical.sort(key=lambda f: f.amount, reverse=True)
    alerts.sort(key=lambda f: f.amount, reverse=True)

    periods = sorted(ledger["period"].unique())
    stale_kinds = ("stale_balance", "old_items")
    stale_total = sum(f.amount for f in historical if f.kind in stale_kinds)
    historical_dicts = [f.to_dict() for f in historical]
    alert_dicts = [f.to_dict() for f in alerts]
    return {
        "company": company,
        "period_from": period_label(periods[0]),
        "period_to": period_label(periods[-1]),
        "months": len(periods),
        "accounts": int(ledger["account"].nunique()),
        "summary": {
            "stale_total": round(float(stale_total), 2),
            "stale_count": sum(1 for f in historical if f.kind in stale_kinds),
            "recurring_count": sum(1 for f in historical if f.kind == "recurring"),
            "alerts_count": len(alerts),
            "alerts_total": round(float(sum(f.amount for f in alerts)), 2),
        },
        "source": "transactions" if lines is not None else "trial_balance",
        "historical": historical_dicts,
        "alerts": alert_dicts,
        "review": build_review(ledger, historical_dicts + alert_dicts, min_amount=s.min_amount),
        "aging": aging_rows,
        "relationships": sorted(relationships, key=lambda r: -r["correlation"]),
        "warnings": warnings or [],
        
    }
