"""Run every check on a ledger and assemble the report the web app shows."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .analysis.anomalies import find_anomalies
from .analysis.common import period_label
from .analysis.recurring import find_recurring
from .analysis.relations import find_relations
from .analysis.stale import find_stale
from .analysis.trends import find_trends
from .model import ALERT, HISTORICAL, Finding

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
            warnings: list[str] | None = None) -> dict:
    s = settings or Settings()
    historical = (find_stale(ledger, min_months=s.stale_months, min_amount=s.min_amount)
                  + find_recurring(ledger, min_amount=s.min_amount))
    relation_findings, relationships = find_relations(ledger, recent=s.recent_months, min_amount=s.min_amount)
    alerts = _dedupe_alerts(
        relation_findings
        + find_trends(ledger, targets=s.targets, min_amount=s.min_amount)
        + find_anomalies(ledger, recent=s.recent_months, min_amount=s.min_amount)
    )
    historical.sort(key=lambda f: f.amount, reverse=True)
    alerts.sort(key=lambda f: f.amount, reverse=True)

    periods = sorted(ledger["period"].unique())
    stale_total = sum(f.amount for f in historical if f.kind == "stale_balance")
    return {
        "company": company,
        "period_from": period_label(periods[0]),
        "period_to": period_label(periods[-1]),
        "months": len(periods),
        "accounts": int(ledger["account"].nunique()),
        "summary": {
            "stale_total": round(float(stale_total), 2),
            "stale_count": sum(1 for f in historical if f.kind == "stale_balance"),
            "recurring_count": sum(1 for f in historical if f.kind == "recurring"),
            "alerts_count": len(alerts),
            "alerts_total": round(float(sum(f.amount for f in alerts)), 2),
        },
        "historical": [f.to_dict() for f in historical],
        "alerts": [f.to_dict() for f in alerts],
        "relationships": sorted(relationships, key=lambda r: -r["correlation"]),
        "warnings": warnings or [],
        "sections": {HISTORICAL: "סריקה היסטורית", ALERT: "התראות"},
    }
