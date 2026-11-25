"""Core data types shared by every part of the radar.

The engine never sees a source system's format. Every connector produces a
*ledger*: a long DataFrame with one row per (account, month) and these columns:

    account   str            account code ("כרטיס")
    name      str            account name
    kind      str            one of KINDS
    group     str            one of GROUPS (what kind of checks apply)
    period    pandas.Period  calendar month
    movement  float          net movement in the month, debit positive
    balance   float          closing balance at month end, debit positive
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Profit & loss kinds
REVENUE = "revenue"
COGS = "cogs"
EXPENSE = "expense"
# Balance-sheet kinds
ASSET = "asset"
LIABILITY = "liability"
EQUITY = "equity"

KINDS = (REVENUE, COGS, EXPENSE, ASSET, LIABILITY, EQUITY)
PNL_KINDS = (REVENUE, COGS, EXPENSE)
CREDIT_NATURE = (REVENUE, LIABILITY, EQUITY)

# Groups decide which historical checks apply to an account.
CLEARING = "clearing"    # חשבונות מעבר / ביניים / לבירור: should be zero at month end
ADVANCE = "advance"      # מקדמות: should be cleared against an invoice
PROVISION = "provision"  # הפרשות חד-פעמיות: should be used or released
ACCRUAL = "accrual"      # הפרשות שוטפות (חופשה, פיצויים): expected to stay open
CUSTOMER = "customer"    # לקוחות: open invoices are aged
SUPPLIER = "supplier"    # ספקים: open invoices are aged
BANK = "bank"            # בנק / קופה: reconciled against statements, not by flux
GENERAL = "general"

GROUPS = (CLEARING, ADVANCE, PROVISION, ACCRUAL, CUSTOMER, SUPPLIER, BANK, GENERAL)
OPEN_ITEM_GROUPS = (CLEARING, ADVANCE, PROVISION, CUSTOMER, SUPPLIER)
SHOULD_CLEAR_GROUPS = (CLEARING, ADVANCE, PROVISION)

LEDGER_COLUMNS = ["account", "name", "kind", "group", "period", "movement", "balance"]

# Finding sections
HISTORICAL = "historical"  # one-off scan of 2–3 years of history
ALERT = "alert"            # what changed in the most recent months

STATUS_TO_REVIEW = "לבדיקה"


def natural_sign(kind: str) -> int:
    """+1 when the account's normal balance is debit, -1 when it is credit."""
    return -1 if kind in CREDIT_NATURE else 1


@dataclass
class SeriesPoint:
    period: str
    value: float


@dataclass
class Evidence:
    """The numbers behind a finding, shaped for a small chart."""

    label: str
    series: list[SeriesPoint]
    highlight: list[str] = field(default_factory=list)
    reference: float | None = None
    reference_label: str | None = None


@dataclass
class Finding:
    section: str               # HISTORICAL or ALERT
    kind: str                  # stale_balance, recurring, anomaly, trend, relation_break
    account: str
    account_name: str
    title: str                 # one line, Hebrew
    reason: str                # why it was flagged, Hebrew
    check: str                 # what the controller should check, Hebrew (never tax advice)
    amount: float              # ILS at stake, used for ranking
    periods: list[str]
    evidence: Evidence
    status: str = STATUS_TO_REVIEW
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.amount = round(float(self.amount), 2)

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.account}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d
