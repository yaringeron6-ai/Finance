"""Infer an account's kind and group when the source does not say.

Source systems use different charts of accounts, so the rules are keyword based
on the account name (Hebrew and English). A `kind` or `group` column in the
input always wins over these guesses.
"""

from __future__ import annotations

import re

from .model import (
    ACCRUAL,
    ADVANCE,
    ASSET,
    CLEARING,
    COGS,
    EQUITY,
    EXPENSE,
    GENERAL,
    GROUPS,
    KINDS,
    LIABILITY,
    PROVISION,
    REVENUE,
)

_KIND_RULES: list[tuple[str, str]] = [
    (COGS, r"עלות המכר|עלות מכר|קניות|חומרי גלם|cost of (goods|sales)|cogs|purchases"),
    (EXPENSE, r"סוציאל|social"),  # "הפרשות סוציאליות" is an expense, not a provision
    (REVENUE, r"הכנסות|הכנסה|מכירות|פדיון|revenue|sales|income"),
    (ASSET, r"מקדמ\S*\s+(ל-?)?ספק|advances? to suppliers?|prepaid"),
    (LIABILITY, r"מקדמ\S*\s+מ?לקוח|customer advances?"),
    (EQUITY, r"הון|עודפים|יתרת רווח|equity|retained|capital"),
    (LIABILITY, r"ספקים|זכאים|הפרש|מע\"?מ עסקאות|הלוואה|אשראי|suppliers|payable|provision|accrued|loan"),
    (EXPENSE, r"הוצאות|הוצאה|שכר|שכירות|פרסום|שיווק|אחזקה|ביטוח|עמלות|פחת|ריבית|"
              r"expense|salary|wages|rent|marketing|maintenance|insurance|fees|depreciation|interest"),
    (ASSET, r"בנק|קופה|לקוחות|חייבים|מקדמ|מלאי|רכוש|ציוד|מעבר|ביניים|לבירור|השעיה|"
            r"bank|cash|customers|receivable|advance|prepaid|inventory|asset|equipment|clearing|suspense"),
]

_GROUP_RULES: list[tuple[str, str]] = [
    (CLEARING, r"מעבר|ביניים|לבירור|השעיה|ממתין|clearing|suspense|transit"),
    (ADVANCE, r"מקדמ|advance|prepayment"),
    (ACCRUAL, r"חופש|הבראה|פיצוי|vacation|severance|holiday"),
    (PROVISION, r"הפרש|provision"),
]

_KIND_ALIASES = {
    "הכנסות": REVENUE, "הכנסה": REVENUE, "income": REVENUE, "sales": REVENUE,
    "עלות המכר": COGS, "cost of sales": COGS,
    "הוצאות": EXPENSE, "הוצאה": EXPENSE, "expenses": EXPENSE,
    "נכס": ASSET, "נכסים": ASSET, "assets": ASSET,
    "התחייבות": LIABILITY, "התחייבויות": LIABILITY, "liabilities": LIABILITY,
    "הון": EQUITY,
}

_GROUP_ALIASES = {
    "מעבר": CLEARING, "חשבון מעבר": CLEARING, "suspense": CLEARING,
    "מקדמות": ADVANCE, "מקדמה": ADVANCE,
    "הפרשה": PROVISION, "הפרשות": PROVISION,
    "הפרשה שוטפת": ACCRUAL,
    "כללי": GENERAL,
}


def _match(rules: list[tuple[str, str]], text: str) -> str | None:
    for label, pattern in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


def parse_kind(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in KINDS:
        return text
    return _KIND_ALIASES.get(text)


def parse_group(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in GROUPS:
        return text
    return _GROUP_ALIASES.get(text)


def infer_kind(name: str) -> str:
    return _match(_KIND_RULES, name) or EXPENSE


def infer_group(name: str, kind: str) -> str:
    if kind in (REVENUE, COGS, EXPENSE, EQUITY):
        return GENERAL
    return _match(_GROUP_RULES, name) or GENERAL
