"""General-ledger lines ("כרטסת"): parse them, and build the monthly ledger from them.

A כרטסת export has one row per journal line, e.g.

    כרטיס | שם כרטיס | תאריך | אסמכתא | פרטים | חובה | זכות

Uploading it is enough on its own: the monthly trial balance is the sum of the
lines per account per month. The lines also let the radar say *which* items are
stuck, not only that a balance is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .classify import infer_group, infer_kind, parse_group, parse_kind
from .i18n import L
from .model import LEDGER_COLUMNS, PNL_KINDS
from .normalize import InputError, NormalizeResult, _clean_header, _map_columns, to_number

TX_SYNONYMS: dict[str, list[str]] = {
    "date": ["date", "תאריך", "תאריך ערך", "תאריך אסמכתא", "תאריך רישום", "ת. ערך", "ת. אסמכתא"],
    "reference": ["reference", "ref", "document", "אסמכתא", "אסמכתא 1", "אסמכתה", "מספר מסמך", "מס מסמך"],
    "description": ["description", "details", "memo", "פרטים", "תיאור", "הערה", "תאור"],
    "counter": ["counter account", "contra", "חשבון נגדי", "כרטיס נגדי", "נגדי"],
}
_CLEAN_TX = {k: {_clean_header(s) for s in v} for k, v in TX_SYNONYMS.items()}


def _find(df: pd.DataFrame, key: str) -> str | None:
    for col in df.columns:
        if _clean_header(col) in _CLEAN_TX[key]:
            return col
    return None


def is_transactions(raw: pd.DataFrame) -> bool:
    """A table is a כרטסת when it has a day-level date plus a reference or description."""
    date = _find(raw, "date")
    if date is None or (_find(raw, "reference") is None and _find(raw, "description") is None):
        return False
    dates = pd.to_datetime(raw[date], errors="coerce", dayfirst=True)
    return dates.notna().mean() > 0.8 and dates.dt.day.nunique() > 1


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(text[iso].str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
    if (~iso).any():
        out[~iso] = pd.to_datetime(values[~iso], errors="coerce", dayfirst=True)
    return out


@dataclass
class TransactionsResult(NormalizeResult):
    lines: pd.DataFrame = field(default_factory=pd.DataFrame)


def normalize_transactions(raw: pd.DataFrame, mapping: dict[str, str] | None = None) -> TransactionsResult:
    raw = raw.dropna(how="all").copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    warnings: list[str] = []
    cols = _map_columns(raw, mapping)
    # In a כרטסת the generic "date" synonym is the line date, not a month.
    cols.pop("period", None)
    date_col = (mapping or {}).get("date") or _find(raw, "date")
    if "account" not in cols:
        raise InputError(L("לא נמצאה עמודת מספר כרטיס (למשל 'כרטיס' או 'account').",
                           "No account column found (for example 'account' or 'כרטיס')."))
    if date_col is None:
        raise InputError(L("לא נמצאה עמודת תאריך בכרטסת.", "No date column found in the ledger lines."))

    lines = pd.DataFrame({"account": raw[cols["account"]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)})
    lines["name"] = raw[cols["name"]].astype(str).str.strip() if "name" in cols else lines["account"]
    lines["date"] = _parse_dates(raw[date_col])
    ref_col, desc_col, counter_col = _find(raw, "reference"), _find(raw, "description"), _find(raw, "counter")
    lines["reference"] = raw[ref_col].astype(str).str.strip().replace({"nan": ""}) if ref_col else ""
    lines["description"] = raw[desc_col].astype(str).str.strip().replace({"nan": ""}) if desc_col else ""
    lines["counter"] = raw[counter_col].astype(str).str.strip().replace({"nan": ""}) if counter_col else ""
    if "debit" in cols and "credit" in cols:
        lines["amount"] = raw[cols["debit"]].map(to_number).fillna(0) - raw[cols["credit"]].map(to_number).fillna(0)
    elif "movement" in cols:
        lines["amount"] = raw[cols["movement"]].map(to_number)
    else:
        raise InputError(L("בכרטסת צריכות להיות עמודות 'חובה' ו'זכות' (או עמודת 'סכום').",
                           "Ledger lines need 'debit' and 'credit' columns (or an 'amount' column)."))
    lines["kind_given"] = raw[cols["kind"]].map(parse_kind).values if "kind" in cols else None
    lines["group_given"] = raw[cols["group"]].map(parse_group).values if "group" in cols else None

    bad = lines["date"].isna() | lines["account"].isin(["", "nan"])
    if bad.any():
        # Subtotal rows ("סה"כ לכרטיס") and blank lines land here.
        warnings.append(L(f"{int(bad.sum())} שורות בלי תאריך או כרטיס דולגו (למשל שורות סיכום).",
                          f"Skipped {int(bad.sum())} rows without a date or account (such as subtotal rows)."))
    lines = lines[~bad & lines["amount"].notna()].copy()
    if lines.empty:
        raise InputError(L("לא נמצאו שורות תנועה תקינות בכרטסת.", "No valid ledger lines found."))

    meta = lines.groupby("account").agg(name=("name", "first"), kind_given=("kind_given", "first"),
                                        group_given=("group_given", "first"))
    kinds, groups = {}, {}
    for account, row in meta.iterrows():
        kinds[account] = row["kind_given"] if isinstance(row["kind_given"], str) else infer_kind(row["name"])
        groups[account] = (row["group_given"] if isinstance(row["group_given"], str)
                           else infer_group(row["name"], kinds[account]))
    lines["name"] = lines["account"].map(meta["name"])
    lines["kind"] = lines["account"].map(kinds)
    lines["group"] = lines["account"].map(groups)
    lines["period"] = lines["date"].dt.to_period("M")
    lines = lines.drop(columns=["kind_given", "group_given"]).sort_values(["account", "date"]).reset_index(drop=True)

    periods = pd.period_range(lines["period"].min(), lines["period"].max(), freq="M")
    if len(periods) < 6:
        raise InputError(L("צריך לפחות 6 חודשים של תנועות כדי לזהות מגמות.",
                           "At least 6 months of entries are needed to find trends."))
    monthly = lines.groupby(["account", "period"])["amount"].sum()
    pieces = []
    years = np.array([p.year for p in periods])
    for account in meta.index:
        movement = monthly.loc[account].reindex(periods, fill_value=0.0).to_numpy(float)
        kind = kinds[account]
        if kind in PNL_KINDS:
            balance = pd.Series(movement).groupby(years).cumsum().to_numpy()
        else:
            balance = np.cumsum(movement)
        pieces.append(pd.DataFrame({
            "account": account, "name": meta.loc[account, "name"], "kind": kind, "group": groups[account],
            "period": periods, "movement": movement, "balance": balance,
        }))
    ledger = pd.concat(pieces, ignore_index=True)[LEDGER_COLUMNS]
    if not lines["description"].str.contains("פתיחה|opening", case=False, regex=True).any():
        warnings.append(L("לא זוהו שורות יתרת פתיחה בכרטסת: יתרות מאזניות חושבו מתחילת הקובץ.",
                          "No opening-balance lines found: balance-sheet balances start from the first line in the file."))
    return TransactionsResult(ledger=ledger, warnings=warnings, lines=lines)


def lines_to_trial_balance(lines: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """The monthly trial balance (Hebrew headers) a set of lines adds up to."""
    rows = ledger.assign(**{
        "חודש": ledger["period"].map(lambda p: f"{p.month:02d}/{p.year}"),
        "חובה": ledger["movement"].clip(lower=0), "זכות": (-ledger["movement"]).clip(lower=0),
    })
    return rows.rename(columns={"account": "כרטיס", "name": "שם כרטיס", "balance": "יתרה"})[
        ["כרטיס", "שם כרטיס", "חודש", "חובה", "זכות", "יתרה"]]

