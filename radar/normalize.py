"""Turn a raw trial-balance table (any column names, long or wide) into a ledger.

Accepted shapes
---------------
Long: one row per account per month, e.g.

    כרטיס | שם כרטיס | חודש | חובה | זכות | יתרה

Wide: one row per account, one column per month holding that month's net
movement, e.g.

    כרטיס | שם כרטיס | 01/2024 | 02/2024 | ...

Amounts: a net `movement` column, or `debit` and `credit`, or only a closing
`balance`. Debit is positive. Missing pieces are derived; anything assumed is
reported back in `warnings`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .classify import infer_group, infer_kind, parse_group, parse_kind
from .i18n import L
from .model import LEDGER_COLUMNS, PNL_KINDS


class InputError(ValueError):
    """Raised with a Hebrew message the user can act on."""


COLUMN_SYNONYMS: dict[str, list[str]] = {
    "account": ["account", "account code", "account_code", "acct", "כרטיס", "מספר כרטיס",
                "מס כרטיס", "מס' כרטיס", "חשבון", "מספר חשבון", "מפתח חשבון", "מפתח"],
    "name": ["name", "account name", "account_name", "description", "שם כרטיס", "שם חשבון",
             "שם", "תיאור", "תאור"],
    "period": ["period", "month", "date", "תקופה", "חודש", "תאריך", "חודש דיווח"],
    "debit": ["debit", "dr", "חובה", "סכום חובה"],
    "credit": ["credit", "cr", "זכות", "סכום זכות"],
    "balance": ["balance", "closing balance", "יתרה", "יתרת סגירה", "יתרה מצטברת", "יתרה סופית"],
    "movement": ["movement", "net", "amount", "net movement", "תנועה", "תנועה נטו", "סכום", "שינוי"],
    "kind": ["kind", "type", "account type", "סוג", "סוג חשבון", "סוג כרטיס"],
    "group": ["group", "category", "קבוצה", "מיון", "סעיף", "קבוצת כרטיסים"],
}


def _clean_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\"'׳״`]", "", text)
    return re.sub(r"\s+", " ", text)


def _clean_synonyms() -> dict[str, set[str]]:
    return {k: {_clean_header(s) for s in v} for k, v in COLUMN_SYNONYMS.items()}


def parse_period(value: object) -> pd.Period | None:
    """Parse a month from the many ways exports write it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "year") and hasattr(value, "month"):
        return pd.Period(year=value.year, month=value.month, freq="M")
    text = str(value).strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?(?:[ T].*)?", text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
    else:
        m = re.fullmatch(r"(?:\d{1,2}[-/.])?(\d{1,2})[-/.](\d{4}|\d{2})", text)
        if not m:
            return None
        month, year = int(m.group(1)), int(m.group(2))
        if year < 100:
            year += 2000
    if not 1 <= month <= 12 or not 1900 < year < 2200:
        return None
    return pd.Period(year=year, month=month, freq="M")


def to_number(value: object) -> float:
    """'1,234.50' -> 1234.5, '(500)' -> -500, '₪ 20' -> 20, blank -> nan."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace("₪", "").replace(",", "").replace("‏", "").strip()
    if not text or text in {"-", "nan"}:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if text.endswith("-"):  # some exports write 500- for negatives
        negative, text = True, text[:-1]
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return -number if negative else number


@dataclass
class NormalizeResult:
    ledger: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _map_columns(df: pd.DataFrame, mapping: dict[str, str] | None) -> dict[str, str]:
    """Return {standard_name: source_column}."""
    found: dict[str, str] = {}
    if mapping:
        for std, src in mapping.items():
            if src not in df.columns:
                raise InputError(L(f"העמודה '{src}' שהוגדרה עבור '{std}' לא נמצאה בקובץ.",
                                   f"Column '{src}' mapped to '{std}' is not in the file."))
            found[std] = src
    synonyms = _clean_synonyms()
    for col in df.columns:
        clean = _clean_header(col)
        for std, names in synonyms.items():
            if std not in found and clean in names:
                found[std] = col
                break
    return found


def _wide_to_long(df: pd.DataFrame, cols: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    month_cols = {c: parse_period(c) for c in df.columns if c not in cols.values()}
    month_cols = {c: p for c, p in month_cols.items() if p is not None}
    if len(month_cols) < 2:
        raise InputError(
            L("לא זוהתה עמודת חודש. צריך עמודה בשם 'חודש' (פורמט ארוך) "
              "או עמודה לכל חודש, למשל 01/2024 (פורמט רחב).",
              "No month column found. Add a 'month' column (long layout) "
              "or one column per month, e.g. 01/2024 (wide layout).")
        )
    id_cols = list(cols.values())
    long = df.melt(id_vars=id_cols, value_vars=list(month_cols), var_name="__period", value_name="__movement")
    long["__period"] = long["__period"].map(month_cols)
    cols = dict(cols, period="__period", movement="__movement")
    return long, cols


def normalize(raw: pd.DataFrame, mapping: dict[str, str] | None = None) -> NormalizeResult:
    if raw is None or raw.empty:
        raise InputError(L("הקובץ ריק.", "The file is empty."))
    raw = raw.dropna(how="all").copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    warnings: list[str] = []

    cols = _map_columns(raw, mapping)
    if "account" not in cols:
        raise InputError(L("לא נמצאה עמודת מספר כרטיס (למשל 'כרטיס' או 'account').",
                           "No account column found (for example 'account' or 'כרטיס')."))
    if "period" not in cols:
        raw, cols = _wide_to_long(raw, cols)

    df = pd.DataFrame({"account": raw[cols["account"]].astype(str).str.strip()})
    df["account"] = df["account"].str.replace(r"\.0$", "", regex=True)
    df["name"] = raw[cols["name"]].astype(str).str.strip() if "name" in cols else df["account"]
    df["period"] = raw[cols["period"]].map(parse_period)
    bad = df["period"].isna() & df["account"].ne("") & df["account"].ne("nan")
    if bad.any():
        warnings.append(L(f"{int(bad.sum())} שורות עם חודש לא מזוהה דולגו.",
                          f"Skipped {int(bad.sum())} rows with an unreadable month."))
    df = df[df["period"].notna() & df["account"].ne("") & df["account"].ne("nan")]
    raw = raw.loc[df.index]

    has_movement = "movement" in cols
    has_dc = "debit" in cols and "credit" in cols
    has_balance = "balance" in cols
    if not (has_movement or has_dc or has_balance):
        raise InputError(L("לא נמצאו עמודות סכום: צריך 'תנועה', או 'חובה' ו'זכות', או 'יתרה'.",
                           "No amount columns found: add 'movement', or 'debit' and 'credit', or 'balance'."))

    if has_movement:
        df["movement"] = raw[cols["movement"]].map(to_number)
    elif has_dc:
        df["movement"] = raw[cols["debit"]].map(to_number).fillna(0) - raw[cols["credit"]].map(to_number).fillna(0)
    else:
        df["movement"] = np.nan
    df["balance"] = raw[cols["balance"]].map(to_number) if has_balance else np.nan

    kinds_given = raw[cols["kind"]].map(parse_kind) if "kind" in cols else pd.Series(None, index=df.index)
    groups_given = raw[cols["group"]].map(parse_group) if "group" in cols else pd.Series(None, index=df.index)
    df["kind_given"] = kinds_given.values
    df["group_given"] = groups_given.values

    # Same account and month twice (e.g. sub-ledgers) -> sum them.
    agg = df.groupby(["account", "period"], as_index=False).agg(
        name=("name", "first"),
        movement=("movement", lambda s: s.sum(min_count=1)),
        balance=("balance", lambda s: s.sum(min_count=1)),
        kind_given=("kind_given", "first"),
        group_given=("group_given", "first"),
    )

    periods = pd.period_range(agg["period"].min(), agg["period"].max(), freq="M")
    if len(periods) < 6:
        raise InputError(L("צריך לפחות 6 חודשים של נתונים כדי לזהות מגמות.",
                           "At least 6 months of data are needed to find trends."))

    pieces = []
    opening_warned = False
    for account, g in agg.groupby("account", sort=True):
        name = g["name"].iloc[0]
        kind = g["kind_given"].dropna().iloc[0] if g["kind_given"].notna().any() else infer_kind(name)
        group = g["group_given"].dropna().iloc[0] if g["group_given"].notna().any() else infer_group(name, kind)
        g = g.set_index("period").reindex(periods)
        movement, balance = g["movement"].astype(float), g["balance"].astype(float)
        is_pnl = kind in PNL_KINDS
        years = pd.Series([p.year for p in periods], index=periods)

        if movement.isna().all():
            # Derive movement from the closing balance.
            balance = balance.ffill()
            if is_pnl:  # P&L balances in a trial balance are year-to-date
                prev = balance.groupby(years).shift(1).fillna(0)
                movement = balance - prev
            else:
                movement = balance.diff()
                movement.iloc[0] = 0.0
        movement = movement.fillna(0.0)

        if balance.isna().all():
            if is_pnl:
                balance = movement.groupby(years).cumsum()
            else:
                balance = movement.cumsum()
                if not opening_warned:
                    opening_warned = True
                    warnings.append(L("לא נמצאה עמודת יתרה: יתרות הכרטיסים המאזניים חושבו מצטבר מאפס (בלי יתרת פתיחה).",
                                      "No balance column: balance-sheet balances were summed from zero "
                                      "(no opening balance)."))
        else:
            balance = balance.ffill().fillna(0.0)

        pieces.append(pd.DataFrame({
            "account": account, "name": name, "kind": kind, "group": group,
            "period": periods, "movement": movement.values, "balance": balance.values,
        }))

    ledger = pd.concat(pieces, ignore_index=True)[LEDGER_COLUMNS]
    return NormalizeResult(ledger=ledger, warnings=warnings)


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a raw DataFrame."""
    import io

    lower = filename.lower()
    buf = io.BytesIO(data)
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(buf, dtype=object)
    if lower.endswith((".csv", ".txt")):
        for encoding in ("utf-8-sig", "cp1255", "iso-8859-8"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, dtype=object, encoding=encoding, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        raise InputError(L("לא הצלחתי לקרוא את הקידוד של קובץ ה-CSV.", "Could not read the CSV file's encoding."))
    raise InputError(L("פורמט לא נתמך. אפשר להעלות Excel (xlsx) או CSV.",
                       "Unsupported format. Upload Excel (xlsx) or CSV."))
