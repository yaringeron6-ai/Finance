"""Web app: upload a trial balance (or run the demo) and get the radar's findings.

Run locally:  uvicorn radar.web.app:app --reload
"""

from __future__ import annotations

import io
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..connectors import REGISTRY, FileConnector
from ..engine import Settings, analyze
from ..normalize import InputError, _clean_header, normalize, read_table, to_number
from ..review import review_to_xlsx
from ..synthetic import COMPANY_NAME, DEMO_TARGETS, demo_transactions, demo_trial_balance
from ..transactions import is_transactions, normalize_transactions

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024

app = FastAPI(title="Finance Radar", docs_url="/api/docs")

# Recent analyses, kept in memory so the review can be downloaded as Excel.
_RECENT: OrderedDict[str, tuple] = OrderedDict()
_RECENT_MAX = 20


def _remember(report: dict, ledger, company: str | None) -> dict:
    report_id = uuid.uuid4().hex
    _RECENT[report_id] = (ledger, report["review"], company)
    while len(_RECENT) > _RECENT_MAX:
        _RECENT.popitem(last=False)
    report["id"] = report_id
    return report


def _run(raw: pd.DataFrame, targets: dict[str, float], company: str | None) -> dict:
    """Analyze a raw table: a כרטסת (ledger lines) or a monthly trial balance."""
    if is_transactions(raw):
        result = normalize_transactions(raw)
        report = analyze(result.ledger, Settings(targets=targets), company=company,
                         warnings=result.warnings, lines=result.lines)
    else:
        result = normalize(raw)
        report = analyze(result.ledger, Settings(targets=targets), company=company, warnings=result.warnings)
    return _remember(report, result.ledger, company)

_ACCOUNT_HEADERS = {_clean_header(h) for h in ("account", "כרטיס", "מספר כרטיס", "חשבון", "מפתח חשבון")}
_TARGET_HEADERS = {_clean_header(h) for h in ("target", "monthly target", "יעד", "יעד חודשי")}


def parse_targets(raw: pd.DataFrame) -> dict[str, float]:
    """A small table of account -> monthly target (in the account's natural sign)."""
    cols = {_clean_header(c): c for c in raw.columns}
    account = next((cols[c] for c in cols if c in _ACCOUNT_HEADERS), None)
    target = next((cols[c] for c in cols if c in _TARGET_HEADERS), None)
    if account is None or target is None:
        raise InputError("בקובץ היעדים צריכות להיות עמודות 'כרטיס' ו'יעד חודשי'.")
    out = {}
    for acc, value in zip(raw[account], raw[target]):
        number = to_number(value)
        if pd.notna(number) and str(acc).strip():
            out[str(acc).strip().removesuffix(".0")] = abs(float(number))
    return out


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "הקובץ גדול מדי (עד 20MB).")
    return data


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/connectors")
def connectors():
    return [vars(c.info) for c in REGISTRY]


@app.get("/api/demo")
def demo():
    return _run(demo_transactions(), DEMO_TARGETS, COMPANY_NAME)


def _csv(df: pd.DataFrame, filename: str) -> StreamingResponse:
    data = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig"))
    return StreamingResponse(data, media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/demo/file")
def demo_file():
    return _csv(demo_trial_balance(), "demo-trial-balance.csv")


@app.get("/api/demo/transactions")
def demo_transactions_file():
    return _csv(demo_transactions(), "demo-ledger-lines.csv")


@app.get("/api/review/{report_id}.xlsx")
def review_excel(report_id: str):
    if report_id not in _RECENT:
        raise HTTPException(404, "הניתוח כבר לא שמור. צריך להריץ אותו שוב.")
    ledger, review, company = _RECENT[report_id]
    data = io.BytesIO(review_to_xlsx(ledger, review, company))
    return StreamingResponse(
        data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=balance-sheet-review.xlsx"})


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), targets: UploadFile | None = File(None)):
    try:
        raw = FileConnector(await _read_upload(file), file.filename or "upload.csv").fetch_raw()
        target_map = {}
        if targets is not None and targets.filename:
            target_map = parse_targets(read_table(await _read_upload(targets), targets.filename))
        return _run(raw, target_map, None)
    except InputError as e:
        raise HTTPException(400, str(e)) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"לא הצלחתי לקרוא את הקובץ: {e}") from e
