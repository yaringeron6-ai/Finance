"""Hello Fin web app: upload ledger lines or a trial balance (or run the demo) and get the findings.

Run locally:  uvicorn radar.web.app:app --reload
"""

from __future__ import annotations

import base64
import hmac
import io
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from ..connectors import REGISTRY, FileConnector
from ..engine import Settings, analyze
from ..normalize import InputError, _clean_header, normalize, read_table, to_number
from ..review import review_to_xlsx
from ..i18n import L, use_lang
from ..synthetic import COMPANY_NAME, COMPANY_NAME_EN, DEMO_TARGETS, demo_transactions, demo_trial_balance
from ..transactions import is_transactions, normalize_transactions

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024

app = FastAPI(title="Hello Fin", docs_url="/api/docs")


@app.middleware("http")
async def require_password(request: Request, call_next):
    """When APP_PASSWORD is set, every request needs it (any user name, that password)."""
    password = os.environ.get("APP_PASSWORD")
    if not password or request.url.path == "/healthz":
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
        except (ValueError, UnicodeDecodeError):
            given = ""
        if hmac.compare_digest(given.encode(), password.encode()):
            return await call_next(request)
    return Response("Password required / נדרשת סיסמה", status_code=401, media_type="text/plain; charset=utf-8",
                    headers={"WWW-Authenticate": 'Basic realm="Hello Fin", charset="UTF-8"'})


@app.get("/healthz")
def healthz():
    return {"ok": True}

# Recent analyses, kept in memory so the review can be downloaded as Excel.
_RECENT: OrderedDict[str, tuple] = OrderedDict()
_RECENT_MAX = 20


def _lang(value: str | None) -> str:
    return "en" if value == "en" else "he"


def _remember(report: dict, ledger, company: str | None, lang: str) -> dict:
    report_id = uuid.uuid4().hex
    _RECENT[report_id] = (ledger, report["review"], company, lang)
    while len(_RECENT) > _RECENT_MAX:
        _RECENT.popitem(last=False)
    report["id"] = report_id
    return report


def _run(raw: pd.DataFrame, targets: dict[str, float], company: str | None, lang: str) -> dict:
    """Analyze a raw table: ledger lines (כרטסת) or a monthly trial balance."""
    with use_lang(lang):
        if is_transactions(raw):
            result = normalize_transactions(raw)
            lines = result.lines
        else:
            result, lines = normalize(raw), None
    report = analyze(result.ledger, Settings(targets=targets), company=company,
                     warnings=result.warnings, lines=lines, lang=lang)
    return _remember(report, result.ledger, company, lang)


_ACCOUNT_HEADERS = {_clean_header(h) for h in ("account", "כרטיס", "מספר כרטיס", "חשבון", "מפתח חשבון")}
_TARGET_HEADERS = {_clean_header(h) for h in ("target", "monthly target", "יעד", "יעד חודשי")}


def parse_targets(raw: pd.DataFrame) -> dict[str, float]:
    """A small table of account -> monthly target (in the account's natural sign)."""
    cols = {_clean_header(c): c for c in raw.columns}
    account = next((cols[c] for c in cols if c in _ACCOUNT_HEADERS), None)
    target = next((cols[c] for c in cols if c in _TARGET_HEADERS), None)
    if account is None or target is None:
        raise InputError(L("בקובץ היעדים צריכות להיות עמודות 'כרטיס' ו'יעד חודשי'.",
                           "The targets file needs 'account' and 'monthly target' columns."))
    out = {}
    for acc, value in zip(raw[account], raw[target]):
        number = to_number(value)
        if pd.notna(number) and str(acc).strip():
            out[str(acc).strip().removesuffix(".0")] = abs(float(number))
    return out


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, L("הקובץ גדול מדי (עד 20MB).", "The file is too large (20MB at most)."))
    return data


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/connectors")
def connectors():
    return [vars(c.info) for c in REGISTRY]


@app.get("/api/demo")
def demo(lang: str = "he"):
    lang = _lang(lang)
    company = COMPANY_NAME_EN if lang == "en" else COMPANY_NAME
    return _run(demo_transactions(lang=lang), DEMO_TARGETS, company, lang)


def _csv(df: pd.DataFrame, filename: str) -> StreamingResponse:
    data = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig"))
    return StreamingResponse(data, media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/demo/file")
def demo_file(lang: str = "he"):
    return _csv(demo_trial_balance(lang=_lang(lang)), "demo-trial-balance.csv")


@app.get("/api/demo/transactions")
def demo_transactions_file(lang: str = "he"):
    return _csv(demo_transactions(lang=_lang(lang)), "demo-ledger-lines.csv")


@app.get("/api/review/{report_id}.xlsx")
def review_excel(report_id: str, lang: str = "he"):
    if report_id not in _RECENT:
        with use_lang(_lang(lang)):
            raise HTTPException(404, L("הניתוח כבר לא שמור. צריך להריץ אותו שוב.",
                                       "This analysis is no longer stored. Run it again."))
    ledger, review, company, stored_lang = _RECENT[report_id]
    with use_lang(stored_lang):
        data = io.BytesIO(review_to_xlsx(ledger, review, company))
    return StreamingResponse(
        data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=balance-sheet-review.xlsx"})


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), targets: UploadFile | None = File(None), lang: str = "he"):
    lang = _lang(lang)
    with use_lang(lang):
        try:
            raw = FileConnector(await _read_upload(file), file.filename or "upload.csv").fetch_raw()
            target_map = {}
            if targets is not None and targets.filename:
                target_map = parse_targets(read_table(await _read_upload(targets), targets.filename))
            return _run(raw, target_map, None, lang)
        except InputError as e:
            raise HTTPException(400, str(e)) from e
        except (ValueError, KeyError) as e:
            raise HTTPException(400, L(f"לא הצלחתי לקרוא את הקובץ: {e}", f"Could not read the file: {e}")) from e
