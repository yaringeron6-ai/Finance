"""Web app: upload a trial balance (or run the demo) and get the radar's findings.

Run locally:  uvicorn radar.web.app:app --reload
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..connectors import REGISTRY, FileConnector
from ..engine import Settings, analyze
from ..normalize import InputError, _clean_header, normalize, read_table, to_number
from ..synthetic import COMPANY_NAME, DEMO_TARGETS, demo_trial_balance

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024

app = FastAPI(title="Finance Radar", docs_url="/api/docs")

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
    result = normalize(demo_trial_balance())
    return analyze(result.ledger, Settings(targets=DEMO_TARGETS), company=COMPANY_NAME, warnings=result.warnings)


@app.get("/api/demo/file")
def demo_file():
    buf = io.StringIO()
    demo_trial_balance().to_csv(buf, index=False)
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    return StreamingResponse(data, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=demo-trial-balance.csv"})


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), targets: UploadFile | None = File(None)):
    try:
        result = FileConnector(await _read_upload(file), file.filename or "upload.csv").load()
        target_map = {}
        if targets is not None and targets.filename:
            target_map = parse_targets(read_table(await _read_upload(targets), targets.filename))
        return analyze(result.ledger, Settings(targets=target_map), company=None, warnings=result.warnings)
    except InputError as e:
        raise HTTPException(400, str(e)) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"לא הצלחתי לקרוא את הקובץ: {e}") from e
