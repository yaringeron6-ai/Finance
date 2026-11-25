import io

from fastapi.testclient import TestClient

from radar.web.app import app

client = TestClient(app)


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Hello Fin" in res.text


def test_demo():
    res = client.get("/api/demo")
    assert res.status_code == 200
    body = res.json()
    assert body["months"] == 36
    assert body["source"] == "transactions"
    assert len(body["historical"]) == 6
    assert len(body["alerts"]) == 3
    assert body["review"]["counts"]["issue"] == 4


def test_review_excel_download():
    report_id = client.get("/api/demo").json()["id"]
    res = client.get(f"/api/review/{report_id}.xlsx")
    assert res.status_code == 200
    from openpyxl import load_workbook
    ws = load_workbook(io.BytesIO(res.content)).active
    assert ws.sheet_view.rightToLeft
    header = [c.value for c in ws[3]]
    assert header[0] == "כרטיס" and "הסבר" in header and "נבדק על ידי" in header
    assert ws.max_row == 3 + 8  # one row per balance-sheet account


def test_upload_ledger_lines():
    csv = client.get("/api/demo/transactions").content
    res = client.post("/api/analyze", files={"file": ("lines.csv", io.BytesIO(csv), "text/csv")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source"] == "transactions"
    assert {f["account"] for f in body["historical"] if f["kind"] == "old_items"} == {"1200", "1300", "2200", "2400"}


def test_upload_demo_file_with_targets():
    csv = client.get("/api/demo/file").content
    targets = "כרטיס,יעד חודשי\n6300,90000\n".encode("utf-8-sig")
    res = client.post("/api/analyze", files={
        "file": ("tb.csv", io.BytesIO(csv), "text/csv"),
        "targets": ("targets.csv", io.BytesIO(targets), "text/csv"),
    })
    assert res.status_code == 200, res.text
    trend = next(f for f in res.json()["alerts"] if f["account"] == "6300")
    assert "יעד" in trend["reason"]


def test_bad_upload_gets_hebrew_error():
    res = client.post("/api/analyze", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert res.status_code == 400
    assert "כרטיס" in res.json()["detail"]


def test_connectors_listed():
    keys = {c["key"]: c["status"] for c in client.get("/api/connectors").json()}
    assert keys["file"] == "available"
    assert keys["priority"] == "planned"


def test_password_required_when_set(monkeypatch):
    import base64

    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    assert client.get("/").status_code == 401
    assert client.get("/healthz").status_code == 200
    good = base64.b64encode(b"user:s3cret").decode()
    bad = base64.b64encode(b"user:nope").decode()
    assert client.get("/", headers={"Authorization": f"Basic {good}"}).status_code == 200
    assert client.get("/", headers={"Authorization": f"Basic {bad}"}).status_code == 401


def test_english():
    body = client.get("/api/demo?lang=en").json()
    assert body["lang"] == "en"
    assert body["company"] == "Example Co. Ltd."
    titles = " ".join(f["title"] for f in body["historical"] + body["alerts"])
    assert "open for over a year" in titles
    assert not any("֐" <= ch <= "׿" for ch in titles)
    res = client.post("/api/analyze?lang=en", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert res.status_code == 400 and "account" in res.json()["detail"]
    from openpyxl import load_workbook
    ws = load_workbook(io.BytesIO(client.get(f"/api/review/{body['id']}.xlsx").content)).active
    assert not ws.sheet_view.rightToLeft and ws["A3"].value == "Account"
