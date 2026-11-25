import io

from fastapi.testclient import TestClient

from radar.web.app import app

client = TestClient(app)


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "מכ\"ם פיננסי" in res.text


def test_demo():
    res = client.get("/api/demo")
    assert res.status_code == 200
    body = res.json()
    assert body["months"] == 36
    assert len(body["historical"]) == 5
    assert len(body["alerts"]) == 3


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
