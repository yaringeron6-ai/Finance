"""Build a single self-contained HTML page of the demo (no server needed), in Hebrew and English.

The page is the web app's own UI with both demo reports embedded, so it can be
shared or viewed anywhere. Uploading a file still needs the real app.

    python scripts/build_static_demo.py out.html
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radar.connectors import REGISTRY  # noqa: E402
from radar.engine import Settings, analyze  # noqa: E402
from radar.synthetic import COMPANY_NAME, COMPANY_NAME_EN, DEMO_TARGETS, demo_transactions  # noqa: E402
from radar.transactions import normalize_transactions  # noqa: E402


def report(lang: str) -> dict:
    result = normalize_transactions(demo_transactions(lang=lang))
    company = COMPANY_NAME_EN if lang == "en" else COMPANY_NAME
    return analyze(result.ledger, Settings(targets=DEMO_TARGETS), company=company,
                   warnings=result.warnings, lines=result.lines, lang=lang)


def build(out: Path) -> None:
    html = (ROOT / "radar/web/static/index.html").read_text(encoding="utf-8")
    head = re.search(r"<head>(.*)</head>", html, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    head = re.sub(r'<meta (charset|name="viewport")[^>]*>\n?', "", head)
    data = {"reports": {"he": report("he"), "en": report("en")},
            "connectors": [vars(c.info) for c in REGISTRY]}
    inject = f"<script>window.HF_STATIC = {json.dumps(data, ensure_ascii=False)};</script>\n"
    body = body.replace("<script>", inject + "<script>", 1)
    out.write_text(head.strip() + "\n" + body, encoding="utf-8")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "demo.html"))
