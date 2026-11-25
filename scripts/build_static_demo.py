"""Build a single self-contained HTML page of the demo report (no server needed).

The page is the web app's own UI with the demo results embedded, so it can be
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
from radar.normalize import normalize  # noqa: E402
from radar.synthetic import COMPANY_NAME, DEMO_TARGETS, demo_trial_balance  # noqa: E402


def build(out: Path) -> None:
    html = (ROOT / "radar/web/static/index.html").read_text(encoding="utf-8")
    result = normalize(demo_trial_balance())
    report = analyze(result.ledger, Settings(targets=DEMO_TARGETS), company=COMPANY_NAME, warnings=result.warnings)
    connectors = [vars(c.info) for c in REGISTRY]

    style = re.search(r"<style>.*?</style>", html, re.S).group(0)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)

    # The upload card becomes a note: this page is the demo only.
    body = re.sub(
        r'<section class="card upload">.*?</section>',
        '<section class="card upload"><div class="warn">זו גרסת תצוגה עם נתוני הדמו של חברה בדויה. '
        'העלאת קובץ אמיתי עובדת באפליקציה המותקנת.</div>'
        '<div class="row"><span class="muted">מקורות נתונים:</span><div class="chips" id="connectors"></div></div>'
        '<div id="error" class="error hidden"></div><input type="file" id="file" hidden>'
        '<input type="file" id="targets" hidden><button id="run" hidden></button>'
        '<button id="demo" hidden></button><div id="drop" hidden></div><span id="fileLabel" hidden></span></section>',
        body, flags=re.S,
    )
    data = (f"const DEMO_REPORT = {json.dumps(report, ensure_ascii=False)};\n"
            f"const CONNECTORS = {json.dumps(connectors, ensure_ascii=False)};\n")
    body = body.replace("<script>", "<script>\n" + data, 1)
    body = body.replace('$("results").scrollIntoView({ behavior: "smooth", block: "start" });', "")
    body = re.sub(r'fetch\("/api/connectors"\)\.then\(\(r\) => r\.json\(\)\)\.then\(\(list\) => \{',
                  "((list) => {", body)
    body = body.replace("`).join(\"\");\n});\n</script>", "`).join(\"\");\n})(CONNECTORS);\n"
                        "render(DEMO_REPORT);\n</script>")
    page = (f'<title>מכ"ם פיננסי</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'{style}\n<div dir="rtl" lang="he">{body}</div>\n')
    out.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "demo.html"))
