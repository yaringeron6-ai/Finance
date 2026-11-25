"""Language for every text the engine writes (Hebrew by default, English on request).

Text is written once per message with both languages side by side:

    L("פתוח יותר משנה", "open for over a year")

`use_lang("en")` sets the language for the code that runs inside it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

LANGS = ("he", "en")
_LANG: ContextVar[str] = ContextVar("lang", default="he")

HE_MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
             "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]


def lang() -> str:
    return _LANG.get()


def L(he: str, en: str) -> str:
    return en if _LANG.get() == "en" else he


def month_name(month: int) -> str:
    return (EN_MONTHS if _LANG.get() == "en" else HE_MONTHS)[month - 1]


def in_month(month: int) -> str:
    """'בדצמבר' / 'in December'."""
    return L(f"ב{HE_MONTHS[month - 1]}", f"in {EN_MONTHS[month - 1]}")


@contextmanager
def use_lang(code: str | None):
    token = _LANG.set(code if code in LANGS else "he")
    try:
        yield
    finally:
        _LANG.reset(token)
