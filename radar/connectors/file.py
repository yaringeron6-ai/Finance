from __future__ import annotations

import pandas as pd

from ..normalize import read_table
from .base import Connector, ConnectorInfo


class FileConnector(Connector):
    """A trial balance exported to Excel or CSV from any accounting system."""

    info = ConnectorInfo(
        key="file",
        display_name="קובץ Excel / CSV",
        status="available",
        description="ייצוא מאזן בוחן חודשי מכל תוכנת הנהלת חשבונות.",
        settings=["file"],
    )

    def __init__(self, data: bytes, filename: str, mapping: dict[str, str] | None = None):
        self.data = data
        self.filename = filename
        self.mapping = mapping

    def fetch_raw(self, period_from=None, period_to=None) -> pd.DataFrame:
        return read_table(self.data, self.filename)

    def column_mapping(self):
        return self.mapping
