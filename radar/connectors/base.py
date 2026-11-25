"""The connector contract.

Every data source (an uploaded file, Priority, Hashavshevet, SAP Business One,
...) implements `Connector`. A connector's only job is to return the trial
balance as a raw table; `normalize()` turns that into the ledger the engine
reads. Adding a new accounting system means writing one class here, nothing in
the engine changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from ..normalize import NormalizeResult, normalize


@dataclass
class ConnectorInfo:
    key: str
    display_name: str
    status: str                       # "available" | "planned"
    description: str
    settings: list[str] = field(default_factory=list)  # what the user must provide
    display_name_en: str = ""
    description_en: str = ""


class Connector(ABC):
    info: ConnectorInfo

    @abstractmethod
    def fetch_raw(self, period_from: pd.Period | None = None, period_to: pd.Period | None = None) -> pd.DataFrame:
        """Return monthly trial-balance rows in any column layout `normalize` accepts."""

    def column_mapping(self) -> dict[str, str] | None:
        """Override when the source's column names are known in advance."""
        return None

    def load(self, period_from: pd.Period | None = None, period_to: pd.Period | None = None) -> NormalizeResult:
        raw = self.fetch_raw(period_from, period_to)
        return normalize(raw, self.column_mapping())
