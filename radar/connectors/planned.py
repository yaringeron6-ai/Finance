"""Direct API connectors that are designed but not built yet.

Each one documents the settings it will need. Implementing one means filling in
`fetch_raw` to call the system's API and return monthly trial-balance rows
(account, name, month, debit, credit, balance), and `column_mapping` for that
system's field names. The endpoints are deliberately left out until they are
checked against each vendor's documentation and a real account.
"""

from __future__ import annotations

from .base import Connector, ConnectorInfo


class _Planned(Connector):
    def __init__(self, **settings):
        self.settings = settings

    def fetch_raw(self, period_from=None, period_to=None):
        raise NotImplementedError(f"החיבור ל-{self.info.display_name} עוד לא נבנה. בינתיים אפשר להעלות קובץ ייצוא.")


class PriorityConnector(_Planned):
    info = ConnectorInfo(
        key="priority",
        display_name="Priority",
        status="planned",
        description="חיבור ישיר דרך ה-REST API של Priority.",
        settings=["server_url", "company", "username", "api_key"],
    )


class HashavshevetConnector(_Planned):
    info = ConnectorInfo(
        key="hashavshevet",
        display_name="חשבשבת",
        status="planned",
        description="חיבור לחשבשבת (Hashavshevet / H-Web).",
        settings=["server_url", "company", "username", "api_key"],
    )


class SapB1Connector(_Planned):
    info = ConnectorInfo(
        key="sap_b1",
        display_name="SAP Business One",
        status="planned",
        description="חיבור דרך ה-Service Layer של SAP Business One.",
        settings=["service_layer_url", "company_db", "username", "password"],
    )
