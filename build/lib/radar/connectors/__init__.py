from .base import Connector, ConnectorInfo
from .file import FileConnector
from .planned import HashavshevetConnector, PriorityConnector, SapB1Connector

REGISTRY: list[type[Connector]] = [FileConnector, PriorityConnector, HashavshevetConnector, SapB1Connector]

__all__ = [
    "Connector", "ConnectorInfo", "FileConnector",
    "PriorityConnector", "HashavshevetConnector", "SapB1Connector", "REGISTRY",
]
