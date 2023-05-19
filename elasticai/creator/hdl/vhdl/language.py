from dataclasses import dataclass as _dataclass


def _language_node(cls):
    return _dataclass(eq=True, frozen=True)(cls)


@_language_node
class SignalDef:
    name: str
    direction: str
    width: int


@_language_node
class Port:
    signal_defs: list[SignalDef]


@_language_node
class Connection:
    _to: str
    _from: str


@_language_node
class Connections:
    children: list[Connection]
