from abc import ABC
from dataclasses import dataclass

from ._signal import Signal


@dataclass(kw_only=True)
class _SignalBaseConfiguration:
    id: str
    default: str
    accepted_names: list[str]


class _SignalBase(Signal, ABC):
    def __init__(self, config: _SignalBaseConfiguration):
        self._id = config.id
        self._default = self._generate_default_suffix(config.default)
        self._accepted_names = config.accepted_names

    def id(self) -> str:
        return self._id

    @staticmethod
    def _generate_default_suffix(default_value: None | str) -> str:
        if default_value is None:
            return ""
        return f" := {default_value}"
