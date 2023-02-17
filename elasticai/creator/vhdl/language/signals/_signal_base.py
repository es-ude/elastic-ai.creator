from typing import Optional


class _SignalBase:
    def __init__(self, id: str, default: Optional[str], accepted_names: list[str]):
        self._id = id
        self._default = self._generate_default_suffix(default)
        self._accepted_names = accepted_names

    def id(self) -> str:
        return self._id

    @staticmethod
    def _generate_default_suffix(default_value: None | str) -> str:
        if default_value is None:
            return ""
        return f" := {default_value}"
