from dataclasses import dataclass


@dataclass(kw_only=True)
class _SignalConfiguration:
    id: str
    default: str
    accepted_names: list[str]
