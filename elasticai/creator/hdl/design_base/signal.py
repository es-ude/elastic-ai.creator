from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Signal:
    name: str
    width: int
