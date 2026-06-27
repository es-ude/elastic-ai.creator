from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaConf:
    width: int
    offset: int = 0
    saturate: bool = True
