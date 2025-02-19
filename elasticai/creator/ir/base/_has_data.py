from typing import Protocol

from .attribute import Attribute


class HasData(Protocol):
    data: dict[str, Attribute]
