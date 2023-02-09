from typing import Protocol

from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.typing import Identifiable


class Port(Protocol):
    def build_port_map(self, prefix: str) -> "PortMap":
        ...


class PortMap(Identifiable, Protocol):
    def signal_definitions(self) -> Code:
        ...

    def instantiation(self) -> Code:
        ...
