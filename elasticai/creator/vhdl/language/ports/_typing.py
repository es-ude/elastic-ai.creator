from typing import Protocol

from elasticai.creator.vhdl.typing import Identifiable


class Port(Protocol):
    def build_port_map(self, prefix: str) -> "PortMap":
        ...


class PortMap(Identifiable, Protocol):
    def signal_definitions(self) -> list[str]:
        ...

    def instantiation(self) -> list[str]:
        ...
