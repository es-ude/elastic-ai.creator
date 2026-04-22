import dataclasses
from abc import abstractmethod
from typing import Any, Iterator, Protocol


class Definable(Protocol):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def define(self) -> Iterator[str]: ...


class Node(Protocol):
    implementation: str
    name: str
    type: str
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    attributes: dict[str, Any]


class VHDLNodeWrapper:
    def __init__(
        self,
        node: Node,
        generic_map: dict[str, str],
        port_map: dict[str, Definable],
    ):
        self._node = node
        self._generics: dict[str, str] = {k.lower(): v for k, v in generic_map.items()}
        self.port_map = port_map

    def add_signal_with_suffix(
        self, signal: Definable, prefix: str | None = None
    ) -> None:
        suffix = self._node.name
        prefix = signal.name if prefix is None else prefix
        self.port_map.update(
            {signal.name: dataclasses.replace(signal, name=f"{prefix}_{suffix}")}
        )

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def implementation(self) -> str:
        return self._node.implementation

    def define_signals(self) -> Iterator[str]:
        for s in self.port_map.values():
            yield from s.define()

    def generate_entity(self) -> Iterator[str]:
        yield ""

    def instantiate(self) -> Iterator[str]:
        yield from (f"{self.name}: entity work.{self.implementation}(rtl) ",)
        generics = tuple(self._generics.items())
        if len(generics) > 0:
            yield "generic map ("
            for key, value in generics[:-1]:
                yield f"  {key.upper()} => {value},"
            for g in generics[-1:]:
                yield f"  {g[0].upper()} => {g[1]}"
            yield "  )"
        port_map = tuple(self.port_map.items())
        yield "  port map ("

        for k, v in port_map[:-1]:
            yield f"    {k} => {v},"
        for k, v in port_map[-1:]:
            yield f"    {k} => {v}"

        yield "  );"
