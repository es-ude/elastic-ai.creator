from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .attribute import Attribute, AttributeMapping


@runtime_checkable
class _Node(Protocol):
    @property
    def attributes(self) -> AttributeMapping: ...


@runtime_checkable
class _ReadOnlyDataGraph[N](Protocol):
    @property
    def attributes(self) -> AttributeMapping: ...

    @property
    def successors(self) -> Mapping[str, Mapping[str, AttributeMapping]]: ...

    @property
    def predecessors(self) -> Mapping[str, Mapping[str, AttributeMapping]]: ...

    @property
    def nodes(self) -> Mapping[str, N]: ...


class IrSerializer[N]:
    def serialize(self, item: Attribute | _Node | _ReadOnlyDataGraph, /) -> Any:
        match item:
            case int() | float() | str() | bool():
                return item
            case tuple():
                return tuple(self.serialize(v) for v in item)
            case AttributeMapping():
                return {k: self.serialize(v) for k, v in item.items()}

            case _ReadOnlyDataGraph():
                return {
                    "nodes": {k: self.serialize(v) for k, v in item.nodes.items()},
                    "edges": {
                        src: {dst: self.serialize(attr) for dst, attr in dsts.items()}
                        for src, dsts in item.successors.items()
                    },
                    "attributes": self.serialize(item.attributes),
                }
            case _Node():
                return self.serialize(item.attributes)
            case _:
                raise TypeError(f"Unsupported attribute type: {type(item)} of {item}.")
