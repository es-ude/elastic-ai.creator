from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ._attribute import Attribute, AttributeMapping


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


class IrSerializer:
    def serialize(self, item: Attribute | _Node | _ReadOnlyDataGraph, /) -> Any:
        match item:
            case int() | float() | str() | bool():
                return item
            case tuple() as item:
                return tuple(self.serialize(v) for v in item)
            case AttributeMapping() as item:
                return {k: self.serialize(v) for k, v in item.items()}

            case _ReadOnlyDataGraph() as item:
                return {
                    "nodes": {k: self.serialize(v) for k, v in item.nodes.items()},
                    "edges": {
                        src: {dst: self.serialize(attr) for dst, attr in dsts.items()}
                        for src, dsts in item.successors.items()
                    },
                    "attributes": self.serialize(item.attributes),
                }
            case _Node() as item:
                return self.serialize(item.attributes)
            case _:
                raise TypeError(f"Unsupported attribute type: {type(item)} of {item}.")


class IrSerializerLegacy:
    """Serializer for the legacy format.

    The only difference is that the attributes are not saved in a dedicated field, but instead directly in the top-level dict.

    Use this if you need to store the data in a format that can be read
    by the old `IrData` based implementations.
    """

    def __init__(self):
        self._serializer = IrSerializer()

    def serialize(self, item: Attribute | _Node | _ReadOnlyDataGraph, /) -> Any:
        data = self._serializer.serialize(item)
        converted = {}
        for k, v in data["attributes"].items():
            converted[k] = v
        converted["nodes"] = data["nodes"]
        converted["edges"] = data["edges"]
        return converted
