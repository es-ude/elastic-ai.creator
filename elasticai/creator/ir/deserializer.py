from typing import Any

from .attribute import AttributeMapping, attribute
from .datagraph import DataGraph, Edge, Node
from .factories import IrFactory


class IrDeserializer[N: Node, E: Edge, G: DataGraph]:
    def __init__(self, factory: IrFactory[N, E, G]) -> None:
        self._factory = factory

    def _node(self, name: str, attributes: dict[str, AttributeMapping]) -> N:
        return self._factory.node(name, self._attributes(attributes))

    def _attributes(self, data: dict[str, Any]) -> AttributeMapping:
        return attribute(data)

    def _graph(self, data: dict[str, Any]) -> G:
        nodes = [
            self._node(name, attributes)
            for name, attributes in data.get("nodes", {}).items()
        ]
        attributes = self._attributes(data.get("attributes", {}))
        edges = []
        for src, dsts in data.get("edges", {}).items():
            for dst, edge_attributes in dsts.items():
                edge = self._factory.edge(
                    src,
                    dst,
                    self._attributes(edge_attributes),
                )
                edges.append(edge)
        return self._factory.graph(attributes).add_nodes(*nodes).add_edges(*edges)

    def deserialize_graph(self, data: dict[str, Any]) -> G:
        return self._graph(data)


class IrDeserializerLegacy[N: Node, E: Edge, G: DataGraph]:
    """Deserializes to the legacy format.

    The only difference is that attributes are not stored in a dedicated field but at the top-level dict.
    Use this if you need to load an IR from data that was serialized using the
    `IrData` data types.
    """

    def __init__(self, factory: IrFactory[N, E, G]):
        self._deserializer = IrDeserializer(factory)

    def deserialize_graph(self, data: dict[str, Any]) -> G:
        data = {
            "nodes": data["nodes"],
            "edges": data["edges"],
            "attributes": {k: data[k] for k in data if k not in ("nodes", "edges")},
        }
        return self._deserializer.deserialize_graph(data)
