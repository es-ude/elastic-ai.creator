from elasticai.creator.graph import BaseGraph as _Graph
from elasticai.creator.ir import Edge, Implementation, Node, edge, node
from elasticai.creator.ir.base.attribute import Attribute


class Graph(Implementation[Node, Edge]):
    name: str
    type: str

    def __init__(self, data: dict[str, Attribute] | None = None):
        g: _Graph[str] = _Graph()
        if data is None:
            data = {}
        super().__init__(
            graph=g,
            node_fn=Node,
            edge_fn=Edge,
            data=data,
        )


def test_graph_is_serialized():
    g = Graph(data=dict(name="network", type="network"))
    g.add_node(node(name="node1", type="type1"))
    g.add_node(node(name="node2", type="type2"))
    g.add_edge(edge(src="node1", sink="node2"))
    assert g.data == {
        "name": "network",
        "type": "network",
        "nodes": {
            "node1": {"name": "node1", "type": "type1"},
            "node2": {"name": "node2", "type": "type2"},
        },
        "edges": {
            ("node1", "node2"): {"src": "node1", "sink": "node2"},
        },
    }


def test_graph_has_required_fields():
    g = Graph(data=dict(name="network", type="network"))
    assert g.name == "network"
