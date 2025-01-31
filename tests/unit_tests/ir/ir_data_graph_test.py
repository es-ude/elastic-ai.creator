from elasticai.creator.ir import Edge, Node, edge, node
from elasticai.creator.ir import Graph as _Graph


class Graph(_Graph[Node, Edge]):
    name: str
    type: str


def test_graph_is_serialized():
    g = Graph(data=dict(name="network", type="network"), node_fn=Node, edge_fn=Edge)
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
    g = Graph(data=dict(name="network", type="network"), node_fn=Node, edge_fn=Edge)
    assert g.name == "network"
