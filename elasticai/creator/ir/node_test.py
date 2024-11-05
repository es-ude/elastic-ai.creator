from elasticai.creator.ir.node import Node as OtherNode

from .node import Node


def test_can_serialize_node_with_attributes():
    n = Node(name="my_node", type="my_type", attributes={"a": "b", "c": (1, 2)})
    assert dict(name="my_node", type="my_type", a="b", c=(1, 2)) == n.as_dict()


def test_can_deserialize_node_with_attributes():
    n = Node(name="my_node", type="my_type", attributes={"a": "b", "c": (1, 2)})
    assert n == Node.from_dict(dict(name="my_node", type="my_type", a="b", c=(1, 2)))


def test_import_path_does_not_matter_for_equality():
    n = Node(name="a", type="a_type", attributes=dict(a="b"))
    other = OtherNode(name="a", type="a_type", attributes=dict(a="b"))
    assert n == other
