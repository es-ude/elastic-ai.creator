import unittest
from typing import Iterable, TypeVar

from elasticai.creator.tests.unit.vhdl.components.fake_connectable_node import (
    FakeNodeFactory,
)
from elasticai.creator.vhdl.data_path_connection.node_iteration import NodeTraversal

T = TypeVar("T")


def first(xs: Iterable[T]) -> T:
    return next(iter(xs))


class DummyNodeTest(unittest.TestCase):
    def test_ids_increment(self):
        f = FakeNodeFactory()
        s1 = f.create([])
        s2 = f.create([])
        self.assertEqual("0", s1.id())
        self.assertEqual("1", s2.id())

    def test_ancestors(self):
        f = FakeNodeFactory()
        root = f.create([])
        root = f.create([root])
        children = list(root.children)
        ancestors = list(children[0].parents)
        self.assertEqual([root], ancestors)

    def test_traversal(self):
        f = FakeNodeFactory()
        root = f.create([])
        root = f.create([root])
        counter = 0
        nodes = NodeTraversal(root)
        for _ in nodes:
            counter += 1
        self.assertEqual(2, counter)


class SignalConnectionTest(unittest.TestCase):

    """
      - node connects its required inputs to provided outputs of parent nodes
      - we transform a data_path_connection of torch nodes to a data_path_connection of vhdl component nodes (with torch.fx) and the vhdl component nodes
        data_path_connection to a data_path_connection of portmaps/ports
      Basic Idea:
        Starting with leaf nodes, connect to higher nodes until no unconnected signals remain.

    Tests:
      - inserting data_path_connection with two nodes and same signals
    """

    pass


if __name__ == "__main__":
    unittest.main()
