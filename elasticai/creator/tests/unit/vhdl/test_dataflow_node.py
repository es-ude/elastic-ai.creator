import unittest

from elasticai.creator.vhdl.data_path_connection.data_flow_node import (
    DataFlowNode as Node,
)


class FakeSink:
    def __init__(self, *accepted_values: int):
        self._accepted_values = accepted_values

    def accepts(self, source: int) -> bool:
        return source in self._accepted_values


class DataFlowNodeTest(unittest.TestCase):
    def test_node_without_sinks_is_satisfied(self):
        n = Node(sinks=set(), sources=set())
        self.assertTrue(n.is_satisfied())

    def test_unconnected_node_with_sink_is_unsatisfied(self):
        child_with_sink = Node(sinks={FakeSink(1)}, sources=set())
        self.assertFalse(child_with_sink.is_satisfied())
