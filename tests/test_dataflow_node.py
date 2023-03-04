import unittest
from typing import Optional

from elasticai.creator.hdl.vhdl.dataflow.data_flow_node import Node
from elasticai.creator.hdl.vhdl.dataflow.data_flow_node import Sink as SinkNode
from elasticai.creator.hdl.vhdl.dataflow.data_flow_node import Source as SourceNode


class FakeSink:
    def __init__(self, *accepted_values: int):
        self.accepted_values = set(accepted_values)

    def accepts(self, source: int) -> bool:
        return source in self.accepted_values

    def __repr__(self):
        return f"FakeSink({self.accepted_values})"


class DataFlowNodeTest(unittest.TestCase):
    @staticmethod
    def node_with_sinks(*values: int):
        return Node(sinks={FakeSink(v) for v in values}, sources=set())

    @staticmethod
    def node_with_sources(*values: int):
        return Node(sinks=set(), sources={v for v in values})

    def test_node_without_sinks_is_satisfied(self):
        n = Node(sinks=set(), sources=set())
        self.assertTrue(n.is_satisfied())

    def test_unconnected_node_with_sink_is_unsatisfied(self):
        node = self.node_with_sinks(1)
        self.assertFalse(node.is_satisfied())

    def test_connected_node_with_sink_is_satisfied(self):
        child_with_sink = self.node_with_sinks(1)
        parent_with_matching_source = self.node_with_sources(1)
        child_with_sink.prepend(parent_with_matching_source)
        self.assertTrue(child_with_sink.is_satisfied())

    def test_node_not_satisfied_by_non_matching_source(self):
        child_with_sink = self.node_with_sinks(1)
        parent_without_matching_source = self.node_with_sources(2)
        parent_without_matching_source.append(child_with_sink)
        self.assertFalse(child_with_sink.is_satisfied())

    def test_sink_is_only_connected_once(self):
        child_with_sink = self.node_with_sinks(1, 2)
        first_parent = self.node_with_sources(1)
        second_parent = self.node_with_sources(1)
        first_parent.append(child_with_sink)
        second_parent.append(child_with_sink)
        self.assertEqual({first_parent}, child_with_sink.parents)

    def test_sink_is_connected_twice(self):
        child_with_sinks = self.node_with_sinks(1, 2)
        first_parent = self.node_with_sources(1)
        second_parent = self.node_with_sources(2)
        first_parent.append(child_with_sinks)
        second_parent.append(child_with_sinks)
        self.assertEqual({first_parent, second_parent}, child_with_sinks.parents)

    def test_can_get_both_child_nodes(self):
        first = self.node_with_sinks(1)
        second = self.node_with_sinks(1)
        parent = self.node_with_sources(1)
        parent.append(first)
        parent.append(second)
        self.assertEqual({first, second}, set(parent.children))

    def test_all_parent_sources_are_used(self):
        child = self.node_with_sinks(1, 2)
        parent = self.node_with_sources(1, 2)
        parent.append(child)
        self.assertTrue(child.is_satisfied())

    def test_sources_from_all_parent_nodes_are_considered_for_satisfaction(self):
        child = self.node_with_sinks(1, 2)
        first = self.node_with_sources(1)
        second = self.node_with_sources(2)
        first.append(child)
        second.append(child)
        self.assertTrue(child.is_satisfied())

    def test_can_reach_connected_sink_via_source(self) -> None:
        fake_sink = FakeSink(1, 2)
        child: Node[int] = Node(sinks={fake_sink}, sources=set())
        parent = self.node_with_sources(1)
        parent.append(child)
        connected_sink: Optional[SinkNode[int]] = None
        for source in parent.sources:
            for sink in source.sinks:
                connected_sink = sink
                break
        if connected_sink is None:
            self.fail()
        else:
            self.assertEqual(fake_sink, connected_sink.data)

    def test_can_reach_connected_source_via_sink(self) -> None:
        child: Node[int] = Node(sinks={FakeSink(1)}, sources=set())
        parent = self.node_with_sources(1)
        parent.append(child)
        connected_source: Optional[SourceNode] = None
        for sink in child.sinks:
            connected_source = sink.source
            if connected_source is not None:
                break
        if connected_source is None:
            self.fail("found no connected source")
        else:
            self.assertEqual(1, connected_source.data)
