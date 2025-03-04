from enum import StrEnum

import pytest

import elasticai.creator.ir as ir
from elasticai.creator.graph import BaseGraph
from elasticai.creator.ir_transforms import SequenceReorderer
from elasticai.creator.torch2ir import (
    Implementation,
    Node,
    input_node,
    new_edge,
    output_node,
)


def reorder_blocks(impl: Implementation) -> Implementation:
    N = NodeType

    def make_nodes(*args: tuple[str, NodeType]) -> list[ir.Node]:
        return [ir.node(name=name, type=type.value) for name, type in args]

    pattern = make_nodes(
        ("conv", N.CONV1D),
        ("pool", N.MAXPOOL1D),
        ("bnorm", N.BATCHNORM1D),
        ("bin", N.BINARIZE),
        ("end", N.ANY),
    )

    replacement = make_nodes(
        ("conv", N.CONV1D),
        ("bnorm", N.BATCHNORM1D),
        ("bin", N.BINARIZE),
        ("pool", N.MAXPOOL1D),
        ("end", N.ANY),
    )

    def constraint(pattern_node: ir.Node, graph_node: ir.Node) -> bool:
        if pattern_node.type == N.ANY:
            return True
        return pattern_node.type == graph_node.type

    reorderer = SequenceReorderer(
        old_order=pattern,
        new_order=replacement,
        node_constraint=constraint,
    )

    reordered = reorderer.reorder(impl)
    return Implementation(data=reordered.data, graph=reordered.graph)


class NodeType(StrEnum):
    CONV1D = "conv1d"
    BATCHNORM1D = "batchnorm1d"
    MAXPOOL1D = "maxpool1d"
    BINARIZE = "binarize"
    ANY = "any"


class Conv1d(Node):
    in_channels: int
    out_channels: int
    kernel_size: int
    bias: bool
    groups: int
    stride: int
    padding: int

    @staticmethod
    def new(
        name: str,
        implementation: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        groups: int,
        stride: int,
        padding: int,
    ) -> "Conv1d":
        args = locals() | {"type": NodeType.CONV1D.value}
        return Conv1d(name, data=args)


class Batchnorm1d(Node):
    num_features: int
    affine: bool

    @staticmethod
    def new(
        name: str, implementation: str, num_features: int, affine: bool
    ) -> "Batchnorm1d":
        args = locals() | {"type": NodeType.BATCHNORM1D.value}
        return Batchnorm1d(name, data=args)


class Binarize(Node):
    @staticmethod
    def new(name: str, implementation: str) -> "Binarize":
        return Binarize(
            name=name,
            data=dict(type=NodeType.BINARIZE.value, implementation=implementation),
        )


class MaxPool1d(Node):
    kernel_size: int
    stride: int

    @staticmethod
    def new(
        name: str,
        implementation: str,
        kernel_size: int,
        stride: int,
    ) -> "MaxPool1d":
        args = locals() | {"type": NodeType.MAXPOOL1D.value}
        return MaxPool1d(name, args)


class SequentialBuilder:
    def __init__(self):
        self.clear()

    def _save_last_node(self, node: Node):
        self._last_node = node

    def clear(self) -> "SequentialBuilder":
        self.ir = Implementation(graph=BaseGraph())
        self.ir.name = "root"
        self.ir.type = "module"
        self._last_node = None
        return self

    def append(self, *nodes: Node) -> "SequentialBuilder":
        if len(nodes) > 1:
            for node in nodes:
                self.append(node)
        else:
            node = nodes[0]
            if self._last_node is None and node.type != "input":
                self.append(input_node())
            self.ir.add_node(node)
            if self._last_node is not None:
                self.ir.add_edge(new_edge(self._last_node.name, node.name))
            self._save_last_node(node)
        return self

    def build(self) -> Implementation:
        if self._last_node.type != "output":
            self.append(output_node())
        return self.ir


class TestReordering:
    # Note: the node names in the replaced subgraph are dictated by the function under test
    # We choose the same names for the nodes in the original graph, to make the test shorter

    @pytest.fixture(scope="class")
    def conv(self):
        return Conv1d.new(
            "conv",
            "conv1",
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            bias=False,
            groups=1,
            stride=1,
            padding=0,
        )

    @pytest.fixture(scope="class")
    def bnorm(self):
        return Batchnorm1d.new("bnorm", "bn", num_features=64, affine=True)

    @pytest.fixture(scope="class")
    def pooling(self):
        return MaxPool1d.new("pool", "maxpool", kernel_size=2, stride=2)

    @pytest.fixture(scope="class")
    def binarize(self):
        return Binarize.new("bin", "binarize")

    @pytest.fixture(scope="class")
    def seq(
        self, conv: Conv1d, bnorm: Batchnorm1d, pooling: MaxPool1d, binarize: Binarize
    ):
        seq = SequentialBuilder()
        seq.append(conv).append(pooling).append(bnorm).append(binarize)
        return seq.build()

    @pytest.fixture(scope="class")
    def expected(
        self, conv: Conv1d, bnorm: Batchnorm1d, pooling: MaxPool1d, binarize: Binarize
    ):
        seq = SequentialBuilder()
        seq.append(conv).append(bnorm).append(binarize).append(pooling)
        return seq.build()

    @pytest.fixture(scope="class")
    def reordered(self, seq: Implementation):
        return reorder_blocks(seq)

    def test_edges_are_in_correct_order(
        self, reordered: Implementation, expected: Implementation
    ):
        assert set(reordered.edges.keys()) == set(expected.edges.keys())

    def test_conv_data_is_copied(self, reordered: Implementation, seq: Implementation):
        assert reordered.nodes["conv"] == seq.nodes["conv"]

    def test_bnorm_data_is_copied(self, reordered: Implementation, seq: Implementation):
        assert reordered.nodes["bnorm"] == seq.nodes["bnorm"]

    def test_binarize_data_is_copied(
        self, reordered: Implementation, seq: Implementation
    ):
        assert reordered.nodes["bin"] == seq.nodes["bin"]

    def test_pooling_data_is_copied(
        self, reordered: Implementation, seq: Implementation
    ):
        assert reordered.nodes["pool"] == seq.nodes["pool"]
