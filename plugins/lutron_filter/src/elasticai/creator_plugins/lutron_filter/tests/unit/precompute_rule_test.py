from collections.abc import Iterable
from typing import override

import pytest
from elasticai.creator_plugins.lutron_filter.rules import _ir as ir
from elasticai.creator_plugins.lutron_filter.rules.precomputation import (
    FilterParameters,
    PrecomputationStrategy,
    make_precompute_rule,
)

import elasticai.creator.ir.datagraph_rewriting as _rew
from elasticai.creator.ir import AttributeMapping


@pytest.fixture
def factory():
    return ObjectUnderTestFactory(MockStrategy)


def test_obtaining_the_expected_number_of_lutrons(factory):
    result = factory.precomputed()
    result_lutrons = get_graph_by_type("lutron", result[1])
    expected_lutrons = get_graph_by_type(type="lutron", registry=factory.expected()[1])

    assert len(expected_lutrons) == 1
    assert len(result_lutrons) == len(expected_lutrons)


def test_content_of_lutron_0_is_correct(factory):
    result = factory.precomputed()
    lutron_0 = result[1]["lutron"].attributes
    expected = factory.expected()[1]["lutron"].attributes
    assert dict(lutron_0) == dict(expected)


def test_added_filter_implementation_is_correct(factory):
    result = factory.precomputed()[1]
    filters = get_graph_by_type("grouped_filter", result)
    assert len(filters) == 1, "missing expected lutron_filter graph in registry"
    filter = filters[0]
    assert filter.attributes["kernel_per_group"] == ("lutron",)


def test_matched_node_is_replaced_by_filter_node(factory):
    result = factory.precomputed()[0]
    assert "conv1d" not in result.nodes
    assert result.nodes["lutron_filter"].type == "filter"
    assert dict(result.nodes["lutron_filter"].attributes) == dict(
        factory.expected()[0].nodes["1"].attributes
    )


def test_ensure_we_set_the_module():
    factory = ObjectUnderTestFactory(ExplodingDummyStrategy)
    factory.precomputed()
    assert True


class MockStrategy(PrecomputationStrategy):
    def __init__(
        self,
        io_pairs: Iterable[Iterable[tuple[str, str]]],
        filter_parameters: FilterParameters,
        pattern_graph: ir.DataGraph,
    ) -> None:
        self._io_pairs = tuple(tuple(table) for table in io_pairs)
        self._filter_params = filter_parameters
        self._pattern_graph = pattern_graph

    @override
    def constraint(self, registry: ir.Registry) -> ir.NodeConstraint:
        def _constr(pattern_node, graph_node):
            if pattern_node.type == "interface":
                return graph_node.type in ("binarize",)
            return pattern_node.type == graph_node.type

        return _constr

    @property
    @override
    def pattern_graph(self) -> _rew.DataGraph:
        return self._pattern_graph

    @override
    def get_filter_parameters(self) -> FilterParameters:
        return self._filter_params

    @override
    def get_io_pairs(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        return self._io_pairs


class Explosion(Exception):
    def __init__(self):
        super().__init__("failed to set the module on current strategy")


class ExplodingDummyStrategy(PrecomputationStrategy):
    def __init__(self, *args, **kwargs):
        self._module_set = False

    @override
    def constraint(self, _: ir.Registry) -> ir.NodeConstraint:
        def _constr(_, __):
            return True

        return _constr

    @override
    def get_io_pairs(self) -> Iterable[Iterable[tuple[str, str]]]:
        if not self._module_set:
            raise Explosion()
        return tuple()

    @property
    @override
    def pattern_graph(self) -> ir.DataGraph:
        return ir.sequential_with_interface("linear")

    @override
    def get_filter_parameters(self) -> FilterParameters:
        return FilterParameters(kernel_size=1, in_channels=1, out_channels=1)

    @override
    def set_module(self, g, reg):
        self._module_set = True
        super().set_module(g, reg)


def get_graph_by_type(type: str, registry: ir.Registry) -> list[ir.DataGraph]:
    result = []
    for g in registry.values():
        if g.type == type:
            result.append(g)
    return result


class ObjectUnderTestFactory:
    def __init__(self, strategy):
        self.strategy = strategy

    def binarize(self) -> AttributeMapping:
        return ir.attribute(type="binarize")

    def linear(self) -> AttributeMapping:
        return ir.attribute(
            type="linear",
        )

    def lutron_linear(self) -> AttributeMapping:
        return ir.attribute(
            type="lutron_filter",
            kernel_per_group=["lutron"],
            input_size=1,
            output_size=1,
            stride=1,
            in_channels=1,
            out_channels=1,
            groups=1,
            kernel_size=1,
        )

    def lutron_0(self) -> AttributeMapping:
        return ir.attribute(
            type="lutron",
            truth_table=(
                ("0", "1"),
                ("1", "0"),
            ),
            input_size=1,
            output_size=1,
        )

    def original(self):
        return ir.build_sequential_ir(
            sequence=(
                "binarize",
                "linear",
                "binarize",
            ),
            registry=dict(
                binarize=self.binarize(),
                linear=self.linear(),
            ),
        )

    def precomputed(self):
        strategy = self.strategy(
            io_pairs=(self.lutron_0()["truth_table"],),
            filter_parameters=FilterParameters(
                kernel_size=1, in_channels=1, out_channels=1
            ),
            pattern_graph=ir.sequential_with_interface(
                ("linear", "linear"),
            ),
        )
        rule = make_precompute_rule(strategy)
        return rule(*self.original())

    def expected(self):
        graph, reg = ir.build_sequential_ir(
            sequence=("binarize", "lutron_filter", "binarize"),
            registry=dict(
                binarize=self.binarize(),
                lutron_filter=self.lutron_linear(),
                lutron=self.lutron_0(),
            ),
        )
        node_attr = self.lutron_linear().drop("kernel_per_group") | dict(
            type="filter", implementation="lutron_filter"
        )

        graph = graph.add_node("1", node_attr)
        return graph, reg
