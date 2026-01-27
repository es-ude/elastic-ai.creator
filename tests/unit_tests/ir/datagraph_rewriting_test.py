from typing import overload

import pytest

from elasticai.creator.ir import AttributeMapping, attribute
from elasticai.creator.ir.datagraph import DataGraph, Node
from elasticai.creator.ir.datagraph_impl import DefaultIrFactory
from elasticai.creator.ir.datagraph_rewriting import (
    Pattern,
    PatternRule,
    PatternRuleSpec,
    StdPattern,
)
from elasticai.creator.ir.registry import Registry


class IrFactory(DefaultIrFactory):
    """Extended factory to make instantiation of typed nodes less verbose"""

    @overload
    def node(
        self, name: str, type: str, attributes: AttributeMapping | dict, /
    ) -> Node: ...

    @overload
    def node(self, name: str, type: str, /) -> Node: ...

    @overload
    def node(
        self, name: str, attributes: AttributeMapping = AttributeMapping()
    ) -> Node: ...

    def node(self, name, *args, **kwargs) -> Node:
        if len(args) == 2:
            _type, attributes = args
            if isinstance(attributes, dict):
                return super().node(
                    name, AttributeMapping(**(attributes | dict(type=_type)))
                )
        elif len(args) == 1 and isinstance(args[0], str):
            return super().node(name=name, attributes=AttributeMapping(type=args[0]))

        return super().node(name, *args, **kwargs)


@pytest.fixture
def factory() -> IrFactory:
    return IrFactory()


@pytest.fixture
def network(factory) -> DataGraph:
    return (
        factory.graph(
            AttributeMapping(
                **{"name": "root", "type": "network"},
            )
        )
        .add_node(factory.node("input", "input"))
        .add_node(factory.node("output", "output"))
    )


@pytest.fixture
def pattern(factory) -> DataGraph:
    return (
        factory.graph(
            AttributeMapping(
                **{"name": "pattern", "type": "pattern"},
            )
        )
        .add_node(factory.node("start", "interface"))
        .add_node(factory.node("end", "interface"))
    )


@pytest.fixture
def replacement(factory) -> DataGraph:
    return (
        factory.graph(
            AttributeMapping(**{"name": "replacement", "type": "replacement"}),
        )
        .add_node(factory.node("start", "interface"))
        .add_node(factory.node("end", "interface"))
    )


class DummyPattern(Pattern):
    def __init__(self, graph, interface, match):
        self._graph = graph
        self._interface = interface
        self._match = match

    @property
    def graph(self) -> DataGraph:
        return self._graph

    @property
    def interface(self) -> set[str]:
        return self._interface

    def match(self, __: DataGraph, _: Registry) -> list[dict[str, str]]:
        return self._match


def test_raise_error_when_interface_is_not_in_replacement(
    pattern, network, replacement, factory
):
    pattern_graph = pattern.add_node(factory.node("more_interface", "interface"))
    network = network.add_node(factory.node("some_node", "some_type")).add_edges(
        factory.edge("input", "some_node"), factory.edge("some_node", "output")
    )

    def build_replacement(_: DataGraph, __: Registry):
        return replacement.add_edge(factory.edge("start", "end")), Registry()

    pattern = DummyPattern(
        graph=pattern_graph,
        interface={"start", "end", "more_interface"},
        match=[{"start": "input", "end": "output", "more_interface": "some_node"}],
    )

    rule = PatternRule(
        PatternRuleSpec(
            pattern=pattern,
            replacement_fn=build_replacement,
        )
    )

    try:
        rule(network, Registry())
    except ValueError as e:
        assert str(e) == "Replacement is missing interface nodes: {'more_interface'}"
    else:
        assert False, "Expected ValueError was not raised."


def test_raise_error_when_interface_is_not_in_pattern(
    pattern, network, replacement, factory
):
    pattern_graph = pattern
    network = network.add_node(factory.node("some_node", "some_type")).add_edges(
        factory.edge("input", "some_node"), factory.edge("some_node", "output")
    )

    def build_replacement(_: DataGraph, __: Registry) -> tuple[DataGraph, Registry]:
        return (
            replacement()
            .add_factory.node(factory.node("more_interface", "interface"))
            .add_edge(factory.edge("start", "more_interface"))
            .add_edge(factory.edge("more_interface", "end"))
        )

    pattern = DummyPattern(
        graph=pattern_graph,
        interface={"start", "end", "more_interface"},
        match=[{"start": "input", "end": "output"}],
    )

    rule = PatternRule(
        PatternRuleSpec(
            pattern=pattern,
            replacement_fn=build_replacement,
        )
    )

    try:
        rule(network, Registry())
    except ValueError as e:
        assert str(e) == "Pattern Graph is missing interface nodes: {'more_interface'}"
    else:
        assert False, "Expected ValueError was not raised."


def test_replace_prelu(network, pattern, replacement, factory):
    impl = (
        network.add_node(factory.node("activation0", "prelu"))
        .add_edge(factory.edge("input", "activation0"))
        .add_edge(factory.edge("activation0", "output"))
    )
    pattern_graph = (
        pattern.add_node(factory.node("prelu", "prelu"))
        .add_edge(factory.edge("start", "prelu"))
        .add_edge(factory.edge("prelu", "end"))
    )

    def build_replacement(_, __):
        return (
            replacement.add_node(
                factory.node(
                    "binarize",
                    "binarize",
                    {
                        "implementation": "binarize",
                    },
                )
            )
            .add_edge(factory.edge("start", "binarize"))
            .add_edge(factory.edge("binarize", "end"))
        ), Registry()

    def node_constraint(pattern_node: Node, original_node: Node) -> bool:
        if pattern_node.name in ("start", "end"):
            return True
        return pattern_node.type == original_node.type

    pattern = StdPattern(
        graph=pattern_graph,
        node_constraint=node_constraint,
        interface={"start", "end"},
    )

    rule = PatternRule(
        PatternRuleSpec(
            pattern=pattern,
            replacement_fn=build_replacement,
        )
    )
    new_impl, _ = rule(impl, Registry())
    assert len(new_impl.successors) == 3
    assert len(new_impl.nodes) == 3
    assert "binarize" in new_impl.nodes
    assert new_impl.nodes["binarize"].type == "binarize"


def test_merge_layers(network, pattern, replacement, factory):
    impl = network.add_nodes(
        factory.node("conv0", "conv1d", {"parameters": {"weight": 4}}),
        factory.node("bnorm0", "batchnorm1d", {"parameters": {"weight": 5}}),
        factory.node("activation0", "relu"),
    ).add_edges(
        factory.edge("input", "conv0"),
        factory.edge("conv0", "bnorm0"),
        factory.edge("bnorm0", "activation0"),
        factory.edge("activation0", "output"),
    )
    pattern = pattern.add_nodes(
        factory.node("conv", "conv1d"),
        factory.node("bnorm", "batchnorm1d"),
        factory.node("act", "relu"),
    ).add_edges(
        factory.edge("start", "conv"),
        factory.edge("conv", "bnorm"),
        factory.edge("bnorm", "act"),
        factory.edge("act", "end"),
    )

    def build_replacement(match: DataGraph, _) -> tuple[DataGraph, Registry]:
        def fuse_conv() -> AttributeMapping:
            bnorm: AttributeMapping = match.nodes["bnorm"].attributes
            conv1d: AttributeMapping = match.nodes["conv"].attributes

            def dummy_combine_bnorm_and_conv_weights(conv_weights, bnorm_weights):
                return conv_weights + bnorm_weights

            fused: AttributeMapping = conv1d.update_path(
                ("parameters", "weight"),
                dummy_combine_bnorm_and_conv_weights(
                    conv1d["parameters"]["weight"], bnorm["parameters"]["weight"]
                ),
            )
            return fused

        return (
            replacement.add_node(factory.node("fused_conv", fuse_conv()))
            .add_edge(factory.edge("start", "fused_conv"))
            .add_edge(factory.edge("fused_conv", "end"))
        ), Registry()

    def node_constraint(pattern_node: Node, original_node: Node) -> bool:
        print(f"{pattern_node=}")
        print(f"{original_node=}")
        if pattern_node.name in ("start", "end"):
            return True
        constraint = pattern_node.type == original_node.type
        print(constraint)
        return constraint

    rule = PatternRule(
        PatternRuleSpec(
            pattern=StdPattern(
                pattern, node_constraint=node_constraint, interface={"start", "end"}
            ),
            replacement_fn=build_replacement,
        )
    )
    new_impl, _ = rule(impl, Registry())
    assert "fused_conv" in set(new_impl.nodes.keys())
    assert "conv0" not in set(new_impl.nodes.keys())


def test_remove_layers_from_matches_with_overlapping_interface_nodes(
    network: DataGraph, pattern: DataGraph, replacement: DataGraph, factory: IrFactory
) -> None:
    impl = network.add_nodes(
        factory.node("conv0", "conv1d"),
        factory.node("bnorm0", "batchnorm1d"),
        factory.node("activation0", "relu"),
        factory.node("conv1", "conv1d"),
        factory.node("bnorm1", "batchnorm1d"),
        factory.node("activation1", "relu"),
    ).add_edges(
        factory.edge("input", "conv0"),
        factory.edge("conv0", "bnorm0"),
        factory.edge("bnorm0", "activation0"),
        factory.edge("activation0", "conv1"),
        factory.edge("conv1", "bnorm1"),
        factory.edge("bnorm1", "activation1"),
        factory.edge("activation1", "output"),
    )
    pattern = pattern.add_nodes(
        factory.node("conv", "conv1d"),
        factory.node("bnorm", "batchnorm1d"),
        factory.node("act", "relu"),
    ).add_edges(
        factory.edge("start", "conv"),
        factory.edge("conv", "bnorm"),
        factory.edge("bnorm", "act"),
        factory.edge("act", "end"),
    )

    def build_replacement(_: DataGraph, reg: Registry) -> tuple[DataGraph, Registry]:
        return (
            replacement.add_node(factory.node("fused_conv", "conv1d"))
            .add_node(factory.node("activation", "relu"))
            .add_edge(factory.edge("start", "fused_conv"))
            .add_edge(factory.edge("fused_conv", "activation"))
            .add_edge(factory.edge("activation", "end"))
        ), reg

    def node_constraint(pattern_node: Node, original_node: Node) -> bool:
        if pattern_node.name in ("start", "end"):
            return True
        constraint = pattern_node.type == original_node.type
        return constraint

    spec = PatternRuleSpec(
        pattern=StdPattern(
            pattern, node_constraint=node_constraint, interface={"start", "end"}
        ),
        replacement_fn=build_replacement,
    )
    rule = PatternRule(spec)

    new_impl, _ = rule(impl, Registry())
    assert len(tuple(new_impl.nodes)) == len(tuple(impl.nodes)) - 1


def test_remove_layers_from_matches_with_overlapping_interface_nodes_with_injected_match(
    network, pattern, replacement, factory
):
    impl = network.add_nodes(
        factory.node("conv0", "conv1d"),
        factory.node("bnorm0", "batchnorm1d"),
        factory.node("activation0", "relu"),
        factory.node("conv1", "conv1d"),
        factory.node("bnorm1", "batchnorm1d"),
        factory.node("activation1", "relu"),
    ).add_edges(
        factory.edge("input", "conv0"),
        factory.edge("conv0", "bnorm0"),
        factory.edge("bnorm0", "activation0"),
        factory.edge("activation0", "conv1"),
        factory.edge("conv1", "bnorm1"),
        factory.edge("bnorm1", "activation1"),
        factory.edge("activation1", "output"),
    )
    pattern = pattern.add_nodes(
        factory.node("conv", "conv1d"),
        factory.node("bnorm", "batchnorm1d"),
        factory.node("act", "relu"),
    ).add_edges(
        factory.edge("start", "conv"),
        factory.edge("conv", "bnorm"),
        factory.edge("bnorm", "act"),
        factory.edge("act", "end"),
    )

    def build_replacement(_, reg):
        return (
            replacement.add_node(factory.node("fused_conv", "conv1d"))
            .add_node(factory.node("activation", "relu"))
            .add_edge(factory.edge("start", "fused_conv"))
            .add_edge(factory.edge("fused_conv", "activation"))
            .add_edge(factory.edge("activation", "end"))
        ), reg

    spec = PatternRuleSpec(
        pattern=DummyPattern(
            graph=pattern,
            interface={"start", "end"},
            match=[
                {
                    "start": "input",
                    "conv": "conv0",
                    "bnorm": "bnorm0",
                    "act": "activation0",
                    "end": "conv1",
                },
            ],
        ),
        replacement_fn=build_replacement,
    )
    rule = PatternRule(spec)

    new_impl, _ = rule(impl, Registry())
    assert len(tuple(new_impl.nodes)) == len(tuple(impl.nodes)) - 1


def test_copy_edge_data_from_edges_connected_to_interface_nodes(
    network, pattern, replacement, factory
):
    impl = network.add_nodes(
        factory.node("conv0", "conv1d"),
        factory.node("bnorm0", "batchnorm1d"),
        factory.node("activation0", "relu"),
        factory.node("conv1", "conv1d"),
        factory.node("bnorm1", "batchnorm1d"),
        factory.node("activation1", "relu"),
    ).add_edges(
        factory.edge("input", "conv0", attribute({"first": "first"})),
        factory.edge("conv0", "bnorm0"),
        factory.edge("bnorm0", "activation0"),
        factory.edge("activation0", "conv1"),
        factory.edge("conv1", "bnorm1", attribute({"third": "third"})),
        factory.edge("bnorm1", "activation1"),
        factory.edge("activation1", "output"),
    )
    pattern = pattern.add_nodes(
        factory.node("conv", "conv1d"),
        factory.node("bnorm", "batchnorm1d"),
        factory.node("act", "relu"),
    ).add_edges(
        factory.edge("start", "conv"),
        factory.edge("conv", "bnorm"),
        factory.edge("bnorm", "act"),
        factory.edge("act", "end"),
    )

    def build_replacement(_, reg):
        return (
            replacement.add_node(factory.node("fused_conv", "conv1d"))
            .add_node(factory.node("activation", "relu"))
            .add_edge(
                factory.edge("start", "fused_conv", attribute({"second": "second"}))
            )
            .add_edge(factory.edge("fused_conv", "activation"))
            .add_edge(factory.edge("activation", "end"))
        ), reg

    spec = PatternRuleSpec(
        pattern=DummyPattern(
            graph=pattern,
            interface={"start", "end"},
            match=[
                {
                    "start": "input",
                    "conv": "conv0",
                    "bnorm": "bnorm0",
                    "act": "activation0",
                    "end": "conv1",
                },
            ],
        ),
        replacement_fn=build_replacement,
    )
    rule = PatternRule(spec)

    new_impl, _ = rule(impl, Registry())
    assert {"second": "second", "third": "third"} == dict(
        new_impl.edges[("input", "fused_conv")].attributes
    ) | dict(new_impl.edges[("conv1", "bnorm1")].attributes)
