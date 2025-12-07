import copy

from elasticai.creator.graph import BaseGraph
from elasticai.creator.ir import (
    Implementation,
    Node,
    PatternRuleSpec,
    edge,
    node,
)
from elasticai.creator.ir.rewriting import (
    DataGraph,
    Pattern,
    PatternRule,
    ReadOnlyDataGraph,
    StdPattern,
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

    def match(self, g: ReadOnlyDataGraph) -> list[dict[str, str]]:
        return self._match


def test_raise_error_when_interface_is_not_in_replacement():
    pattern_graph = make_pattern().add_node(node("more_interface", "interface"))
    network = (
        make_network()
        .add_node(node("some_node", "some_type"))
        .add_edges((edge("input", "some_node"), edge("some_node", "output")))
    )

    def replacement(_: ReadOnlyDataGraph):
        return make_replacement().add_edge(edge("start", "end"))

    pattern = DummyPattern(
        graph=pattern_graph,
        interface={"start", "end", "more_interface"},
        match=[{"start": "input", "end": "output", "more_interface": "some_node"}],
    )

    rule = PatternRule(
        PatternRuleSpec(
            pattern=pattern,
            replacement_fn=replacement,
        )
    )

    try:
        rule.apply(network)
    except ValueError as e:
        assert str(e) == "Replacement is missing interface nodes: {'more_interface'}"
    else:
        assert False, "Expected ValueError was not raised."


def test_raise_error_when_interface_is_not_in_pattern():
    pattern_graph = make_pattern()
    network = (
        make_network()
        .add_node(node("some_node", "some_type"))
        .add_edges((edge("input", "some_node"), edge("some_node", "output")))
    )

    def replacement(_: ReadOnlyDataGraph):
        return (
            make_replacement()
            .add_node(node("more_interface", "interface"))
            .add_edge(edge("start", "more_interface"))
            .add_edge(edge("more_interface", "end"))
        )

    pattern = DummyPattern(
        graph=pattern_graph,
        interface={"start", "end", "more_interface"},
        match=[{"start": "input", "end": "output"}],
    )

    rule = PatternRule(
        PatternRuleSpec(
            pattern=pattern,
            replacement_fn=replacement,
        )
    )

    try:
        rule.apply(network)
    except ValueError as e:
        assert str(e) == "Pattern Graph is missing interface nodes: {'more_interface'}"
    else:
        assert False, "Expected ValueError was not raised."


def test_replace_prelu():
    impl = (
        make_network()
        .add_node(node("activation0", "prelu"))
        .add_edge(edge("input", "activation0"))
        .add_edge(edge("activation0", "output"))
    )
    pattern_graph = (
        make_pattern()
        .add_node(node("prelu", "prelu"))
        .add_edge(edge("start", "prelu"))
        .add_edge(edge("prelu", "end"))
    )

    def replacement(match):
        return (
            make_replacement()
            .add_node(
                node(
                    "binarize",
                    "binarize",
                    {
                        "implementation": "binarize",
                    },
                )
            )
            .add_edge(edge("start", "binarize"))
            .add_edge(edge("binarize", "end"))
        )

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
            replacement_fn=replacement,
        )
    )
    new_impl = rule.apply(impl)
    assert len(new_impl.graph.nodes) == 3
    assert "binarize" in new_impl.nodes
    assert new_impl.nodes["binarize"].type == "binarize"


def test_merge_layers():
    impl = (
        make_network()
        .add_nodes(
            (
                node("conv0", "conv1d", {"parameters": {"weight": 4}}),
                node("bnorm0", "batchnorm1d", {"parameters": {"weight": 5}}),
                node("activation0", "relu"),
            )
        )
        .add_edges(
            (
                edge("input", "conv0"),
                edge("conv0", "bnorm0"),
                edge("bnorm0", "activation0"),
                edge("activation0", "output"),
            )
        )
    )
    pattern = (
        make_pattern()
        .add_nodes(
            (node("conv", "conv1d"), node("bnorm", "batchnorm1d"), node("act", "relu"))
        )
        .add_edges(
            (
                edge("start", "conv"),
                edge("conv", "bnorm"),
                edge("bnorm", "act"),
                edge("act", "end"),
            )
        )
    )

    def replacement(match):
        def fuse_conv():
            bnorm = match.nodes["bnorm"].data
            conv1d = match.nodes["conv"].data
            fused = copy.deepcopy(conv1d)

            def dummy_combine_bnorm_and_conv_weights(conv_weights, bnorm_weights):
                return conv_weights + bnorm_weights

            fused["parameters"]["weight"] = dummy_combine_bnorm_and_conv_weights(
                conv1d["parameters"]["weight"], bnorm["parameters"]["weight"]
            )
            return fused

        return (
            make_replacement()
            .add_node(Node("fused_conv", fuse_conv()))
            .add_edge(edge("start", "fused_conv"))
            .add_edge(edge("fused_conv", "end"))
        )

    def node_constraint(pattern_node: Node, original_node: Node) -> bool:
        if pattern_node.name in ("start", "end"):
            return True
        constraint = pattern_node.type == original_node.type
        return constraint

    rule = PatternRule(
        PatternRuleSpec(
            pattern=StdPattern(
                pattern, node_constraint=node_constraint, interface={"start", "end"}
            ),
            replacement_fn=replacement,
        )
    )
    new_impl = rule.apply(impl)
    assert "fused_conv" in set(new_impl.nodes.keys())
    assert "conv0" not in set(new_impl.nodes.keys())


def make_network() -> Implementation:
    return (
        Implementation(graph=BaseGraph(), data={"name": "root", "type": "network"})
        .add_node(node("input", "input"))
        .add_node(node("output", "output"))
    )


def make_pattern() -> Implementation:
    return (
        Implementation(graph=BaseGraph(), data={"name": "pattern", "type": "pattern"})
        .add_node(node("start", "interface"))
        .add_node(node("end", "interface"))
    )


def make_replacement() -> Implementation:
    return (
        Implementation(
            graph=BaseGraph(), data={"name": "replacement", "type": "replacement"}
        )
        .add_node(node("start", "interface"))
        .add_node(node("end", "interface"))
    )


def test_remove_layers_from_matches_with_overlapping_interface_nodes():
    impl = (
        make_network()
        .add_nodes(
            (
                node("conv0", "conv1d"),
                node("bnorm0", "batchnorm1d"),
                node("activation0", "relu"),
                node("conv1", "conv1d"),
                node("bnorm1", "batchnorm1d"),
                node("activation1", "relu"),
            )
        )
        .add_edges(
            (
                edge("input", "conv0"),
                edge("conv0", "bnorm0"),
                edge("bnorm0", "activation0"),
                edge("activation0", "conv1"),
                edge("conv1", "bnorm1"),
                edge("bnorm1", "activation1"),
                edge("activation1", "output"),
            )
        )
    )
    pattern = (
        make_pattern()
        .add_nodes(
            (node("conv", "conv1d"), node("bnorm", "batchnorm1d"), node("act", "relu"))
        )
        .add_edges(
            (
                edge("start", "conv"),
                edge("conv", "bnorm"),
                edge("bnorm", "act"),
                edge("act", "end"),
            )
        )
    )

    def replacement(match):
        return (
            make_replacement()
            .add_node(node("fused_conv", "conv1d"))
            .add_node(node("activation", "relu"))
            .add_edge(edge("start", "fused_conv"))
            .add_edge(edge("fused_conv", "activation"))
            .add_edge(edge("activation", "end"))
        )

    def node_constraint(pattern_node: Node, original_node: Node) -> bool:
        if pattern_node.name in ("start", "end"):
            return True
        constraint = pattern_node.type == original_node.type
        return constraint

    spec = PatternRuleSpec(
        pattern=StdPattern(
            pattern, node_constraint=node_constraint, interface={"start", "end"}
        ),
        replacement_fn=replacement,
    )
    rule = PatternRule(spec)

    new_impl = rule.apply(impl)
    assert len(tuple(new_impl.nodes)) == len(tuple(impl.nodes)) - 1


def test_remove_layers_from_matches_with_overlapping_interface_nodes_with_injected_match():
    impl = (
        make_network()
        .add_nodes(
            (
                node("conv0", "conv1d"),
                node("bnorm0", "batchnorm1d"),
                node("activation0", "relu"),
                node("conv1", "conv1d"),
                node("bnorm1", "batchnorm1d"),
                node("activation1", "relu"),
            )
        )
        .add_edges(
            (
                edge("input", "conv0"),
                edge("conv0", "bnorm0"),
                edge("bnorm0", "activation0"),
                edge("activation0", "conv1"),
                edge("conv1", "bnorm1"),
                edge("bnorm1", "activation1"),
                edge("activation1", "output"),
            )
        )
    )
    pattern = (
        make_pattern()
        .add_nodes(
            (node("conv", "conv1d"), node("bnorm", "batchnorm1d"), node("act", "relu"))
        )
        .add_edges(
            (
                edge("start", "conv"),
                edge("conv", "bnorm"),
                edge("bnorm", "act"),
                edge("act", "end"),
            )
        )
    )

    def replacement(match):
        return (
            make_replacement()
            .add_node(node("fused_conv", "conv1d"))
            .add_node(node("activation", "relu"))
            .add_edge(edge("start", "fused_conv"))
            .add_edge(edge("fused_conv", "activation"))
            .add_edge(edge("activation", "end"))
        )

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
        replacement_fn=replacement,
    )
    rule = PatternRule(spec)

    new_impl = rule.apply(impl)
    assert len(tuple(new_impl.nodes)) == len(tuple(impl.nodes)) - 1


def test_copy_edge_data_from_edges_connected_to_interface_nodes():
    impl = (
        make_network()
        .add_nodes(
            (
                node("conv0", "conv1d"),
                node("bnorm0", "batchnorm1d"),
                node("activation0", "relu"),
                node("conv1", "conv1d"),
                node("bnorm1", "batchnorm1d"),
                node("activation1", "relu"),
            )
        )
        .add_edges(
            (
                edge("input", "conv0", {"first": "first"}),
                edge("conv0", "bnorm0"),
                edge("bnorm0", "activation0"),
                edge("activation0", "conv1"),
                edge("conv1", "bnorm1", {"third": "third"}),
                edge("bnorm1", "activation1"),
                edge("activation1", "output"),
            )
        )
    )
    pattern = (
        make_pattern()
        .add_nodes(
            (node("conv", "conv1d"), node("bnorm", "batchnorm1d"), node("act", "relu"))
        )
        .add_edges(
            (
                edge("start", "conv"),
                edge("conv", "bnorm"),
                edge("bnorm", "act"),
                edge("act", "end"),
            )
        )
    )

    def replacement(match):
        return (
            make_replacement()
            .add_node(node("fused_conv", "conv1d"))
            .add_node(node("activation", "relu"))
            .add_edge(edge("start", "fused_conv", {"second": "second"}))
            .add_edge(edge("fused_conv", "activation"))
            .add_edge(edge("activation", "end"))
        )

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
        replacement_fn=replacement,
    )
    rule = PatternRule(spec)

    new_impl = rule.apply(impl)
    assert {"second": "second", "third": "third"} == dict(
        new_impl.edges[("input", "fused_conv")].attributes
    ) | dict(new_impl.edges[("conv1", "bnorm1")].attributes)
