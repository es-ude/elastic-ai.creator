import copy

from elasticai.creator.graph import BaseGraph
from elasticai.creator.ir import Implementation, Node, Rewriter, RewriteRule, edge, node


def test_replace_prelu():
    rewriter = Rewriter()
    impl = (
        make_network()
        .add_node(node("activation0", "prelu"))
        .add_edge(edge("input", "activation0"))
        .add_edge(edge("activation0", "output"))
    )
    pattern = (
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

    rule = RewriteRule(
        pattern=pattern,
        replacement=replacement,
        node_constraint=node_constraint,
        interface={"start", "end"},
    )
    rewriter.add_rule(rule)
    new_impl = rewriter.apply(impl)
    assert len(new_impl.graph.nodes) == 3
    assert "binarize" in new_impl.nodes
    assert new_impl.nodes["binarize"].type == "binarize"


def test_merge_layers():
    rewrite = Rewriter()
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

    rule = RewriteRule(
        pattern=pattern,
        replacement=replacement,
        node_constraint=node_constraint,
        interface={"start", "end"},
    )
    rewrite.add_rule(rule)
    new_impl = rewrite.apply(impl)
    assert "fused_conv" in new_impl.nodes
    assert "conv0" not in new_impl.nodes


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
    rewrite = Rewriter()
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

    rule = RewriteRule(
        pattern=pattern,
        replacement=replacement,
        node_constraint=node_constraint,
        interface={"start", "end"},
    )
    rewrite.add_rule(rule)
    new_impl = rewrite.apply(impl)
    assert len(tuple(new_impl.nodes)) == len(tuple(impl.nodes)) - 1
