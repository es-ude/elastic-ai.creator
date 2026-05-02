from elasticai.creator import ir as _ir

from . import _ir as _lf_ir
from ._ir import pattern_rule, sequential_with_interface


def _constraint(
    pattern_node: _lf_ir.Node,
    graph_node: _lf_ir.Node,
    /,
) -> bool:
    match pattern_node.name:
        case "start":
            return graph_node.type in ("filter", "input")
        case "end":
            return True
        case _:
            return graph_node.type in ("binarize", "flatten")


def _match_only_final_sigmoid(
    pattern_node: _lf_ir.Node, graph_node: _lf_ir.Node
) -> bool:
    match pattern_node.name:
        case "start":
            return graph_node.type in ("filter", "input")
        case "end":
            return graph_node.type == "output"
        case _:
            return graph_node.type == "sigmoid"


def _replacement_fn[G: _ir.DataGraph](
    match: G, registry: _ir.Registry[G]
) -> tuple[G, _ir.Registry[G]]:
    edge = ("start", "end")
    new_seq = [match.nodes[n] for n in edge]
    new_g = match.clear().add_nodes(*new_seq).add_edge(*edge)
    return new_g, registry


def _make_remover(
    constraint: _lf_ir.NodeConstraint,
) -> _lf_ir.Rule[_lf_ir.DataGraph, _lf_ir.DataGraph]:

    remove_redundant_layers = pattern_rule(
        graph=sequential_with_interface("redundant"),
        replacement_fn=_replacement_fn,
        make_node_constraint=lambda _: constraint,
    )
    return remove_redundant_layers


remove_redundant_layers = _ir.compose_rules(
    *map(_make_remover, (_constraint, _match_only_final_sigmoid))
)
