import logging
from pprint import pformat

import elasticai.creator.ir.datagraph as dgraph
from elasticai.creator.ir import IrSerializer, Registry

from ._ir import DataGraph as _DataGraph
from ._ir import NameRegistry as _NameRegistry
from ._ir import Node as _Node
from ._ir import NodeConstraint as _NodeConstraint
from ._ir import attribute as _attribute
from ._ir import pattern_rule as _pattern_rule
from ._ir import sequential_with_interface as _sequential

logger = logging.getLogger(__name__)


def pretty_log_debug(g: dgraph.ReadOnlyDataGraph):
    serialized = IrSerializer().serialize(g)
    logger.debug(pformat(serialized), stacklevel=2)


def _pattern_graph() -> _DataGraph:
    return _sequential("maxpool1d", "batchnorm1d", "binarize")


def _make_constraint(_: Registry[_DataGraph], /) -> _NodeConstraint:
    def constraint(pattern_node: _Node, graph_node: _Node, /) -> bool:
        possible_start_node_types = ("maxpool1d", "input", "conv1d")
        possible_end_node_types = ("linear", "output", "conv1d")
        match pattern_node.name:
            case "start":
                return graph_node.type in possible_start_node_types
            case "end":
                return graph_node.type in possible_end_node_types
        return graph_node.type == pattern_node.type

    return constraint


def _replacement_fn(
    match: _DataGraph, registry: Registry[_DataGraph]
) -> tuple[_DataGraph, Registry[_DataGraph]]:
    logger.debug("replacing match {}".format(dict(match.nodes)))

    bnorm = registry[match.nodes["batchnorm1d"].implementation]
    num_bnorm_features = bnorm.attributes["num_features"]
    scaling = bnorm.attributes["parameters"]["weight"]
    needs_scaling_compensation = any(s < 0.0 for s in scaling)  # mypy: ignore
    if needs_scaling_compensation:
        logging.debug("using scaling compensation")
        logging.debug(f"scaling: {scaling}, num_bnorm_features: {num_bnorm_features}")
        scaling_compensation_vals = [
            [[1.0]] if s >= 0.0 else [[-1.0]] for s in scaling
        ]  # mypy: ignore
        implementation_names = _NameRegistry().prepopulate(registry.keys())
        new_name = implementation_names.get_unique_name("scaling_compensation")
        rescaling_attributes = _attribute(
            **{
                "type": "conv1d",
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": False,
                "groups": num_bnorm_features,
                "in_channels": num_bnorm_features,
                "out_channels": num_bnorm_features,
                "parameters": _attribute(
                    weight=scaling_compensation_vals,
                    bias=[0.0 for _ in scaling_compensation_vals],
                ),
            },
        )
        registry = registry | Registry(
            **{new_name: match.clear().with_attributes(rescaling_attributes)}
        )
        logging.debug("extended registry with")
        pretty_log_debug(registry[new_name])

        replacement = _sequential(
            "batchnorm1d",
            "binarize",
            "scaling_compensationA",
            "maxpool1d",
            "scaling_compensationB",
        ).add_nodes(
            (
                "scaling_compensationA",
                _attribute(type="conv1d", implementation=new_name),
            ),
            (
                "scaling_compensationB",
                _attribute(type="conv1d", implementation=new_name),
            ),
        )
    else:
        replacement = _sequential("batchnorm1d", "binarize", "maxpool1d")
    replacement = replacement.add_nodes(
        match.nodes["batchnorm1d"],
        match.nodes["binarize"],
        match.nodes["maxpool1d"],
    )
    return replacement, registry


reorder = _pattern_rule(
    _pattern_graph(),
    make_node_constraint=_make_constraint,
    replacement_fn=_replacement_fn,
)
