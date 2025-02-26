from collections.abc import Iterator

from elasticai.creator.ir2vhdl import Implementation

from .vhdl_nodes.node_factory import InstanceFactoryForCombinatorial
from .wiring import (
    connect_data_signals,
)


def wrap_node(n):
    return InstanceFactoryForCombinatorial(n)


def build_declarations_for_combinatorial(impl: Implementation) -> Iterator[str]:
    nodes = tuple(wrap_node(n) for n in impl.nodes.values())
    signal_defs = (line for n in nodes for line in n.define_signals())
    yield from signal_defs


def build_instantiations_for_combinatorial(impl: Implementation) -> Iterator[str]:
    nodes = tuple(wrap_node(n) for n in impl.nodes.values())
    instances = tuple(n for n in nodes if n.name not in ("input", "output"))
    instantiations = (line for n in instances for line in n.instantiate())
    yield from instantiations


def build_data_signal_connections_for_combinatorial(
    impl: Implementation,
) -> Iterator[str]:
    connections = tuple((e.src, e.dst, e.src_dst_indices) for e in impl.edges.values())
    data_signal_connections = connect_data_signals(connections)
    yield from data_signal_connections


def wrap_in_architecture(name, declarations, definitions):
    def _indent(line):
        return f"  {line}"

    yield f"architecture rtl of {name} is"
    yield from map(_indent, declarations)
    yield "begin"
    yield from map(
        _indent,
        (
            "d_in_input <= d_in;",
            "d_out_input <= d_in_input;",
            "d_out_output <= d_in_output;",
            "d_out <= d_out_output;",
        ),
    )
    yield from map(_indent, definitions)
    yield "end architecture;"
