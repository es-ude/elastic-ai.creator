from collections.abc import Sequence
from itertools import chain

import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator import graph as gr
from elasticai.creator.ir2vhdl import DataGraph, type_handler

from .combinatorial import (
    build_data_signal_connections_for_combinatorial,
    build_declarations_for_combinatorial,
    build_instantiations_for_combinatorial,
    wrap_in_architecture,
)
from .language import Port, VHDLEntity


def _is_clocked_node(node):
    return node.type in [
        "sliding_window",
        "shift_register",
    ]


@type_handler()
def clocked_combinatorial(
    impl: DataGraph, registry: ir.Registry
) -> tuple[str, Sequence[str]]:
    def _iter():
        input_size = impl.nodes["input"].input_shape.size()
        output_size = impl.nodes["output"].output_shape.size()
        entity = VHDLEntity(
            name=impl.name,
            port=Port(
                inputs=dict(
                    clk="std_logic",
                    en="std_logic",
                    rst="std_logic",
                    d_in=f"std_logic_vector({input_size} - 1 downto 0)",
                    src_valid="std_logic",
                    dst_ready="std_logic",
                ),
                outputs=dict(
                    d_out=f"std_logic_vector({output_size} - 1 downto 0)",
                    valid="std_logic",
                    ready="std_logic",
                ),
            ),
            generics=dict(),
        )
        yield from entity.generate_entity()
        yield ""
        declarations = build_declarations_for_combinatorial(impl)

        def build_ctrl_declaration(name) -> str:
            return f"signal {name} : std_logic := '0';"

        declarations = chain(
            declarations,
            map(
                build_ctrl_declaration,
                ("dst_ready_input", "src_valid_output", "valid_input", "ready_output"),
            ),
        )
        definitions = build_data_signal_connections_for_combinatorial(impl)
        connected_valid_signals = []
        valid_in_out_pairs = tuple(_get_valid_in_out_pairs(impl).items())
        connected_valid_signals.extend(
            ("valid_input <= src_valid;", "ready <= dst_ready_input;")
        )
        for dst, src in valid_in_out_pairs:
            connected_valid_signals.extend(
                (f"src_valid_{dst} <= valid_{src};", f"dst_ready_{src} <= ready_{dst};")
            )
        connected_valid_signals.extend(
            ("valid <= src_valid_output;", "ready_output <= dst_ready;")
        )

        definitions = chain(
            definitions,
            connected_valid_signals,
            ("", ""),
            build_instantiations_for_combinatorial(impl),
        )

        definitions = chain(
            definitions,
        )
        yield from wrap_in_architecture(impl.name, declarations, definitions)

    return impl.name, tuple(_iter())


def _get_valid_in_out_pairs(impl: DataGraph) -> dict[str, str]:
    is_clocked = _is_clocked_node

    def iterate(node: str):
        def pred(node: str):
            return impl.predecessors[node]

        def succ(node: str):
            return impl.successors[node]

        for node in gr.bfs_iter_up(successors=succ, predecessors=pred, start=node):
            yield impl.nodes[node]

    adjacency: dict[str, str] = {}
    last_clocked_node = "input"
    for node in filter(is_clocked, impl.nodes.values()):
        last_clocked_node = node.name
        if len(adjacency) == 0:
            adjacency[node.name] = "input"
        else:
            for pred in filter(
                is_clocked,
                iterate(node.name),
            ):
                adjacency[node.name] = pred.name
                break
    adjacency["output"] = last_clocked_node
    return adjacency
