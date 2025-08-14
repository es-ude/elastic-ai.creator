from collections.abc import Sequence
from itertools import chain

from elasticai.creator import graph as gr
from elasticai.creator.ir2vhdl import Implementation, type_handler

from .combinatorial import (
    build_data_signal_connections_for_combinatorial,
    build_declarations_for_combinatorial,
    build_instantiations_for_combinatorial,
    wrap_in_architecture,
)
from .language import Port, VHDLEntity


@type_handler
def clocked_combinatorial(impl: Implementation) -> tuple[str, Sequence[str]]:
    def _iter():
        input_size = impl.nodes["input"].input_shape.size()
        output_size = impl.nodes["output"].output_shape.size()
        entity = VHDLEntity(
            name=impl.name,
            port=Port(
                inputs=dict(
                    clk="std_logic",
                    rst="std_logic",
                    d_in=f"std_logic_vector({input_size} - 1 downto 0)",
                    valid_in="std_logic",
                ),
                outputs=dict(
                    d_out=f"std_logic_vector({output_size} - 1 downto 0)",
                    valid_out="std_logic",
                ),
            ),
            generics=dict(),
        )
        yield from entity.generate_entity()
        yield ""
        declarations = build_declarations_for_combinatorial(impl)
        declarations = chain(
            declarations,
            (
                "signal valid_out_input : std_logic := '0';",
                "signal valid_in_output : std_logic := '0';",
            ),
        )
        definitions = build_data_signal_connections_for_combinatorial(impl)
        connected_valid_signals = []
        valid_in_out_pairs = tuple(_get_valid_in_out_pairs(impl).items())
        connected_valid_signals.extend(("valid_out_input <= valid_in;",))
        for dst, src in valid_in_out_pairs:
            connected_valid_signals.append(f"valid_in_{dst} <= valid_out_{src};")
        connected_valid_signals.append("valid_out <= valid_in_output;")
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


def _get_valid_in_out_pairs(impl: Implementation) -> dict[str, str]:
    def is_clocked(node):
        return node.type in [
            "striding_shift_register",
            "sliding_window",
            "shift_register",
        ]

    def iterate(node: str):
        def pred(node: str):
            return impl.predecessors(node)

        def succ(node: str):
            return impl.successors(node)

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
