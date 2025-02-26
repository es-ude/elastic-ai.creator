from collections.abc import Callable, Iterator, Sequence
from itertools import chain

from elasticai.creator.ir2vhdl import Implementation, type_handler

from .combinatorial import wrap_in_architecture
from .language import Port, VHDLEntity
from .vhdl_nodes.node_factory import InstanceFactoryForCombinatorial
from .wiring import connect_data_signals


@type_handler
def unclocked_combinatorial(impl: Implementation) -> tuple[str, Sequence[str]]:
    def _generate_entity(name, input_size, output_size):
        entity = VHDLEntity(
            name=name,
            port=Port(
                inputs=dict(
                    d_in=f"std_logic_vector({input_size} - 1 downto 0)",
                ),
                outputs=dict(
                    d_out=f"std_logic_vector({output_size} - 1 downto 0)",
                ),
            ),
            generics=dict(),
        )
        return entity

    return impl.name, combinatorial(impl, _generate_entity)


def combinatorial(
    impl: Implementation, entity_fn: Callable[[str, int, int], VHDLEntity]
) -> Iterator[str]:
    name = impl.name
    nodes = []
    for n in impl.nodes.values():
        nodes.append(InstanceFactoryForCombinatorial(n))
    nodes = tuple(nodes)
    instances = tuple(n for n in nodes if n.name not in ("input", "output"))
    input = impl.nodes["input"]
    output = impl.nodes["output"]

    connections = tuple((e.src, e.dst, e.src_dst_indices) for e in impl.edges.values())
    signal_defs = (line for n in nodes for line in n.define_signals())

    instantiations = (line for n in instances for line in n.instantiate())
    data_signal_connections = connect_data_signals(connections)
    entity = entity_fn(name, input.input_shape.size(), output.output_shape.size())

    yield from entity.generate_entity()

    yield ""
    declarations = signal_defs
    implementation = chain(data_signal_connections, instantiations)
    yield from wrap_in_architecture(name, declarations, implementation)
