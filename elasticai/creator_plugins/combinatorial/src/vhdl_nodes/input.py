from elasticai.creator.ir2vhdl import Instance, LogicVectorSignal, VhdlNode

from .node_factory import InstanceFactoryForCombinatorial


@InstanceFactoryForCombinatorial.register
def input(node: VhdlNode) -> Instance:
    return Instance(
        node=node,
        generic_map={},
        port_map=dict(
            d_in=LogicVectorSignal(f"d_in_{node.name}", node.input_shape.size()),
            d_out=LogicVectorSignal(f"d_out_{node.name}", node.output_shape.size()),
        ),
    )
