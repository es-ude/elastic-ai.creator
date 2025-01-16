from typing import Any, Callable

from elasticai.creator.ir2vhdl import Instance, LogicVectorSignal, VhdlNode

from .node_factory import (
    InstanceFactoryForCombinatorial,
)


@InstanceFactoryForCombinatorial.register("output")
@InstanceFactoryForCombinatorial.register("input")
@InstanceFactoryForCombinatorial.register("lutron")
@InstanceFactoryForCombinatorial.register
def unclocked_combinatorial(node):
    return UnclockedInstance(node)


class UnclockedInstance(Instance):
    def __init__(
        self,
        node: VhdlNode,
        generic_map: dict[str, Any] | Callable[[], dict[str, Any]] = lambda: {},
    ):
        if callable(generic_map):
            generic_map = generic_map()
        super().__init__(
            node,
            generic_map=node.attributes.get("generic_map", {}) | generic_map,  # pyright: ignore
            port_map=dict(
                d_in=LogicVectorSignal(name="d_in", width=node.input_shape.size()),
                d_out=LogicVectorSignal(name="d_out", width=node.output_shape.size()),
            ),
        )
