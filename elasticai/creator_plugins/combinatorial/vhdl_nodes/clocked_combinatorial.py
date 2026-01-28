from typing import Any, Callable

import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import (
    Instance,
    LogicSignal,
    LogicVectorSignal,
    NullDefinedLogicSignal,
    Signal,
)
from elasticai.creator.ir2vhdl import (
    Node as VhdlNode,
)

from .node_factory import (
    InstanceFactoryForCombinatorial,
)


@InstanceFactoryForCombinatorial.register()
def clocked_combinatorial(node):
    return ClockedInstance(
        node, input_width=node.input_shape.size(), output_width=node.output_shape.size()
    )


class ClockedInstance(Instance):
    _logic_signals_with_default_suffix: tuple[str, ...] = tuple()
    _vector_signals_with_default_suffix: tuple[tuple[str, int], ...] = tuple()

    def __init__(
        self,
        node: VhdlNode,
        input_width: int,
        output_width: int,
        generic_map: dict[str, Any] | Callable[[], dict[str, Any]] = lambda: {},
    ):
        if callable(generic_map):
            generic_map = generic_map()

        port_map: dict[str, Signal] = dict(
            clk=NullDefinedLogicSignal("clk"),
            rst=NullDefinedLogicSignal("rst"),
            en=NullDefinedLogicSignal(name="en"),
            d_in=LogicVectorSignal(name="d_in", width=input_width),
            d_out=LogicVectorSignal(name="d_out", width=output_width),
        )
        for name, width in self._vector_signals_with_default_suffix:
            port_map[name] = LogicVectorSignal(name, width)
        for name in self._logic_signals_with_default_suffix:
            port_map[name] = LogicSignal(name)
        super().__init__(
            node,
            generic_map=dict(
                self._check_generic_map(
                    node.attributes.get("generic_map", ir.attribute())
                )
            )
            | generic_map,  #  pyright: ignore
            port_map=port_map,
        )

    @staticmethod
    def _check_generic_map(map: Any) -> ir.AttributeMapping:
        if not isinstance(map, ir.AttributeMapping):
            raise TypeError("generic map has to be an AttributeMapping")
        return map


class _ClockedCombinatorial(ClockedInstance):
    _logic_signals_with_default_suffix = ("src_valid", "valid", "dst_ready", "ready")
