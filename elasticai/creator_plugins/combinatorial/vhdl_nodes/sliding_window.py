from elasticai.creator.ir2vhdl import VhdlNode

from .clocked_combinatorial import ClockedInstance
from .node_factory import InstanceFactoryForCombinatorial


@InstanceFactoryForCombinatorial.register
def sliding_window(node):
    return SlidingWindowNode(node)


class SlidingWindowNode(ClockedInstance):
    _logic_signals_with_default_suffix = ("valid_in", "valid_out")

    def __init__(
        self,
        node: VhdlNode,
    ):
        data_width = node.input_shape.size()
        num_points = node.output_shape.size()
        if (
            "generic_map" in node.attributes  #  pyright: ignore
            and "stride" in node.attributes["generic_map"]  #  pyright: ignore
        ):
            stride = node.attributes["generic_map"]["stride"]  # pyright: ignore
        else:
            stride = node.attributes["stride"]  # pyright: ignore

        stride = stride * node.output_shape.depth
        super().__init__(
            node,
            input_width=data_width,
            output_width=num_points,
            generic_map=dict(
                stride=stride,
                input_width=data_width,
                output_width=num_points,
            ),
        )
