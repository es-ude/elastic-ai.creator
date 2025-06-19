import warnings

from elasticai.creator.ir2vhdl import VhdlNode

from .clocked_combinatorial import ClockedInstance
from .node_factory import (
    InstanceFactoryForCombinatorial,
)


@InstanceFactoryForCombinatorial.register
def sliding_window(node):
    return SlidingWindowNode(node)


def _check_input_output_shape_compatibility(node):
    input_shape = node.input_shape
    output_shape = node.output_shape
    if input_shape.size() % output_shape.size() != 0:
        raise ValueError(
            "Found incompatible input output shapes for sliding_window. Total input size has to be an integer multiple of total output size, but found output={output} and input={input}.".format(
                output=output_shape, input=input_shape
            )
        )
    if output_shape.depth != input_shape.depth:
        warnings.warn(
            'Detected mismatching input output shapes for sliding_window for node "{}". Depth of output and input shape should usually be equal, but found output={} and input={}.'.format(
                node.name, output_shape, input_shape
            ),
            stacklevel=3,
        )


class SlidingWindowNode(ClockedInstance):
    _logic_signals_with_default_suffix = ("valid_in", "valid_out")

    def __init__(
        self,
        node: VhdlNode,
    ):
        _check_input_output_shape_compatibility(node)
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
