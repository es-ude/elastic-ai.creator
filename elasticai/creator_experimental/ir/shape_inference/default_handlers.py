from elasticai.creator import ir
from elasticai.creator_experimental.ir.shape_inference.shapes_calculation_functions import (
    adaptiveavgpool2d_output_shape,
    add_output_shape,
    batchnorm1d_output_shape,
    batchnorm2d_output_shape,
    conv1d_output_shape,
    conv2d_output_shape,
    flatten_output_shape,
    linear_output_shape,
    maxpool1d_output_shape,
    maxpool2d_output_shape,
    relu_output_shape,
    sigmoid_output_shape,
)

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]


def _unwrap1(v):
    """Extract scalar from a 1-element tuple/list (e.g. Conv1d stores kernel_size as (3,))."""
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


def linear(dg_node: DataGraph, input_shapes: tuple[tuple[int, ...]]) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for linear. {input_shapes=}"
        )
    attr = dg_node.attributes
    if input_shapes[0][-1] != attr.get("in_features"):
        raise Exception(
            f"expected {attr.get('in_features')} inputs for linear. Got {input_shapes[0][-1]=}"
        )
    out_features = attr.get("out_features")
    if not isinstance(out_features, int):
        raise Exception(
            f"expected {out_features} inputs for linear. Got {type(out_features)=}"
        )
    return linear_output_shape(input_shapes[0][0], out_features)  # ty:ignore[invalid-return-type]


def conv1d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for conv1d. {input_shapes=}"
        )
    if len(input_shapes[0]) != 3:
        raise Exception(f"expected 3-D input (NCW) for conv1d. Got {input_shapes[0]=}")
    attr = dg_node.attributes
    padding = attr.get("padding")
    if isinstance(padding, (tuple, list)):
        padding = padding[0]
    return conv1d_output_shape(
        x_shape=input_shapes[0],
        out_channels=attr.get("out_channels"),
        kernel_size=_unwrap1(attr.get("kernel_size")),
        stride=_unwrap1(attr.get("stride")),
        padding=padding,
        dilation=_unwrap1(attr.get("dilation")),
    )


def conv2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for conv2d. {input_shapes=}"
        )
    if len(input_shapes[0]) != 4:
        raise Exception(f"expected 4-D input (NCHW) for conv2d. Got {input_shapes[0]=}")
    attr = dg_node.attributes
    return conv2d_output_shape(
        x_shape=input_shapes[0],
        out_channels=attr.get("out_channels"),
        kernel_size=attr.get("kernel_size"),
        stride=attr.get("stride"),
        padding=attr.get("padding"),
        dilation=attr.get("dilation"),
    )


def maxpool1d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for maxpool1d. {input_shapes=}"
        )
    attr = dg_node.attributes
    return maxpool1d_output_shape(
        x_shape=input_shapes[0],
        kernel_size=_unwrap1(attr.get("kernel_size")),
        stride=_unwrap1(attr.get("stride")),
        padding=_unwrap1(attr.get("padding")),
        dilation=_unwrap1(attr.get("dilation")),
        ceil_mode=attr.get("ceil_mode", False),
    )


def maxpool2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for maxpool2d. {input_shapes=}"
        )
    attr = dg_node.attributes
    return maxpool2d_output_shape(
        x_shape=input_shapes[0],
        kernel_size=attr.get("kernel_size"),
        stride=attr.get("stride"),
        padding=attr.get("padding"),
        dilation=attr.get("dilation"),
        ceil_mode=attr.get("ceil_mode", False),
    )


def adaptiveavgpool2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for adaptiveavgpool2d. {input_shapes=}"
        )
    attr = dg_node.attributes
    return adaptiveavgpool2d_output_shape(
        x_shape=input_shapes[0],
        output_size=attr.get("output_size"),
    )


def batchnorm1d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for batchnorm1d. {input_shapes=}"
        )
    attr = dg_node.attributes
    return batchnorm1d_output_shape(
        x_shape=input_shapes[0],
        num_features=attr.get("num_features"),
    )


def batchnorm2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for batchnorm2d. {input_shapes=}"
        )
    attr = dg_node.attributes
    return batchnorm2d_output_shape(
        x_shape=input_shapes[0],
        num_features=attr.get("num_features"),
    )


def relu(dg_node: DataGraph, input_shapes: tuple[tuple[int, ...]]) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for relu. {input_shapes=}"
        )
    return relu_output_shape(input_shapes[0])


def sigmoid(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...]]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for sigmoid. {input_shapes=}"
        )
    return sigmoid_output_shape(input_shapes[0])


def flatten(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for flatten. {input_shapes=}"
        )
    attr = dg_node.attributes
    return flatten_output_shape(
        x_shape=input_shapes[0],
        start_dim=attr.get("start_dim", 1),
        end_dim=attr.get("end_dim", -1),
    )


def add(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) == 0:
        raise Exception("add requires at least one input shape")
    return add_output_shape(input_shapes[0])
