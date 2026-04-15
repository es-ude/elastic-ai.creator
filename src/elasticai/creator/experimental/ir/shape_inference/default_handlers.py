from collections.abc import Callable
from typing import cast

from elasticai.creator.experimental.ir.shape_inference.shapes_calculation_functions import (
    Padding,
    Padding2D,
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

from .shape_inference import DataGraph, IrShapeInference, Shape, TypeHandler

_module_handlers: list[TypeHandler] = []
_type_handlers: list[Callable[[tuple[Shape, ...]], Shape]] = []


def _register_module(fn: TypeHandler) -> TypeHandler:
    _module_handlers.append(fn)
    return fn


def _register_type(
    fn: Callable[[tuple[Shape, ...]], Shape],
) -> Callable[[tuple[Shape, ...]], Shape]:
    _type_handlers.append(fn)
    return fn


def get_default_shape_inference() -> IrShapeInference:
    infer = IrShapeInference()
    for m in _module_handlers:
        infer.register()(m)
    for t in _type_handlers:
        infer.register_type()(t)
    return infer


def _unwrap1(v):
    """Extract scalar from a 1-element tuple/list (e.g. Conv1d stores kernel_size as (3,))."""
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


@_register_module
def linear(
    graph: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for linear. {input_shapes=}"
        )
    attr = graph.attributes
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


@_register_module
def conv1d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    def validate_shape(
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[int, int, int]:
        match input_shapes:
            case ((num_samples, channels, width),):
                return num_samples, channels, width
            case (_, _):
                raise Exception(
                    f"multiple input_shapes are not supported for conv1d. {input_shapes=}"
                )
            case _:
                raise Exception(
                    f"input shape for conv1d has to have the dimensions (N, C, W), but got {input_shapes[0]=}"
                )

    attr = dg_node.attributes
    padding = attr.get("padding")
    if isinstance(padding, (tuple, list)):
        padding = padding[0]
    return conv1d_output_shape(
        x_shape=validate_shape(input_shapes),
        out_channels=cast(int, attr.get("out_channels")),
        kernel_size=cast(int, _unwrap1(attr.get("kernel_size"))),
        stride=cast(int, _unwrap1(attr.get("stride"))),
        padding=cast(Padding, padding),
        dilation=cast(int, _unwrap1(attr.get("dilation"))),
    )


@_register_module
def conv2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    def validate_shape(
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[int, int, int, int]:
        match input_shapes:
            case ((num_samples, channels, height, width),):
                return num_samples, channels, height, width
            case (_, _):
                raise Exception(
                    f"multiple input_shapes are not supported for conv2d. {input_shapes=}"
                )
            case _:
                raise Exception(
                    f"expected 4-D input (NCHW) for conv2d. Got {input_shapes[0]=}"
                )

    attr = dg_node.attributes
    return conv2d_output_shape(
        x_shape=validate_shape(input_shapes),
        out_channels=cast(int, attr.get("out_channels")),
        kernel_size=cast(int | tuple[int, int], attr.get("kernel_size")),
        stride=cast(int | tuple[int, int], attr.get("stride")),
        padding=cast(Padding2D, attr.get("padding")),
        dilation=cast(int | tuple[int, int], attr.get("dilation")),
    )


@_register_module
def maxpool1d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for maxpool1d. {input_shapes=}"
        )
    attr = dg_node.attributes
    x_shape = cast(tuple[int, int, int], input_shapes[0])
    return maxpool1d_output_shape(
        x_shape=x_shape,
        kernel_size=cast(int, _unwrap1(attr.get("kernel_size"))),
        stride=cast(int, _unwrap1(attr.get("stride"))),
        padding=cast(int, _unwrap1(attr.get("padding"))),
        dilation=cast(int, _unwrap1(attr.get("dilation"))),
        ceil_mode=cast(bool, attr.get("ceil_mode", False)),
    )


@_register_module
def maxpool2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for maxpool2d. {input_shapes=}"
        )
    attr = dg_node.attributes
    x_shape = cast(tuple[int, int, int, int], input_shapes[0])
    return maxpool2d_output_shape(
        x_shape=x_shape,
        kernel_size=cast(int | tuple[int, int], attr.get("kernel_size")),
        stride=cast(int | tuple[int, int] | None, attr.get("stride")),
        padding=cast(int | tuple[int, int], attr.get("padding")),
        dilation=cast(int | tuple[int, int], attr.get("dilation")),
        ceil_mode=cast(bool, attr.get("ceil_mode", False)),
    )


@_register_module
def adaptiveavgpool2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for adaptiveavgpool2d. {input_shapes=}"
        )
    attr = dg_node.attributes
    x_shape = cast(tuple[int, int, int, int], input_shapes[0])
    return adaptiveavgpool2d_output_shape(
        x_shape=x_shape,
        output_size=cast(int | tuple[int, int], attr.get("output_size")),
    )


@_register_module
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
        num_features=cast(int, attr.get("num_features")),
    )


@_register_module
def batchnorm2d(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for batchnorm2d. {input_shapes=}"
        )
    attr = dg_node.attributes
    x_shape = cast(tuple[int, int, int, int], input_shapes[0])
    return batchnorm2d_output_shape(
        x_shape=x_shape,
        num_features=cast(int, attr.get("num_features")),
    )


@_register_module
def relu(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for relu. {input_shapes=}"
        )
    return relu_output_shape(input_shapes[0])


@_register_module
def sigmoid(
    dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    if len(input_shapes) != 1:
        raise Exception(
            f"multiple input_shapes are not supported for sigmoid. {input_shapes=}"
        )
    return sigmoid_output_shape(input_shapes[0])


@_register_module
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
        start_dim=cast(int, attr.get("start_dim", 1)),
        end_dim=cast(int, attr.get("end_dim", -1)),
    )


@_register_type
def add(input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    if len(input_shapes) == 0:
        raise Exception("add requires at least one input shape")
    return add_output_shape(input_shapes[0])
