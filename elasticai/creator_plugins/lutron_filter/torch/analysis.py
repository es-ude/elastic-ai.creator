from abc import abstractmethod
from typing import Callable, Protocol, runtime_checkable

from torch.nn import Sequential


@runtime_checkable
class _ConvProt(Protocol):
    @property
    @abstractmethod
    def kernel_size(self) -> tuple[int, ...]: ...
    @property
    @abstractmethod
    def stride(self) -> tuple[int, ...]: ...


@runtime_checkable
class _ConvBlockProto(Protocol):
    @property
    @abstractmethod
    def conv(self) -> _ConvProt: ...

    @property
    @abstractmethod
    def mpool(self) -> _ConvProt | Callable: ...


def compute_required_input_size(model: Sequential, target_size: int) -> int:
    def get_layer_output_size(target_size, layer):
        kernel_size = layer.kernel_size
        stride = layer.stride
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]

        if isinstance(stride, tuple):
            stride = stride[0]
        return compute_input_size(target_size, kernel_size, stride)

    for layer in reversed(tuple(model.children())):
        if isinstance(layer, _ConvProt):
            target_size = get_layer_output_size(target_size, layer)
        elif isinstance(layer, Sequential):
            target_size = compute_required_input_size(layer, target_size)
        elif isinstance(layer, _ConvBlockProto):
            if hasattr(layer.mpool, "kernel_size"):
                target_size = get_layer_output_size(target_size, layer.mpool)
            target_size = get_layer_output_size(target_size, layer.conv)
        elif layer.__class__.__name__ in (
            "Linear",
            "Sigmoid",
            "Binarize",
            "Flatten",
            "BatchNorm1d",
            "ParametrizedLinear",
        ):
            continue
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)} of {layer}")

    return target_size


def compute_input_size(output_size: int, kernel_size: int, stride: int):
    input_size = (output_size - 1) * stride + kernel_size
    return input_size
