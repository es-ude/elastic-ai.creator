from .nn import Binarize as Binarize
from .precompute.truth_table_generation import (
    generate_input_tensor_1d as generate_input_tensor_1d,
)
from .rules import (
    AttachFilterParametersRule,
    FilterParameters,
    FilterParamsProducer,
    InferMaxPool1dInChannelsRule,
    InferNodeShapesRule,
    binarize_activations,
    make_split_conv_rule,
    precompute_linear,
    remove_redundant_layers,
    reorder,
)
from .rules._ir import DataGraph, Registry, build_sequential_ir
from .tensor_conversion import (
    torch1d_input_tensor_to_grouped_strings as torch1d_input_tensor_to_grouped_strings,
)
from .torch2ir import get_default_torch2ir
from .torch_analysis import compute_required_input_size

__all__ = [
    "AttachFilterParametersRule",
    "Binarize",
    "DataGraph",
    "FilterParameters",
    "FilterParamsProducer",
    "InferMaxPool1dInChannelsRule",
    "InferNodeShapesRule",
    "Registry",
    "binarize_activations",
    "build_sequential_ir",
    "compute_required_input_size",
    "generate_input_tensor_1d",
    "get_default_torch2ir",
    "make_split_conv_rule",
    "precompute_linear",
    "remove_redundant_layers",
    "reorder",
    "torch1d_input_tensor_to_grouped_strings",
]
