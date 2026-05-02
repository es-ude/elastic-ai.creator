from elasticai.creator.ir import Edge

from ._translate import prepare_for_training, translate
from .nn import Binarize as Binarize
from .precompute.truth_table_generation import (
    generate_input_tensor_1d as generate_input_tensor_1d,
)
from .rules import (
    AttachFilterParametersRule,
    FilterParameters,
    FilterParamsProducer,
    InferMaxPool1dInChannelsRule,
    binarize_activations,
    create_shape_inference,
    make_split_conv_rule,
    precompute_linear,
    precompute,
    remove_redundant_layers,
    reorder,
)
from .rules._ir import DataGraph, Node, Registry, build_sequential_ir
from .rules._ir import ir_factory as factory
from .rules._precomputation_impls import precompute
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
    "create_shape_inference",
    "Registry",
    "Edge",
    "Node",
    "binarize_activations",
    "build_sequential_ir",
    "compute_required_input_size",
    "generate_input_tensor_1d",
    "get_default_torch2ir",
    "make_split_conv_rule",
    "precompute_linear",
    "precompute",
    "remove_redundant_layers",
    "reorder",
    "torch1d_input_tensor_to_grouped_strings",
    "factory",
    "translate",
    "prepare_for_training",
]
