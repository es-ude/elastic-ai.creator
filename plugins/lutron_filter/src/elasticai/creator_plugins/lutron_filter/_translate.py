import json
import logging
from collections.abc import Callable
from pathlib import Path
from pprint import pformat

import torch.nn as tnn

import elasticai.creator.function_dispatch as FD
import elasticai.creator.ir as ir
from elasticai.creator import ir2vhdl as ir2vhdl
from elasticai.creator.hdl_ir import Shape
from elasticai.creator.ir2torch import DataGraph, get_default_converter
from elasticai.creator.ir2vhdl import Ir2Vhdl
from elasticai.creator_plugins.grouped_filter import FilterParameters, grouped_filter
from elasticai.creator_plugins.time_multiplexed_sequential.src import (
    network,
    sequential,
)

from .nn import Binarize as _Binarize
from .rules import binarize_activations, create_shape_inference
from .rules import ir_factory as _factory
from .rules import (
    make_add_bitwidths_rule as _make_add_bitwidths_rule,
)
from .rules import make_split_conv_rule as _make_split_conv_rule
from .rules import (
    precompute as _precompute,
)
from .rules import (
    reorder as _reorder,
)
from .torch2ir import get_default_torch2ir, get_torch2ir_without_params

_convert_torch2ir_for_precomp = get_default_torch2ir()
_convert_torch2ir_for_preparation = get_torch2ir_without_params()
_convert_ir2torch = get_default_converter()


@_convert_ir2torch.register()
def binarize(_):
    return _Binarize()


@_convert_ir2torch.register()
def prelu(_):
    return tnn.PReLU()


@_convert_ir2torch.register()
def batchnorm1d(original: DataGraph):
    attr = original.attributes
    return tnn.BatchNorm1d(
        num_features=attr.get_int("num_features"),
        affine=attr.get_bool("affine", default=True),
    )


def translate(
    model: tnn.Module,
    input_shape: Shape,
    num_input_bits: int,
    build_dir: Path,
    convert_to_vhdl: Ir2Vhdl | None = None,
) -> None:
    """Translate a precomputable module to hardware and store intermediate results."""
    if convert_to_vhdl is None:
        convert_to_vhdl = Ir2Vhdl()
    loader = ir2vhdl.PluginLoader(convert_to_vhdl)

    for p in [
        "shift_register",
        "combinatorial",
        "skeleton",
        "lutron",
    ]:
        loader.load_from_package(p)
    logger = logging.getLogger(__name__)
    torch_graph, torch_reg = _convert_torch2ir_for_precomp(model)
    ir_rep = _factory.graph_from_other(torch_graph), _factory.registry(torch_reg)
    rule = ir.compose_rules(
        _reorder, _make_add_bitwidths_rule(num_input_bits), _precompute
    )
    g, reg = rule(*ir_rep)
    reg = reg.add("network", g)
    vhd_reg = ir2vhdl.factory.registry(reg)
    vhd_reg = vhd_reg.add(
        "network",
        create_shape_inference()(reg["network"], input_shape, num_input_bits),
    )
    vhd_reg = clean_registry_from_leftover_dgraphs("network", vhd_reg)
    pretty_log_debug(vhd_reg["network"])

    serializer = ir.IrSerializer()
    hl_dir = build_dir / "hl"
    hl_dir.mkdir(exist_ok=True)
    for name, g in reg.items():
        serialized = serializer.serialize(g)
        with open(hl_dir / f"{name}.json", "w") as f:
            json.dump(serialized, f, indent=1)

    reg = to_low_level(vhd_reg)
    ll_dir = build_dir / "ll"
    ll_dir.mkdir(exist_ok=True)
    for name, g in reg.items():
        serialized = serializer.serialize(g)
        with open(ll_dir / f"{name}.json", "w") as f:
            json.dump(serialized, f, indent=1)
    for name, g in reg.items():
        logger.debug(name)
    code = convert_to_vhdl(reg)
    for name, lines in code:
        with open(build_dir / f"{name}", "w") as f:
            for line in lines:
                f.write(line)
                f.write("\n")


type _Handler = Callable[
    [ir2vhdl.DataGraph, ir.Registry[ir2vhdl.DataGraph]],
    tuple[ir2vhdl.DataGraph, ir.Registry[ir2vhdl.DataGraph]],
]


class LowLevelIRTranslator:
    @FD.dispatch_method()
    def _call_handler(
        self, fn: _Handler, g: ir2vhdl.DataGraph, reg: ir.Registry[ir2vhdl.DataGraph]
    ) -> tuple[ir2vhdl.DataGraph, ir.Registry[ir2vhdl.DataGraph]]:
        g, reg = fn(g, reg)
        return g, reg

    @_call_handler.key_from_args
    def _key_from_args(
        self, g: ir2vhdl.DataGraph, reg: ir.Registry[ir2vhdl.DataGraph]
    ) -> str:
        return g.type

    @_call_handler.default_register
    def register(self, _: str | None, handler: _Handler, /) -> _Handler:
        return handler

    def __call__(
        self, reg: ir.Registry[ir2vhdl.DataGraph]
    ) -> ir.Registry[ir2vhdl.DataGraph]:
        logger = logging.getLogger(__name__)
        new_reg = ir2vhdl.factory.registry(reg)
        result_reg = ir2vhdl.factory.registry()
        for name, g in reg.items():
            logger.debug(f"handling: {g.type} of name {name}")
            g, tmp_reg = self._call_handler(g, new_reg)
            tmp_reg = tmp_reg.add(name, g)
            logger.debug(f"registry: {[k for k in tmp_reg]}")
            result_reg = tmp_reg | result_reg
        return result_reg


def clean_registry_from_leftover_dgraphs[G: ir2vhdl.DataGraph](
    root_name: str, reg: ir.Registry[G]
) -> ir.Registry[G]:
    cleanedreg = ir2vhdl.collect_transitive_implementation_closure(root_name, reg)
    reg = cleanedreg | {k: v for k, v in reg.items() if v.type == "lutron"}

    reg = reg.add(
        root_name, reg[root_name].with_attributes(ir.attribute(type="network"))
    )
    return reg


to_low_level = LowLevelIRTranslator()


def lutron_handler(g, reg):
    return g, reg


_ = to_low_level.register()(network)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
_ = to_low_level.register()(sequential)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
_ = to_low_level.register()(grouped_filter)
_ = to_low_level.register("lutron")(lutron_handler)


def prepare_for_training(
    model: tnn.Module,
    handle_filter_params: Callable[
        [FilterParameters], tuple[FilterParameters, FilterParameters]
    ],
) -> tnn.Module:
    """binarize activations and split convolutions using handle_filter_params."""
    original, reg = _convert_torch2ir_for_preparation(model)
    ir_representation = (
        _factory.graph().with_data_from(original),
        reg.apply(lambda g: _factory.graph().with_data_from(g)),
    )

    split = _make_split_conv_rule(handle_filter_params)
    rule = ir.compose_rules(binarize_activations, split)
    g, new_reg = rule(*ir_representation)
    pretty_log_debug(g)
    for _g in new_reg.values():
        pretty_log_debug(_g)
    return _convert_ir2torch(g, new_reg.apply(original.with_data_from))


def pretty_log_debug(g):
    logger = logging.getLogger(__name__)
    serialized = ir.IrSerializer().serialize(g)
    logger.debug(pformat(serialized), stacklevel=2)
