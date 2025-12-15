from collections.abc import Callable
from typing import TypeAlias

import elasticai.creator.function_dispatch as F
import elasticai.creator.plugin as pl
from elasticai.creator.ir import LoweringPass as _LoweringPass

from .convolution import DummyLowerable

LoweringPass: TypeAlias = _LoweringPass[DummyLowerable, str]
LoweringFn: TypeAlias = Callable[[DummyLowerable], str]
SymbolFn: TypeAlias = pl.PluginSymbolFn[LoweringPass, [DummyLowerable], str]


@F.registrar
def type_handler(name: str | None, fn: LoweringFn) -> SymbolFn:
    if name is None:
        name = fn.__name__

    def load_into(lower: LoweringPass) -> None:
        lower.register(name)(fn)

    return pl.make_plugin_symbol(load_into, fn)


@type_handler()
def lowering_pass_conv(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
