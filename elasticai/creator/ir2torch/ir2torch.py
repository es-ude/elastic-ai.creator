from collections.abc import Callable, Iterable

import torch.nn as nn
from torch import fx

from elasticai.creator.function_utils import KeyedFunctionDispatcher
from elasticai.creator.torch2ir import Implementation

from .default_handlers import handlers


class Ir2Torch:
    def __init__(self) -> None:
        def key_fn(n: Implementation) -> str:
            return n.type

        self._handle_type: KeyedFunctionDispatcher[Implementation, nn.Module] = (
            KeyedFunctionDispatcher(key_fn)
        )

    def convert(self, ir: dict[str, Implementation]) -> nn.Module:
        root = nn.Module()
        for impl in ir.values():
            if impl.name != "root":
                root.add_module(impl.name, self._handle_type(impl))

        graph = fx.Graph()
        ir_root = ir["root"]
        for ir_node in ir_root.nodes.values():
            if ir_node.type not in ("input", "output"):
                graph.call_module(
                    ir_node.implementation, tuple(ir_root.predecessors(ir_node))
                )

        return fx.GraphModule(root, graph)

    def register_type_handlers(
        self, handlers: Iterable[Callable[[Implementation], nn.Module]]
    ) -> None:
        for handler in handlers:
            self._handle_type.register(handler.__name__, handler)

    @classmethod
    def get_default_converter(cls) -> "Ir2Torch":
        converter = cls()
        converter.register_type_handlers(handlers)
        return converter
