from collections.abc import Callable, Iterable

import torch.nn as nn
from torch import fx

from elasticai.creator.ir import LoweringPass
from elasticai.creator.torch2ir import Implementation

from .default_handlers import handlers


class Ir2Torch(LoweringPass[Implementation, nn.Module]):
    def convert(self, ir: dict[str, Implementation]) -> nn.Module:
        root_module = nn.Module()
        root = ""
        for impl in ir.values():
            if impl.type != "module":
                modules = list(self([impl]))
                assert len(modules) == 1
                root_module.add_module(impl.name, modules[0])
            else:
                root = impl.name

        graph = fx.Graph()
        ir_root = ir[root]
        for ir_node in ir_root.nodes.values():
            if ir_node.type not in ("input", "output"):
                graph.call_module(
                    ir_node.implementation, tuple(ir_root.predecessors(ir_node))
                )

        return fx.GraphModule(root_module, graph)

    def register_type_handlers(
        self, handlers: Iterable[Callable[[Implementation], nn.Module]]
    ) -> None:
        for handler in handlers:
            self.register(handler.__name__)(handler)


def get_default_converter() -> Ir2Torch:
    converter = Ir2Torch()
    converter.register_type_handlers(handlers)
    return converter
