from collections.abc import Callable, Iterable
from typing import Any

import torch.nn as nn
from torch import fx

from elasticai.creator.ir import LoweringPass
from elasticai.creator.torch2ir import Implementation

from .default_handlers import handlers


class Ir2Torch(LoweringPass[Implementation, nn.Module]):
    def convert(
        self, ir: Iterable[Implementation], state_dict: dict[str, Any] | None = None
    ) -> nn.Module:
        """Rebuild the original pytorch model from a given IR.

        :param: `ir`: You need to make sure that `Ir2Torch` has a type handler for each implementation in `ir`
        :param: `state_dict`: You can optionally pass a state dict. This should be a state dict created
            from the original model via `nn.Module.state_dict`. As the `Torch2Ir` stage got rid of all
            duplicate submodules, we will strip all unknown keys from the `state_dict` and then load it.
        """
        root_module = nn.Module()
        root = ""
        _ir = {impl.name: impl for impl in ir}
        for impl in _ir.values():
            if impl.type != "module":
                modules = list(self([impl]))
                assert len(modules) == 1
                root_module.add_module(impl.name, modules[0])
            else:
                root = impl.name

        graph = fx.Graph()
        ir_root = _ir[root]
        for ir_node in ir_root.nodes.values():
            if ir_node.type not in ("input", "output"):
                graph.call_module(
                    ir_node.implementation, tuple(ir_root.predecessors(ir_node))
                )

        module = fx.GraphModule(root_module, graph)

        if state_dict is not None:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                submodule = ".".join(key.split(".")[:-1])
                if submodule in _ir:
                    filtered_state_dict[key] = value
            module.load_state_dict(filtered_state_dict)
        return module

    def register_type_handlers(
        self, handlers: Iterable[Callable[[Implementation], nn.Module]]
    ) -> None:
        for handler in handlers:
            self.register(handler.__name__)(handler)


def get_default_converter() -> Ir2Torch:
    converter = Ir2Torch()
    converter.register_type_handlers(handlers)
    return converter
