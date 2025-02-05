from collections.abc import Callable, Iterable, Iterator
from typing import Any

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

    def convert(
        self, ir: Iterator[Implementation], state_dict: dict[str, Any] | None = None
    ) -> nn.Module:
        """Rebuild the original pytorch model from a given IR.

        :param: `ir`: You need to make sure that `Ir2Torch` has a type handler for each implementation in `ir`
        :param: `state_dict`: You can optionally pass a state dict. This should be a state dict created
            from the original model via `nn.Module.state_dict`. As the `Torch2Ir` stage got rid of all
            duplicate submodules, we will strip all unknown keys from the `state_dict` and then load it.
        """
        root = nn.Module()
        _ir = dict((impl.name, impl) for impl in ir)
        for impl in _ir.values():
            if impl.name != "":
                root.add_module(impl.name, self._handle_type(impl))

        graph = fx.Graph()
        ir_root = _ir[""]
        for ir_node in ir_root.nodes.values():
            if ir_node.type not in ("input", "output"):
                graph.call_module(
                    ir_node.implementation, tuple(ir_root.predecessors(ir_node))
                )
        module = fx.GraphModule(root, graph)

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
            self._handle_type.register(handler.__name__, handler)

    @classmethod
    def get_default_converter(cls) -> "Ir2Torch":
        converter = cls()
        converter.register_type_handlers(handlers)
        return converter
