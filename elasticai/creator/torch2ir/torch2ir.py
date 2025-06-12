from collections.abc import Callable, Iterable, Iterator
from typing import cast

from torch.fx import Node as FxNode
from torch.fx import Tracer
from torch.nn import Module

from elasticai.creator.function_utils import KeyedFunctionDispatcher
from elasticai.creator.graph import BaseGraph

from .core import Edge, Implementation, new_node
from .default_handlers import handlers as default_handlers


class _DefaultTracer(Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        if type(m).__qualname__.startswith("elasticai"):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class LoweringError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Torch2Ir:
    def __init__(self, tracer: Tracer = _DefaultTracer()):
        super().__init__()
        self._tracer = tracer
        self._registry: dict[str, Implementation] = {}
        self._root = Implementation(graph=BaseGraph())
        self._root.type = "module"
        self._root.name = ""
        self._registry[""] = self._root
        self._extractors: KeyedFunctionDispatcher[Module, dict] = (
            KeyedFunctionDispatcher(self._get_module_key)
        )

    def register(
        self, module_type: str, handler: Callable[[Module], dict]
    ) -> Callable[[Module], dict]:
        """The handlers are used to extract the attributes of the module"""
        self._extractors.register(module_type, handler)
        return handler

    def register_handlers(
        self, handlers: Iterable[Callable[[Module], dict]]
    ) -> "Torch2Ir":
        for handler in handlers:
            self.register(handler.__name__, handler)
        return self

    @staticmethod
    def _get_module_key(module: Module) -> str:
        return module.__class__.__name__.lower()

    def convert(self, model: Module) -> Iterator[Implementation]:
        self.model = model
        torch_graph = self._tracer.trace(model)
        for node in torch_graph.nodes:
            self._handle_fx_node(node)
        registry = self._registry
        self._registry = {}
        yield from registry.values()

    def __call__(self, model: Module) -> Iterator[Implementation]:
        yield from self.convert(model)

    def _get_successors(self, node: FxNode) -> list[FxNode]:
        return list(node.users)

    def _get_type(self, node: FxNode) -> str:
        error = LoweringError("""
                              Cannot handle function calls or getting 
                              attributes during translation. Please use 
                              supported Modules to design your model
                              and change every function call to a
                            module call.""")
        match node.op:
            case "call_module":
                return type(
                    self.model.get_submodule(cast(str, node.target))
                ).__name__.lower()

            case "call_function":
                raise error
            case "placeholder":
                return "input"
            case "output":
                return "output"
            case "get_attr":
                raise error
            case _:
                raise Exception(f"Unknown node type: {node.op}")

    def _get_implementation(self, node: FxNode) -> str:
        match self._get_type(node):
            case "input":
                return "input"
            case "output":
                return "output"
            case _:
                return cast(str, node.target)

    def _handle_fx_node(self, node: FxNode) -> None:
        ir_node = new_node(
            name=node.name,
            type=self._get_type(node),
            implementation=self._get_implementation(node),
            attributes={},
        )
        self._root.add_node(ir_node)
        impl = ir_node.implementation
        if impl not in self._registry and impl not in ("input", "output"):
            self._registry[impl] = Implementation(
                name=impl,
                type=ir_node.type,
                graph=BaseGraph(),
                data=dict(**self._extract_attributes(node)),
            )

        for successor in self._get_successors(node):
            self._root.add_edge(Edge(src=node.name, dst=successor.name, data={}))

    def _extract_attributes(self, node: FxNode) -> dict:
        if self._get_type(node) in ("input", "output"):
            return {}
        module = self.model.get_submodule(cast(str, node.target))
        return self._extractors(module)


def get_default_converter() -> Torch2Ir:
    converter = Torch2Ir()
    converter.register_handlers(default_handlers)
    return converter
