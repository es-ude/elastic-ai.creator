from collections.abc import Callable
from typing import cast

from torch.fx import Node as FxNode
from torch.fx import Tracer
from torch.nn import Module

import elasticai.creator.function_dispatch as FD
import elasticai.creator.ir as ir

from .default_handlers import handlers as default_handlers


class _DefaultTracer(Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        if type(m).__qualname__.startswith("elasticai"):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class LoweringError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


type DataGraph = ir.DataGraph[ir.Node, ir.Edge]
type Registry = ir.Registry[DataGraph]
type TypeHandler = Callable[[Module], dict]


class Torch2Ir:
    def __init__(self, tracer=_DefaultTracer()) -> None:
        self._tracer = tracer
        self._ir_factory = ir.DefaultIrFactory()
        self._registry: Registry = ir.Registry()
        self._root = self._ir_factory.graph(ir.attribute(type="module"))

    @FD.dispatch_method(str)
    def _extractors(
        self,
        fn: TypeHandler,
        module: Module,
    ) -> dict:
        return fn(module)

    @_extractors.key_from_args
    def _get_type_from_module(self, module: Module) -> str:
        return module.__class__.__name__.lower()

    @staticmethod
    def _check_and_get_name(name: str | None, fn: TypeHandler) -> str:
        if name is None:
            if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
                return fn.__name__
            else:
                raise TypeError(
                    "specify the type handler's type explicitly if you want to register a non-function callable"
                )
        return name

    @FD.registrar_method
    def register(self, key: str | None, fn: TypeHandler) -> TypeHandler:
        key = self._check_and_get_name(key, fn)
        self._extractors.register(key, fn)
        return fn

    def _handle_fx_node(self, node: FxNode) -> None:
        self._root = self._root.add_node(
            node.name,
            ir.attribute(
                type=self._get_type(node), implementation=self._get_implementation(node)
            ),
        )
        ir_node = self._root.nodes[node.name]
        impl = ir_node.attributes["implementation"]
        if impl not in self._registry and impl not in ("input", "output"):
            attributes = self._extract_attributes(node)
            self._registry = self._registry | {
                impl: self._ir_factory.graph(
                    ir.attribute(type=ir_node.type, **attributes),
                )
            }
        for successor in self._get_successors(node):
            self._root = self._root.add_edge(node.name, successor.name)

    def convert(self, model: Module) -> tuple[DataGraph, Registry]:
        self.model = model
        torch_graph = self._tracer.trace(model)
        registry: Registry = ir.Registry()
        for node in torch_graph.nodes:
            self._handle_fx_node(node)
        registry = self._registry
        self._registry = ir.Registry()
        root = self._root
        self._root = self._ir_factory.graph(ir.attribute(type="module"))
        return root, registry

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

    def _extract_attributes(self, node: FxNode) -> dict:
        if self._get_type(node) in ("input", "output"):
            return {}
        module = self.model.get_submodule(cast(str, node.target))
        return self._extractors(module)

    def __call__(self, model: Module) -> tuple[DataGraph, Registry]:
        return self.convert(model)


def get_default_converter() -> Torch2Ir:
    converter = Torch2Ir()
    for handler in default_handlers:
        converter.register()(handler)
    return converter
