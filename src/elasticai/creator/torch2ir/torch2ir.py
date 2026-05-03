from collections.abc import Callable
from typing import Any, cast

from torch.fx import Node as FxNode
from torch.fx import Tracer
from torch.nn import Module

import elasticai.creator.function_dispatch as FD
import elasticai.creator.ir as ir
from elasticai.creator.ir2torch import DataGraph as DataGraph
from elasticai.creator.ir2torch import IrFactory

from .default_handlers import handlers as default_handlers


class _DefaultTracer(Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        if type(m).__qualname__.startswith("elasticai"):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class LoweringError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


type TypeHandler = Callable[[Module], dict]


class Torch2Ir:
    def __init__(self, tracer=_DefaultTracer()) -> None:
        self._tracer = tracer
        self._ir_factory = IrFactory()
        self._registry: ir.Registry[DataGraph] = ir.Registry()
        self._root = self._ir_factory.graph(ir.attribute(type="module"))

    @FD.dispatch_method()
    def _extractors(
        self,
        fn: TypeHandler,
        module: Module,
    ) -> dict:
        return fn(module)

    @_extractors.key_from_args
    def _get_type_from_module(self, module: Module) -> str:
        return module.__class__.__name__.lower()

    @_extractors.default_register
    def register(self, _: str | None, fn: TypeHandler) -> TypeHandler:
        return fn

    @_extractors.default_override
    def override(self, _: str | None, fn: TypeHandler) -> TypeHandler:
        return fn

    def _handle_fx_node(self, node: FxNode) -> None:
        attribute = ir.attribute(type=self._get_type(node))
        if self._has_implementation(node):
            attribute |= dict(implementation=self._get_implementation(node))

        self._root = self._root.add_node(node.name, attribute)
        ir_node = self._root.nodes[node.name]
        if self._has_implementation(node):
            impl = ir_node.attributes["implementation"]
            if impl not in self._registry and ir_node.type != "function":
                attributes = self._extract_attributes(node)
                self._registry = self._registry | {
                    impl: self._ir_factory.graph(
                        ir.attribute(type=ir_node.type, **attributes),
                    )
                }
        for successor in self._get_successors(node):
            self._root = self._root.add_edge(node.name, successor.name)

    def convert(self, model: Module) -> tuple[DataGraph, ir.Registry[DataGraph]]:
        self.model = model
        torch_graph = self._tracer.trace(model)
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
        error = LoweringError(f"""
                              Handling {node.name} failed. 
                              For function calls only add is supported.
                              If you use pytorch functionals, please swap them for pytorch modules.
                              Getting attributes is not supported.""")
        match node.op:
            case "call_module":
                return type(
                    self.model.get_submodule(cast(str, node.target))
                ).__name__.lower()

            case "call_function":
                return "function"
            case "placeholder":
                return "input"
            case "output":
                return "output"
            case "get_attr":
                raise error
            case _:
                raise Exception(f"Unknown node type: {node.op}")

    def _has_implementation(self, node: FxNode) -> bool:
        return self._get_type(node) not in ("input", "output")

    def _get_implementation(self, node: FxNode) -> str:
        match node.op:
            case "call_module":
                return cast(str, node.target)
            case "call_function":
                return node.target.__name__  # type: ignore
            case _:
                raise ValueError(
                    f"unsupported operation {node.op} for node {node.name}"
                )

    def _extract_attributes(self, node: FxNode) -> dict:
        if self._get_type(node) in ("input", "output", "function"):
            return {}
        module = self.model.get_submodule(cast(str, node.target))
        return self._extractors(module)

    def __call__(self, model: Module) -> tuple[DataGraph, ir.Registry[DataGraph]]:
        return self.convert(model)


def nested_list_to_tuple(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(nested_list_to_tuple(x) for x in obj)
    return obj


class Torch2IrWithParams(Torch2Ir):
    def _add_params(self, module) -> dict:
        params = {}
        for name, param in module.named_parameters():
            params[name] = nested_list_to_tuple(param.data.detach().numpy().tolist())
        return params

    def _extract_attributes(self, node: FxNode) -> dict:
        if self._get_type(node) in ("input", "output", "function"):
            return {}
        module = self.model.get_submodule(cast(str, node.target))
        return self._extractors(module) | self._add_params(module)


class Torch2IrWithParamsAndBuffers(Torch2IrWithParams):
    def _add_buffers(self, module: Module) -> dict:
        buffers = {}
        for name, buffer in module.named_parameters():
            buffers[name] = nested_list_to_tuple(buffer.data.detach().numpy().tolist())
        return buffers

    def _extract_attributes(self, node: FxNode) -> dict:
        if self._get_type(node) in ("input", "output", "function"):
            return {}
        module = self.model.get_submodule(cast(str, node.target))
        return (
            self._extractors(module)
            | self._add_params(module)
            | self._add_buffers(module)
        )


def get_default_converter() -> Torch2Ir:
    converter = Torch2Ir()
    for handler in default_handlers:
        converter.register()(handler)
    return converter
