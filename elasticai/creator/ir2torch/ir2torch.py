import warnings
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import torch.nn as nn
from torch import fx

import elasticai.creator.function_dispatch as FD
import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir import Implementation, LoweringPass
from elasticai.creator.translation_pass_plugin import TranslationPass

from .default_handlers import handlers


class DataGraph(ir.DataGraph[ir.Node, ir.Edge], Protocol):
    @property
    def type(self) -> str: ...


class _DataGraph(ir.DataGraphImpl[ir.Node, ir.Edge]):
    @property
    def type(self) -> str:
        result = self.attributes.get("type", "<none>")
        if not isinstance(result, str):
            raise TypeError("expected 'type' field to be of type str")
        return result


def _new_graph(
    factory: ir.NodeEdgeFactory[ir.Node, ir.Edge], attributes: ir.AttributeMapping
) -> DataGraph:
    return _DataGraph(
        factory,
        attributes,
        ir.GraphImpl(lambda: ir.AttributeMapping()),
        ir.AttributeMapping(),
    )


class IrFactory(ir.IrFactory[ir.Node, ir.Edge, ir.DataGraph]):
    def __init__(self):
        self._node_edge = ir.DefaultNodeEdgeFactory()

        def graph(attributes):
            return _new_graph(self, attributes)

        self._graph = graph

    def node(
        self, name: str, attributes: ir.AttributeMapping = ir.AttributeMapping()
    ) -> ir.Node:
        return self._node_edge.node(name, attributes)

    def edge(
        self,
        src: str,
        dst: str,
        attributes: ir.AttributeMapping = ir.AttributeMapping(),
    ) -> ir.Edge:
        return self._node_edge.edge(src, dst, attributes)

    def graph(
        self, attributes: ir.AttributeMapping = ir.AttributeMapping()
    ) -> DataGraph:
        return self._graph(attributes)


type TypeHandler = Callable[[DataGraph], nn.Module]


class Ir2TorchTranslationPass(TranslationPass[[DataGraph], nn.Module]):
    @FD.dispatch_method(str)
    def _build_submodule(
        self,
        fn: TypeHandler,
        dgraph: DataGraph,
        /,
    ) -> nn.Module:
        return fn(dgraph)

    @_build_submodule.key_from_args
    def _get_key_from_data_graph(self, dgraph: DataGraph) -> str:
        return dgraph.type

    @staticmethod
    def _check_and_get_name_from_fn(name, fn) -> str:
        if name is None:
            if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
                return fn.__name__
            else:
                raise TypeError(
                    "If the registered type handler is not a function, you need to specify the type name explicitly"
                )
        return name

    @FD.registrar_method
    def register(
        self,
        name: str | None,
        fn: TypeHandler,
    ) -> TypeHandler:
        return self._build_submodule.register(
            self._check_and_get_name_from_fn(name, fn), fn
        )

    @FD.registrar_method
    def override(
        self,
        name: str | None,
        fn: TypeHandler,
    ) -> TypeHandler:
        return self._build_submodule.override(
            self._check_and_get_name_from_fn(name, fn), fn
        )

    def __call__(
        self,
        ir_root: DataGraph,
        registry: ir.Registry[DataGraph],
        state_dict: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Rebuild the original pytorch model from a given IR.

        Implemenation names containing dots will result in the corresponding modules sorted
        into a corresponding object hierarchy. E.g., for the implementation
        name `'top.linear'` we will create a pytorch container module under the name
        `'top'` and add the linear layer to it under the name `'linear'`. Note that this
        is an implementation detail of Ir2Torch and not a semantic meaning assigned to
        the `'.'` character.

        :param: `ir`: You need to make sure that `Ir2Torch` has a type handler for each implementation in `ir`
        :param: `state_dict`: You can optionally pass a state dict. This should be a state dict created
            from the original model via `nn.Module.state_dict`. As the `Torch2Ir` stage got rid of all
            duplicate submodules, we will strip all unknown keys from the `state_dict` and then load it.
        """
        root_module = nn.Module()
        for name, impl in registry.items():
            modules = list(self._build_submodule(m) for m in registry.values())
            assert len(modules) == 1
            last_parent = root_module
            while "." in name:
                parent_name, name = name.rsplit(".", 1)
                last_children = dict(last_parent.named_children())
                if parent_name not in last_children:
                    current_parent = nn.Module()
                    last_parent.add_module(parent_name, current_parent)
                else:
                    current_parent = last_children[parent_name]
                last_parent = current_parent
            last_parent.add_module(name, modules[0])

        graph = fx.Graph()
        nodes: dict[str, fx.Node] = {}

        def _add_node(ir_node: str) -> None:
            node_type = ir_root.nodes[ir_node].type
            node_impl: str = ir_root.nodes[ir_node].attributes["implementation"]
            if ir_root.nodes[ir_node].type not in ("input", "output"):
                predecessors = tuple(
                    nodes[node] for node in ir_root.predecessors[ir_node]
                )
                node = graph.create_node(
                    op="call_module",
                    target=node_impl,
                    args=predecessors,
                    name=ir_node,
                )
                nodes[ir_node] = node
            elif node_type == "input":
                n = graph.create_node(op="placeholder", name=ir_node, target=ir_node)
                nodes[ir_node] = n
            elif node_type == "output":
                predecessors = tuple(
                    nodes[node] for node in ir_root.predecessors[ir_node]
                )
                n = graph.create_node(
                    op="output",
                    target=ir_node,
                    args=predecessors,
                )
                nodes[ir_node] = n

        def visit_node(ir_node: str):
            if ir_node not in nodes:
                for pred in ir_root.predecessors[ir_node]:
                    visit_node(pred)
                if ir_node not in nodes:
                    _add_node(ir_node)

        visit_node("output")

        module = fx.GraphModule(root_module, graph)

        if state_dict is not None:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                submodule = ".".join(key.split(".")[:-1])
                if submodule in registry:
                    filtered_state_dict[key] = value
            module.load_state_dict(filtered_state_dict)
        return module


class Ir2Torch(LoweringPass[Implementation, nn.Module]):
    def convert(
        self, ir: Iterable[Implementation], state_dict: dict[str, Any] | None = None
    ) -> nn.Module:
        """Rebuild the original pytorch model from a given IR.

        Implemenation names containing dots will result in the corresponding modules sorted
        into a corresponding object hierarchy. E.g., for the implementation
        name `'top.linear'` we will create a pytorch container module under the name
        `'top'` and add the linear layer to it under the name `'linear'`. Note that this
        is an implementation detail of Ir2Torch and not a semantic meaning assigned to
        the `'.'` character.

        :param: `ir`: You need to make sure that `Ir2Torch` has a type handler for each implementation in `ir`
        :param: `state_dict`: You can optionally pass a state dict. This should be a state dict created
            from the original model via `nn.Module.state_dict`. As the `Torch2Ir` stage got rid of all
            duplicate submodules, we will strip all unknown keys from the `state_dict` and then load it.
        """
        warnings.warn(
            "You are using the deprected Ir2Torch lowering pass. Use the Ir2TorchTranslationPass instead",
            stacklevel=2,
        )
        root_module = nn.Module()
        root = ""
        _ir = {impl.name: impl for impl in ir}
        for impl in _ir.values():
            if impl.type != "module":
                modules = list(self([impl]))
                assert len(modules) == 1
                name = impl.name
                last_parent = root_module
                while "." in name:
                    parent_name, name = name.rsplit(".", 1)
                    last_children = dict(last_parent.named_children())
                    if parent_name not in last_children:
                        current_parent = nn.Module()
                        last_parent.add_module(parent_name, current_parent)
                    else:
                        current_parent = last_children[parent_name]
                    last_parent = current_parent
                last_parent.add_module(name, modules[0])

            else:
                root = impl.name

        graph = fx.Graph()
        ir_root = _ir[root]
        nodes: dict[str, fx.Node] = {}

        def _add_node(ir_node) -> None:
            if ir_node.type not in ("input", "output"):
                predecessors = tuple(
                    nodes[node] for node in ir_root.predecessors(ir_node)
                )
                node = graph.create_node(
                    op="call_module",
                    target=ir_node.data["implementation"],
                    args=predecessors,
                    name=ir_node.name,
                )
                nodes[ir_node.name] = node
            elif ir_node.type == "input":
                n = graph.create_node(
                    op="placeholder", name=ir_node.name, target=ir_node.name
                )
                nodes[ir_node.name] = n
            elif ir_node.type == "output":
                predecessors = tuple(
                    nodes[node] for node in ir_root.predecessors(ir_node)
                )
                n = graph.create_node(
                    op="output",
                    target=ir_node.name,
                    args=predecessors,
                )
                nodes[ir_node.name] = n

        def visit_node(ir_node):
            if ir_node.name not in nodes:
                for pred in ir_root.predecessors(ir_node).values():
                    visit_node(pred)
                if ir_node.name not in nodes:
                    _add_node(ir_node)

        visit_node(ir_root.nodes["output"])

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
            if hasattr(handler, "__name__") and isinstance(handler.__name__, str):
                self.register(handler.__name__)(handler)
            else:
                raise TypeError(
                    f"unsupported type handler type, expected a function type, but found {type(handler)}"
                )


def get_default_converter() -> Ir2Torch:
    converter = Ir2Torch()
    converter.register_type_handlers(handlers)
    return converter
