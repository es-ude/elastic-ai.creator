from collections.abc import Callable
from itertools import starmap
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import fx

import elasticai.creator.function_dispatch as FD
from elasticai.creator import ir
from elasticai.creator.ir import StdIrFactory

from .default_handlers import dgraph_handlers, node_handlers


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


class IrFactory(StdIrFactory[ir.Node, ir.Edge, DataGraph]):
    def __init__(self) -> None:
        super().__init__(ir.NodeImpl, ir.EdgeImpl, _DataGraph)


type TypeHandler = Callable[[DataGraph], nn.Module]


class Ir2Torch:
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

    @FD.dispatch_method(str)
    def _make_fn_call(
        self, fn: Callable[[ir.Node], Callable[[Any], torch.Tensor]], node: ir.Node, /
    ) -> Callable[[Any], torch.Tensor]:
        return fn(node)

    @_make_fn_call.key_from_args
    def _key_for_fn_call(self, node: ir.Node) -> str:
        return node.attributes["implementation"]

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
    def register_node(
        self, name: str | None, fn: Callable[[ir.Node], Callable[[Any], torch.Tensor]]
    ):
        return self._make_fn_call.register(
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
        factory = IrFactory()

        root_module = nn.Module()

        def to_new_graph(name, graph):
            return name, factory.graph_from_other(graph)

        for name, impl in starmap(to_new_graph, registry.items()):
            layer = self._build_submodule(impl)
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
            last_parent.add_module(name, layer)

        graph = fx.Graph()
        nodes: dict[str, fx.Node] = {}

        def _add_node(ir_node: str) -> None:
            node = ir_root.nodes[ir_node]
            match node.type:
                case "input":
                    n = graph.create_node(
                        op="placeholder", name=node.name, target=node.name
                    )
                    nodes[node.name] = n
                case "output":
                    predecessors = tuple(
                        nodes[node] for node in ir_root.predecessors[node.name]
                    )
                    n = graph.create_node(
                        op="output",
                        target=ir_node,
                        args=predecessors,
                    )
                    nodes[ir_node] = n
                case _:
                    predecessors = tuple(
                        nodes[n] for n in ir_root.predecessors[ir_node]
                    )
                    if node.type == "function":
                        op = "call_function"
                        target = self._make_fn_call(node)
                    else:
                        op = "call_module"
                        target = node.attributes["implementation"]
                    fxnode = graph.create_node(
                        op=op,
                        target=target,
                        args=predecessors,
                        name=ir_node,
                    )
                    nodes[ir_node] = fxnode

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


def get_default_converter() -> Ir2Torch:
    converter = Ir2Torch()
    for handler in dgraph_handlers:
        converter.register()(handler)
    for name, handler in node_handlers.items():
        converter.register_node(name, handler)
    return converter
