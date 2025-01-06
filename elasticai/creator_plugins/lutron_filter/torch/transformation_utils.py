from typing import Any, Callable, Dict, Iterable, Tuple, cast

import torch
from torch import Tensor
from torch.fx import Graph, GraphModule, Interpreter, Node, Tracer
from torch.fx.node import Argument
from torch.nn import Module

from elasticai.creator.ir.helpers import FilterParameters

from .lutron_module_protocol import LutronModule


class LeafTracer(Tracer):
    def __init__(self, classes: tuple[type | str, ...]):
        self.classes = classes
        super().__init__()

    @staticmethod
    def _check_class(m: torch.nn.Module, cls: type | str):
        if isinstance(cls, type):
            return isinstance(m, cls)
        elif isinstance(cls, str):
            return m.__class__.__qualname__.endswith(cls)
        return False

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if any(self._check_class(m, c) for c in self.classes):
            return True
        else:
            return super().is_leaf_module(m, module_qualified_name)


class LutronTracer(LeafTracer):
    def __init__(self):
        super().__init__((LutronModule,))


class CallSequenceRecorder(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.call_sequence = []

    def call_module(
        self, target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        result = self.module.get_submodule(cast(str, target))(*args, **kwargs)
        if not isinstance(args[0], Tensor):
            raise ValueError("Expected first argument to be a tensor")
        self.call_sequence.append((target, args[0].shape))
        return result


def get_module_for_node(n: Node, root: Module):
    return root.get_submodule(cast(str, n.target))


def node_matches(root: Module, n: Node, expression: type | Callable | str) -> bool:
    if n.op == "call_module":
        module = root.get_submodule(cast(str, n.target))
    else:
        module = None
    if isinstance(expression, type):
        return isinstance(module, expression)
    elif callable(expression):
        return n.op == "call_function" and n.target == expression
    elif isinstance(expression, str):
        class_name = module.__class__.__qualname__
        return class_name.endswith(expression)
    return False


def nodes_for_types_while_sequential(
    root: Module, current_node: Node, types: Iterable[type | Callable | str]
):
    for current_type in types:
        if node_matches(root, current_node, current_type):
            yield current_node
        else:
            break
        users = tuple(current_node.users.keys())
        if len(users) == 1:
            current_node = users[0]
        else:
            break


def get_filter_parameters_conv(layers):
    layer = layers[0]
    return FilterParameters(
        in_channels=layer.in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size[0],
        groups=layer.groups,
        stride=layer.stride[0],
    )


def get_filter_parameters_mpool(layers):
    return FilterParameters(
        in_channels=1,
        out_channels=1,
        kernel_size=layers[0].kernel_size,
        groups=1,
        stride=layers[0].stride,
    )


def get_filter_parameters_linear(layers):
    m = layers[0]
    return FilterParameters(
        input_size=m.in_features,
        kernel_size=0,
        in_channels=0,
        out_channels=m.out_features,
        stride=1,
        groups=1,
    )


def move_to_submodule(
    m: Module,
    g: Graph,
    node_names: tuple[str, ...],
    name: str,
    wrapper: Callable[[Module], Module] = lambda m: m,
) -> tuple[GraphModule, Node]:
    """Move a sequence of nodes to a sub graph and add it as a submodule to the original module, replacing the original
    call sequence.

    args:
        - m: The original module to alter
        - g: The original graph, will be altered as well
        - node_names: The sequence of names corresponding to the sequence of called nodes, we want to replace.
          IMPORTANT: If this is not a valid connected sequence in the original graph, the behaviour is undefined.
        - name: The name to use for the new node and new submodule
        - wrapper: A wrapper function to call with the newly created GraphModule.
            Use this, if you need to
            apply additional modifications to the module, e.g., add members to the object.

    returns:
        - the transformed graph module and the newly added node
    """
    m = GraphModule(m, g)
    sub_g = Graph()
    nodes = {n.name: n for n in g.nodes}
    replaced_by_x = nodes[node_names[0]].all_input_nodes[0]
    x = sub_g.placeholder("x")
    args: dict[Node, Node] = {}

    def arg_transform(n):
        if n.name == replaced_by_x.name:
            return x
        else:
            return args[n]

    latest_node = None
    for n in g.nodes:
        if n.name in node_names:
            args[n] = sub_g.node_copy(n, arg_transform)
            latest_node = n
    if latest_node is not None:
        sub_g.output(args[latest_node])
    else:
        sub_g.output("x")
    sub_gm = GraphModule(m, sub_g)
    m.add_module(name=name, module=wrapper(sub_gm))
    input_tensor = nodes[node_names[0]].args[0]
    new_node = g.call_module(module_name=name, args=(input_tensor,))
    replaced_by_x.append(new_node)
    last_old_node = nodes[node_names[-1]]
    last_old_node.replace_all_uses_with(new_node)
    g.eliminate_dead_code()

    gm = GraphModule(m, g)
    return gm, new_node
