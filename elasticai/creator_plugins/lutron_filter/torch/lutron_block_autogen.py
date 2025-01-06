from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Protocol,
    Tuple,
    TypeGuard,
    cast,
)

import torch
from torch.fx import Graph, GraphModule, Interpreter
from torch.fx.node import Target
from torch.nn import Module

from .lutron_block_detection import PatternMatch
from .lutron_module_protocol import LutronModule
from .lutron_modules.lutron_filter import (
    LutronConv,
    LutronLinear,
    LutronMaxPool,
)
from .transformation_utils import (
    get_filter_parameters_conv,
    get_filter_parameters_linear,
    get_filter_parameters_mpool,
    move_to_submodule,
)


def _target_is_string(t: Target) -> TypeGuard[str]:
    return isinstance(t, str)


class _ShapeRecorder(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.call_sequence = []

    def call_module(
        self, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if _target_is_string(target):
            m = self.module.get_submodule(target)
            input_shape = args[0].shape
            result = self.module.get_submodule(target)(*args, **kwargs)
            out_shape = result.shape
            if isinstance(m, LutronModule):
                m.infer_shape(input_shape, out_shape)

            self.call_sequence.append((target, args[0].shape))
            return result
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")


def create_pooling(m, layers_of_interest) -> Module:
    return LutronMaxPool(m, get_filter_parameters_mpool(layers_of_interest))


def create_conv(m, layers_of_interest) -> Module:
    return LutronConv(m, get_filter_parameters_conv(layers_of_interest))


def create_linear(m, layers_of_interest) -> Module:
    return LutronLinear(m, get_filter_parameters_linear(layers_of_interest))


def _is_lutron_module(m: Module) -> TypeGuard[LutronModule]:
    return isinstance(m, LutronModule)


def replace_by_lutron_module(
    root: Module,
    g: Graph,
    matches: Iterable[PatternMatch],
    create_lutron_module: Callable[[Module, tuple[Module, ...]], Module],
):
    m = root

    def create_wrapper(layers_of_interest) -> Callable[[Module], Module]:
        def wrapper(m: Module) -> Module:
            return create_lutron_module(m, layers_of_interest)

        return wrapper

    matched_modules = set()
    for match in matches:
        if set(match.matched_sequence).isdisjoint(matched_modules):
            matched_modules = matched_modules.union(set(match.matched_sequence))
            node_names = tuple(n.name for n in match.matched_sequence)
            nodes_of_interest = tuple(
                match.matched_sequence[i] for i in match.nodes_of_interest
            )
            layers_of_interest = tuple(
                m.get_submodule(cast(str, n.target)) for n in nodes_of_interest
            )
            primary_node = match.matched_sequence[match.nodes_of_interest[0]]
            name = f"{primary_node.name}_lutron_filter"
            m, _ = move_to_submodule(
                m, g, node_names, name, create_wrapper(layers_of_interest)
            )

    return m


class LutronBlockMatcher(Protocol):
    @abstractmethod
    def maxpool1d(self) -> Iterator[PatternMatch]: ...
    @abstractmethod
    def conv1d(self) -> Iterator[PatternMatch]: ...
    @abstractmethod
    def linear(self) -> Iterator[PatternMatch]: ...


def autogenerate_lutron_blocks(
    m: Module,
    input_shape: tuple[int, int, int],
    graph: Graph,
    block_matcher_factory: Callable[[Module, Graph], LutronBlockMatcher],
) -> GraphModule:
    """
    args:
      - input_shape: tuple (B, C, N)
          B: batch size
          C: num channels
          N: num sample points per channel (spatial size)
    """
    g = graph
    matcher = block_matcher_factory(m, g)
    mpooling_blocks = tuple(matcher.maxpool1d())
    conv_blocks = tuple(matcher.conv1d())
    lin_blocks = tuple(matcher.linear())

    m = replace_by_lutron_module(m, g, mpooling_blocks, create_pooling)
    m = replace_by_lutron_module(m, g, conv_blocks, create_conv)

    m = replace_by_lutron_module(m, g, lin_blocks, create_linear)
    gm = GraphModule(m, g)
    gm.eval()
    rec = _ShapeRecorder(
        module=gm,
    )
    rec.run(torch.randn(input_shape))

    return gm
