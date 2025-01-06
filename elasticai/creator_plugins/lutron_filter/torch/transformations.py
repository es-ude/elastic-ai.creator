import copy
from collections.abc import Sequence
from typing import Callable, Iterator, cast

from torch.fx import Graph, GraphModule
from torch.fx import Graph as fxGraph
from torch.fx import Node as fxNode
from torch.nn import Conv1d, Identity, Module

from elasticai.creator_plugins.lutron_filter.nn.lutron.binarize import Binarize

from .default_lutron_block_matcher import (
    LutronBlockMatcher as DefaultLutronBlockMatcher,
)
from .lutron_block_autogen import LutronBlockMatcher
from .lutron_block_detection import (
    PatternMatch,
    SequentialPattern,
    detect_type_sequences,
)
from .transformation_utils import (
    LeafTracer,
    get_module_for_node,
    nodes_for_types_while_sequential,
)


def remove_identities(root: Module, graph: fxGraph) -> GraphModule:
    for n in graph.nodes:
        if n.op == "call_module":
            m = get_module_for_node(n, root)
            if isinstance(m, Identity):
                input_node = n.all_input_nodes[0]
                n.replace_all_uses_with(input_node)
                graph.erase_node(n)
    return GraphModule(root, graph)


def reorder_conv_blocks(
    m: Module, graph: fxGraph, type_lists: Sequence[tuple[type | str | Callable, ...]]
) -> GraphModule:
    def get_longest_match(n: fxNode):
        matches: list[list[fxNode]] = [[]]
        for type_list in type_lists:
            node_sequence: list[fxNode] = list(
                nodes_for_types_while_sequential(m, n, type_list)
            )  # type: ignore
            if len(node_sequence) == len(type_list):
                matches.append(node_sequence)
        longest = 0
        for i, match in enumerate(matches):
            if len(match) > len(matches[longest]):
                longest = i
        return matches[longest]

    for n in graph.nodes:
        node_sequence: list[fxNode] = get_longest_match(n)
        if len(node_sequence) == 4:
            conv, pool, bn, quant = node_sequence
            bn.replace_input_with(pool, conv)
            quant.replace_all_uses_with(pool)
            pool.replace_input_with(conv, quant)
            quant.append(pool)

    return GraphModule(m, graph)


class SplitBlockMatcher(LutronBlockMatcher):
    def __init__(self, model: Module, graph: Graph):
        self._m = model
        self._g = graph
        self._default_m = DefaultLutronBlockMatcher(model, graph)

    def maxpool1d(self) -> Iterator[PatternMatch]:
        yield from self._default_m.maxpool1d()

    def conv1d(self) -> Iterator[PatternMatch]:
        patterns = (
            SequentialPattern(
                (0, 1), ("Conv1d", "MaxPool1d", "BatchNorm1d", "Binarize")
            ),
            SequentialPattern(
                (0, 1), ("Conv1d", "Identity", "BatchNorm1d", "Binarize")
            ),
            SequentialPattern(
                (0, 2), ("Conv1d", "BatchNorm1d", "MaxPool1d", "Binarize")
            ),
            SequentialPattern((0, 1), ("Conv1d", "MaxPool1d", "Binarize")),
        )
        yield from detect_type_sequences(self._m, self._g, patterns)

    def linear(self) -> Iterator[PatternMatch]:
        yield from self._default_m.linear()


def get_parent_module(root: Module, target: str):
    parent_path = ".".join(target.split(".")[:-1])
    if len(parent_path) == 0:
        parent = root
    else:
        parent = root.get_submodule(parent_path)
    return parent


def split_convolutions(m: Module, replace_fn: Callable[[Conv1d], Module]) -> Module:
    t = LeafTracer((Binarize,))
    g = t.trace(m)
    matcher = SplitBlockMatcher(m, g)
    modules = {name: m for name, m in m.named_modules()}
    m_copy = copy.deepcopy(m)
    conv_blocks = tuple(matcher.conv1d())

    for match in conv_blocks:
        conv_node = match.matched_sequence[0]
        conv = modules[conv_node.target]
        new_conv = replace_fn(cast(Conv1d, conv))
        target = cast(str, conv_node.target)
        parent = get_parent_module(m_copy, target)
        module_name = f"{target.split('.')[-1]}_split"
        parent.add_module(module_name, new_conv)
        conv_node.target = ".".join(target.split(".")[:-1] + [module_name])
    gm = GraphModule(m_copy, g)
    return gm
