from dataclasses import dataclass
from typing import Callable, Iterator

from torch.fx import Graph, Node
from torch.nn import Module

from .transformation_utils import nodes_for_types_while_sequential


@dataclass
class SequentialPattern:
    nodes_of_interest: tuple[int, ...]
    type_sequence: tuple[type | Callable | str, ...]


@dataclass
class PatternMatch:
    nodes_of_interest: tuple[int, ...]
    matched_sequence: tuple[Node, ...]


def _default_seq_pattern(*items: type | Callable) -> SequentialPattern:
    return SequentialPattern(nodes_of_interest=(0,), type_sequence=items)


def detect_type_sequences(
    root: Module, g: Graph, patterns: tuple[SequentialPattern, ...]
) -> Iterator[PatternMatch]:
    for n in g.nodes:
        for pattern in patterns:
            matches = tuple(
                nodes_for_types_while_sequential(root, n, pattern.type_sequence)
            )
            if len(matches) == len(pattern.type_sequence):
                yield PatternMatch(
                    nodes_of_interest=pattern.nodes_of_interest,
                    matched_sequence=matches,
                )
