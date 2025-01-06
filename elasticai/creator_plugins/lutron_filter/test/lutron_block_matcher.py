import operator
from typing import Callable, Iterator, ParamSpec

import torch.fx
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Flatten,
    Linear,
    MaxPool1d,
    Module,
    PReLU,
    Sigmoid,
)

from elasticai.creator_plugins.lutron_filter.torch import (
    PatternMatch,
    SequentialPattern,
    detect_type_sequences,
)

from .models import HumbleBinarization as BinaryQuantization

P = ParamSpec("P")


class LutronBlockMatcher:
    def __init__(self, module: Module, graph: torch.fx.Graph):
        self._m = module
        self._g = graph

    @staticmethod
    def _pattern_starting_with_node_of_interest(
        *items: type | Callable,
    ) -> SequentialPattern:
        return SequentialPattern(nodes_of_interest=(0,), type_sequence=items)  # type: ignore

    def conv1d(self) -> Iterator[PatternMatch]:
        type_lists = (
            (Conv1d, BatchNorm1d, PReLU, BinaryQuantization),
            (Conv1d, BatchNorm1d, BinaryQuantization),
            (Conv1d, PReLU, BinaryQuantization),
            (Conv1d, BinaryQuantization),
        )
        patterns = tuple(
            LutronBlockMatcher._pattern_starting_with_node_of_interest(*seq)
            for seq in type_lists
        )
        yield from detect_type_sequences(self._m, self._g, patterns)

    @staticmethod
    def _seq_with_node_of_interest(
        *nodes_of_interest: int,
    ):
        def wrapped(*type_sequence: type | Callable | str) -> SequentialPattern:
            return SequentialPattern(
                nodes_of_interest=nodes_of_interest, type_sequence=type_sequence
            )

        return wrapped

    def linear(self) -> Iterator[PatternMatch]:
        self._seq_with_node_of_interest(0)
        noi_1 = self._seq_with_node_of_interest(1)
        type_lists = (
            noi_1(Flatten, Linear, BinaryQuantization),
            noi_1(torch.flatten, Linear, BinaryQuantization),
            noi_1(
                torch.flatten,
                Linear,
                Module,
                BinaryQuantization,
                torch.flatten,
            ),
            noi_1(
                torch.flatten,
                Linear,
                Sigmoid,
                torch.flatten,
                BinaryQuantization,
            ),
            noi_1(
                torch.flatten,
                Linear,
                operator.truediv,
                "Sigmoid",
                torch.flatten,
                BinaryQuantization,
            ),
            noi_1(torch.flatten, Linear, "Sigmoid", torch.flatten, BinaryQuantization),
        )
        yield from detect_type_sequences(self._m, self._g, type_lists)

    def maxpool1d(self) -> Iterator[PatternMatch]:
        _default_seq_pattern = (
            LutronBlockMatcher._pattern_starting_with_node_of_interest
        )
        yield from detect_type_sequences(
            self._m,
            self._g,
            (
                _default_seq_pattern(
                    MaxPool1d,
                ),
            ),
        )
