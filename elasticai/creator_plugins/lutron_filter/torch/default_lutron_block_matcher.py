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
)

from ..nn.lutron.binarize import Binarize
from .lutron_block_detection import (
    PatternMatch,
    SequentialPattern,
    detect_type_sequences,
)

P = ParamSpec("P")


def seq_with_node_of_interest(
    *nodes_of_interest: int,
):
    def wrapped(*type_sequence: type | Callable | str) -> SequentialPattern:
        return SequentialPattern(
            nodes_of_interest=nodes_of_interest, type_sequence=type_sequence
        )

    return wrapped


class LutronBlockMatcher:
    def __init__(
        self, module: Module, graph: torch.fx.Graph, binarize: type = Binarize
    ):
        self._m = module
        self._g = graph
        self._binarize = binarize

    def conv1d(self) -> Iterator[PatternMatch]:
        type_lists = (
            (Conv1d, BatchNorm1d, PReLU, self._binarize),
            (Conv1d, BatchNorm1d, self._binarize),
            (Conv1d, PReLU, self._binarize),
            (Conv1d, self._binarize),
        )
        noi_0 = self._seq_with_node_of_interest(0)
        patterns = tuple(noi_0(*seq) for seq in type_lists)
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
        noi_1 = self._seq_with_node_of_interest(1)
        type_lists = (
            noi_1(Flatten, Linear, self._binarize),
            noi_1(torch.flatten, Linear, self._binarize),
            noi_1(Flatten, Linear, Module, Flatten, self._binarize),
            noi_1(Flatten, Linear, Module),
            noi_1(Flatten, Linear),
        )
        yield from detect_type_sequences(self._m, self._g, type_lists)

    def maxpool1d(self) -> Iterator[PatternMatch]:
        noi_0 = self._seq_with_node_of_interest(0)
        yield from detect_type_sequences(
            self._m,
            self._g,
            (
                noi_0(
                    MaxPool1d,
                ),
            ),
        )
