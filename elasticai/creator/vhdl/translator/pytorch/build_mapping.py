from typing import Callable

import torch

from elasticai.creator.vhdl.translator.abstract.translatable import Translatable
from elasticai.creator.vhdl.translator.pytorch.build_functions import build_lstm_cell

BuildFunction = Callable[[torch.nn.Module], Translatable]


class BuildMapping:
    def __init__(self) -> None:
        self._mapping = {"torch.nn.LSTMCell": build_lstm_cell}

    @staticmethod
    def _get_cls_name(layer_cls: type[torch.nn.Module]) -> str:
        return f"{layer_cls.__module__}.{layer_cls.__name__}"

    def get(self, layer_cls: type[torch.nn.Module]) -> BuildFunction | None:
        return self._mapping.get(self._get_cls_name(layer_cls))

    def set(
        self, layer_cls: type[torch.nn.Module], build_function: BuildFunction
    ) -> None:
        self._mapping[self._get_cls_name(layer_cls)] = build_function
