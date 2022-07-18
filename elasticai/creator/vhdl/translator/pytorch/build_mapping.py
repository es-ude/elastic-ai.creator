from typing import Callable

import torch

from elasticai.creator.vhdl.translator.abstract.translatable import Translatable
from elasticai.creator.vhdl.translator.pytorch.build_functions import build_lstm_cell

BuildFunction = Callable[[torch.nn.Module], Translatable]


class BuildMapping:
    def __init__(self) -> None:
        self._mapping = {"torch.nn.LSTMCell": build_lstm_cell}

    @staticmethod
    def _infer_type(x: type | object) -> type:
        return x if isinstance(x, type) else type(x)

    @staticmethod
    def _get_cls_name(cls: type) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    def get(
        self, layer: type[torch.nn.Module] | torch.nn.Module
    ) -> BuildFunction | None:
        return self._mapping.get(self._get_cls_name(self._infer_type(layer)))

    def set(
        self,
        layer: type[torch.nn.Module] | torch.nn.Module,
        build_function: BuildFunction,
    ) -> None:
        self._mapping[self._get_cls_name(self._infer_type(layer))] = build_function
