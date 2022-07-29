from collections.abc import Mapping
from typing import Callable, Iterator

import torch

from elasticai.creator.vhdl.translator.abstract.translatable import Translatable
from elasticai.creator.vhdl.translator.pytorch.build_functions import build_lstm_cell

BuildFunction = Callable[[torch.nn.Module], Translatable]


class BuildFunctionMapping(Mapping[str, BuildFunction]):
    def __init__(self, mapping: dict[str, BuildFunction]):
        self._mapping = mapping

    @staticmethod
    def _infer_type(x: type | object) -> type:
        return x if isinstance(x, type) else type(x)

    @staticmethod
    def _get_cls_name(cls: type) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    def get_from_layer(
        self, layer: torch.nn.Module | type[torch.nn.Module]
    ) -> BuildFunction | None:
        return self._mapping.get(self._get_cls_name(self._infer_type(layer)))

    def __getitem__(self, key: str) -> BuildFunction:
        return self._mapping[key]

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def to_dict(self) -> dict[str, BuildFunction]:
        return self._mapping.copy()


class DefaultBuildFunctionMapping(BuildFunctionMapping):
    def __init__(self):
        super().__init__(
            mapping={"torch.nn.modules.rnn.LSTMCell": build_lstm_cell},
        )
