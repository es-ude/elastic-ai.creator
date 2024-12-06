from typing import Protocol

from elasticai.creator.ir import Lowerable
from elasticai.creator.ir2vhdl import type_handler


class DummyLowerable(Lowerable, Protocol):
    more_data: list[str]


@type_handler
def convolution(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
