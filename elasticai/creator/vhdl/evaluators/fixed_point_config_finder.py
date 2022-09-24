from typing import Any, Iterable

from elasticai.creator.mlframework.typing import Module
from elasticai.creator.vhdl.evaluators.evaluator import Evaluator

DataLoader = Iterable[tuple[Any, Any]]


class FixedPointConfigFinder(Evaluator):
    def __init__(self, model: Module, data: DataLoader) -> None:
        self.model = model
        self.data = data

    def run(self) -> dict[str, int]:
        return dict(total_bits=2, frac_bits=0)
