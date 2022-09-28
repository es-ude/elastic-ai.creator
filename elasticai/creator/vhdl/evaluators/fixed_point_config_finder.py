from copy import deepcopy
from typing import Any, Callable, Iterable

from elasticai.creator.mlframework.typing import Module
from elasticai.creator.vhdl.evaluators.evaluator import Evaluator
from elasticai.creator.vhdl.number_representations import FixedPoint

DataLoader = Iterable[tuple[Any, Any]]


class FixedPointConfigFinder(Evaluator):
    def __init__(self, model: Module, data: DataLoader, total_bits: int) -> None:
        self.model = model
        self.data = data
        self.total_bits = total_bits

    def _to_fixed_point(self, value: float, frac_bits: int) -> int:
        to_fp = FixedPoint.get_factory(self.total_bits, frac_bits)
        return to_fp(value).to_signed_int()

    def _to_float(self, value: int, frac_bits: int) -> float:
        fp = FixedPoint.from_int(value, self.total_bits, frac_bits, signed_int=True)
        return float(fp)

    def _apply_to_params(self, model: Module, callable: Callable) -> None:
        if model.parameters() is not None:
            state_dict = model.state_dict()
            print(state_dict)

    def run(self) -> dict[str, int]:
        for frac_bits in range(self.total_bits):
            model = deepcopy(self.model)
            self._apply_to_params(model, self._to_fixed_point)

        return dict(total_bits=2, frac_bits=0)
