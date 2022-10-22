import re
from copy import deepcopy
from typing import Any, Callable, Iterable

from elasticai.creator.mlframework.typing import Module
from elasticai.creator.vhdl.evaluators.evaluator import Evaluator
from elasticai.creator.vhdl.number_representations import ClippedFixedPoint

DataLoader = Iterable[tuple[Any, Any]]


def get_attribute_names(obj: object, regex: str) -> list[str]:
    all_attributes = obj.__dict__.keys()
    return [attr for attr in all_attributes if re.fullmatch(regex, attr) is not None]


class FixedPointConfigFinder(Evaluator):
    def __init__(self, model: Module, data: DataLoader, total_bits: int) -> None:
        self.model = model
        self.data = data
        self.total_bits = total_bits

    def _to_fixed_point(self, value: float, frac_bits: int) -> int:
        to_fp = ClippedFixedPoint.get_factory(self.total_bits, frac_bits)
        return to_fp(value).to_signed_int()

    def _to_float(self, value: int, frac_bits: int) -> float:
        fp = ClippedFixedPoint.from_signed_int(value, self.total_bits, frac_bits)
        return float(fp)

    def _apply_to_params(self, model: Module, callable: Callable) -> None:
        ...

    def run(self) -> dict[str, int]:
        for frac_bits in range(self.total_bits):
            model = deepcopy(self.model)
            self._apply_to_params(model, self._to_fixed_point)

        return dict(total_bits=2, frac_bits=0)
