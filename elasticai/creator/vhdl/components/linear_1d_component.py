import math
from typing import Callable

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint


class Linear1dComponent:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_factory: Callable[[float], FixedPoint],
    ) -> None:

        if out_features != 1:
            raise NotImplementedError(
                "Currently only one bias is supported (which implies that out_features must be 1)."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.data_width = self._derive_data_width(fixed_point_factory)
        self.addr_width = self._calculate_addr_width(in_features * out_features)

    @staticmethod
    def _derive_data_width(fixed_point_factory: Callable[[float], FixedPoint]) -> int:
        return fixed_point_factory(0).total_bits

    @staticmethod
    def _calculate_addr_width(num_items: int) -> int:
        return max(1, math.ceil(math.log2(num_items)))

    @property
    def file_name(self) -> str:
        return "linear_1d.vhd"

    def __call__(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "linear_1d.tpl.vhd")

        code = template.format(
            addr_width=self.addr_width,
            data_width=self.out_features,
            in_feature_count=self.in_features,
            out_feature_count=self.out_features,
        )

        yield from code.splitlines()
