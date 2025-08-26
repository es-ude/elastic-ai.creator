from typing import Any, cast

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.conv1d import Conv1d as Conv1dBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.conv1d.design import Conv1dDesign
from elasticai.creator.nn.fixed_point.conv1d.testbench import Conv1dTestbench
from elasticai.creator.nn.fixed_point.math_operations import MathOperations


class Conv1d(DesignCreatorModule, Conv1dBase):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        in_channels: int,
        out_channels: int,
        signal_length: int,
        kernel_size: int | tuple[int],
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._params = FxpParams(total_bits=total_bits, frac_bits=frac_bits)
        self._config = FxpArithmetic(self._params)
        self._signal_length = signal_length
        super().__init__(
            operations=MathOperations(config=self._config),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            device=device,
        )

    def create_design(self, name: str) -> Conv1dDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.cut_as_integer(value)

        def flatten_tuple(x: int | tuple[int, ...]) -> int:
            return x[0] if isinstance(x, tuple) else x

        bias = [0] * self.out_channels if self.bias is None else self.bias.tolist()
        signed_int_weights = cast(
            list[list[list[int]]], float_to_signed_int(self.weight.tolist())
        )
        signed_int_bias = cast(list[int], float_to_signed_int(bias))

        return Conv1dDesign(
            name=name,
            total_bits=self._config.total_bits,
            frac_bits=self._config.frac_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            signal_length=self._signal_length,
            kernel_size=flatten_tuple(self.kernel_size),
            weights=signed_int_weights,
            bias=signed_int_bias,
        )

    def create_testbench(self, name: str, uut: Conv1dDesign) -> Conv1dTestbench:
        return Conv1dTestbench(name=name, uut=uut, fxp_params=self._params)
