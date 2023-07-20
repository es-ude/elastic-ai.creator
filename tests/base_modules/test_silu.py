import torch

from elasticai.creator.base_modules.arithmetics import float_arithmetics
from elasticai.creator.base_modules.arithmetics.arithmetics import Arithmetics
from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.arithmetics.float_arithmetics import (
    FloatArithmetics,
)
from elasticai.creator.base_modules.silu import SiLU
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from tests.tensor_test_case import TensorTestCase


def tensor(data: list) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


def silu_base(arithmetics: Arithmetics) -> SiLU:
    silu = SiLU(arithmetics=arithmetics)
    return silu


class SiLUTest(TensorTestCase):
    def test_float(self):
        silu = silu_base(FloatArithmetics())

        input = tensor(list(range(-10, 10)))
        actual = silu(input)
        expected = torch.nn.functional.silu(input)

        self.assertTensorEqual(expected, actual)

    # self.round(self.clamp(a))
    def test_fixed_point(self):
        config = FixedPointConfig(total_bits=8, frac_bits=4)
        silu = silu_base(FixedPointArithmetics(config=config))

        input = tensor(list(range(-5, 5)))
        actual = silu(input)
