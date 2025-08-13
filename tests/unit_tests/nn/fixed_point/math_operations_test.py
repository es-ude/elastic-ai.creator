import torch

from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from elasticai.creator.nn.fixed_point.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from tests.tensor_test_case import TensorTestCase


class FixedPointMathOperationsTest(TensorTestCase):
    def setUp(self) -> None:
        self.config: FixedPointConfig = FixedPointConfig(total_bits=4, frac_bits=2)
        self.operations = MathOperations(config=self.config)

    def test_quantize_clamps_minus5_to_minus2(self) -> None:
        a = torch.tensor([-5.0])
        actual = self.operations.quantize(a)
        expected = [-2.0]
        self.assertTensorEqual(expected, actual)

    def test_quantize_rounds_1_1_to_1_0(self) -> None:
        a = torch.tensor([1.1])
        actual = self.operations.quantize(a)
        expected = [1.0]
        self.assertTensorEqual(expected, actual)

    def test_add(self) -> None:
        a = torch.tensor([-0.25, 0.5, 1.0])
        b = torch.tensor([-1.5, 1.0, 1.5])
        actual = self.operations.add(a, b)
        expected = [-1.75, 1.5, 1.75]
        self.assertTensorEqual(expected, actual)

    def test_matmul(self) -> None:
        a = torch.tensor([[-2.0, -1.75, -1.5], [-0.25, 0.0, 0.25], [1.25, 1.5, 1.75]])
        b = torch.tensor([[-0.25], [0.5], [0.25]])
        actual = self.operations.matmul(a, b)
        expected = [[-0.75], [0.0], [0.75]]
        self.assertTensorEqual(expected, actual)

    def test_mul(self) -> None:
        a = torch.tensor([-0.5, 1.5, 0.5])
        b = torch.tensor([0.5, 1.5, 1.25])
        actual = self.operations.mul(a, b)
        expected = [-0.25, 1.75, 0.5]
        self.assertTensorEqual(expected, actual)

    def test_config_float_to_integer_2_2(self) -> None:
        config = FixedPointConfig(total_bits=2, frac_bits=2)
        chck = [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
        stimuli = [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
        rslt = [config.cut_as_integer(val) for val in stimuli]
        self.assertListEqual(rslt, chck)

    def test_config_float_to_integer_4_3(self) -> None:
        config = FixedPointConfig(total_bits=4, frac_bits=3)
        chck = [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
        stimuli = [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
        rslt = [config.cut_as_integer(val) for val in stimuli]
        self.assertListEqual(rslt, chck)

    def test_config_float_to_integer_8_4(self) -> None:
        config = FixedPointConfig(total_bits=8, frac_bits=4)
        chck = [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
        stimuli = [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
        rslt = [config.cut_as_integer(val) for val in stimuli]
        self.assertListEqual(rslt, chck)

    def test_config_T_to_integer_2_2(self) -> None:
        config = FixedPointConfig(total_bits=2, frac_bits=2)

        chck = torch.Tensor(
            [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
        )
        stimuli = torch.Tensor(
            [
                config.minimum_as_rational,
                -config.minimum_step_as_rational,
                0,
                config.minimum_step_as_rational,
                config.maximum_as_rational,
            ]
        )
        rslt = config.cut_as_integer(stimuli)
        self.assertTensorEqual(rslt, chck)

    def test_config_T_to_integer_4_3(self) -> None:
        config = FixedPointConfig(total_bits=4, frac_bits=3)

        chck = torch.Tensor(
            [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
        )
        stimuli = torch.Tensor(
            [
                config.minimum_as_rational,
                -config.minimum_step_as_rational,
                0,
                config.minimum_step_as_rational,
                config.maximum_as_rational,
            ]
        )
        rslt = config.cut_as_integer(stimuli)
        self.assertTensorEqual(rslt, chck)

    def test_config_T_to_integer_8_4(self) -> None:
        config = FixedPointConfig(total_bits=8, frac_bits=4)

        chck = torch.Tensor(
            [config.minimum_as_integer, -1, 0, +1, config.maximum_as_integer]
        )
        stimuli = torch.Tensor(
            [
                config.minimum_as_rational,
                -config.minimum_step_as_rational,
                0,
                config.minimum_step_as_rational,
                config.maximum_as_rational,
            ]
        )
        rslt = config.cut_as_integer(stimuli)
        self.assertTensorEqual(rslt, chck)

    def test_config_x_to_integer_4_3(self) -> None:
        config = FixedPointConfig(total_bits=4, frac_bits=3)

        stimuli_tensor = torch.Tensor(
            [
                config.minimum_as_rational,
                -config.minimum_step_as_rational,
                0,
                config.minimum_step_as_rational,
                config.maximum_as_rational,
            ]
        )
        stimuli_float = [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
        rslt_tensor = config.cut_as_integer(stimuli_tensor).tolist()
        rslt_float = [config.cut_as_integer(val) for val in stimuli_float]
        self.assertListEqual(rslt_tensor, rslt_float)

    def test_config_x_to_integer_8_4(self) -> None:
        config = FixedPointConfig(total_bits=8, frac_bits=4)

        stimuli_tensor = torch.Tensor(
            [
                config.minimum_as_rational,
                -config.minimum_step_as_rational,
                0,
                config.minimum_step_as_rational,
                config.maximum_as_rational,
            ]
        )
        stimuli_float = [
            config.minimum_as_rational,
            -config.minimum_step_as_rational,
            0,
            config.minimum_step_as_rational,
            config.maximum_as_rational,
        ]
        rslt_tensor = config.cut_as_integer(stimuli_tensor).tolist()
        rslt_float = [config.cut_as_integer(val) for val in stimuli_float]
        self.assertListEqual(rslt_tensor, rslt_float)
