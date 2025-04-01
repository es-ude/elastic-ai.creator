import pytest
import torch

from elasticai.creator.base_modules.linear import Linear
from elasticai.creator.nn.binary.math_operations import MathOperations

operations = MathOperations()


class TestBinaryOperations:
    def test_quantization_returns_1_for_0_8(self) -> None:
        x = torch.tensor([0.8])
        x = operations.quantize(x)
        x = x.tolist()
        assert x == [1.0]

    def test_quantization_returns_1_for_2_3(self) -> None:
        x = torch.tensor([2.3])
        x = operations.quantize(x)
        x = x.tolist()
        assert x == [1.0]

    def test_quantization_returns_minus_1_for_minus_0_7(self) -> None:
        x = torch.tensor([-0.7])
        x = operations.quantize(x)
        x = x.tolist()
        assert x == [-1.0]

    def test_quantization_returns_minus_1_for_minus_3(self) -> None:
        x = torch.tensor([-3])
        x = operations.quantize(x)
        x = x.tolist()
        assert x == [-1.0]

    def test_quantization_produces_gradient_1_for_minus_0_3(self) -> None:
        x = torch.tensor([-0.3], requires_grad=True)
        y = operations.quantize(x)
        y.backward()
        g = x.grad
        assert g.tolist() == [1.0]

    def test_quantization_produces_gradient_0_for_3(self) -> None:
        x = torch.tensor([3.0], requires_grad=True)
        y = operations.quantize(x)
        y.backward()
        g = x.grad
        assert g.tolist() == [0.0]

    @pytest.mark.parametrize(
        argnames=("a", "b", "result"),
        argvalues=(
            (-1, 1, 1),
            (-1, -1, -1),
            (1, 1, 1),
        ),
    )
    def test_addition(self, a, b, result) -> None:
        x = torch.tensor([float(a)])
        y = torch.tensor([float(b)])
        assert operations.add(x, y) == torch.tensor([float(result)])

    @pytest.fixture(params=((-1, -1, 1), (1, 1, -1), (-1, -1, -1), (1, 1, 1)))
    def operants(self, request):
        return request.param

    @pytest.fixture
    def operants_as_tensors(self, operants) -> torch.Tensor:
        return torch.tensor([float(x) for x in operants])

    def test_matmul(self) -> None:
        a = [[-1.0, -1.0], [1.0, -1]]
        b = [[1.0, 1.0], [-1.0, 1.0]]
        expected = [[1.0, -1.0], [1.0, 1.0]]
        actual = operations.matmul(torch.tensor(a), torch.tensor(b)).tolist()
        assert actual == expected

    def test_mul(self) -> None:
        a = [-1.0, -1.0, 1.0, 1.0]
        b = [1.0, -1.0, -1.0, 1.0]
        expected = [-1.0, 1.0, -1.0, 1.0]
        actual = operations.mul(torch.tensor(a), torch.tensor(b)).tolist()
        assert actual == expected


class TestBinaryOperationsWithLinearLayer:
    def test_binary_linear_layer(self) -> None:
        x = [-0.4, 0.1]
        weight = [[0.1, -0.8], [-1.2, 4.0]]
        expected = [-1.0, 1.0]
        linear = Linear(
            operations=MathOperations(),
            in_features=len(x),
            out_features=len(weight),
            bias=False,
        )
        with torch.no_grad():
            linear.weight = torch.nn.Parameter(torch.tensor(weight))

        actual = linear(torch.tensor(x)).tolist()
        assert actual == expected
