import pytest
import torch

from elasticai.creator.nn.binary_arithmetics import BinaryArithmetics

arithmetic = BinaryArithmetics()


def test_quantization_returns_1_for_0_8():
    x = torch.tensor([0.8])
    x = arithmetic.quantize(x)
    x = x.tolist()
    assert x == [1.0]


def test_quantization_returns_1_for_2_3():
    x = torch.tensor([2.3])
    x = arithmetic.quantize(x)
    x = x.tolist()
    assert x == [1.0]


def test_quantization_returns_minus_1_for_minus_0_7():
    x = torch.tensor([-0.7])
    x = arithmetic.quantize(x)
    x = x.tolist()
    assert x == [-1.0]


def test_quantization_returns_minus_1_for_minus_3():
    x = torch.tensor([-3])
    x = arithmetic.quantize(x)
    x = x.tolist()
    assert x == [-1.0]


def test_quantization_produces_gradient_1_for_minus_0_3():
    x = torch.tensor([-0.3], requires_grad=True)
    y = arithmetic.quantize(x)
    y.backward()
    g = x.grad
    assert g.tolist() == [1.0]


def test_quantization_produces_gradient_0_for_3():
    x = torch.tensor([3.0], requires_grad=True)
    y = arithmetic.quantize(x)
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
def test_addition(a, b, result):
    x = torch.tensor([float(a)])
    y = torch.tensor([float(b)])
    assert arithmetic.add(x, y) == torch.tensor([float(result)])


@pytest.fixture(params=((-1, -1, 1), (1, 1, -1), (-1, -1, -1), (1, 1, 1)))
def operants(request):
    return request.param


@pytest.fixture
def operants_as_tensors(operants) -> torch.Tensor:
    return torch.tensor([float(x) for x in operants])


def test_summation(operants, operants_as_tensors):
    number_of_ones = len(tuple((filter(lambda x: x > 0, operants))))
    expected = -1.0
    if number_of_ones > len(operants) / 2:
        expected = 1.0
    result = arithmetic.sum(operants_as_tensors).tolist()
    assert result == expected


def test_matmul():
    a = [[-1.0, -1.0], [1.0, -1]]
    b = [[1.0, 1.0], [-1.0, 1.0]]
    expected = [[1.0, -1.0], [1.0, 1.0]]
    actual = arithmetic.matmul(torch.tensor(a), torch.tensor(b)).tolist()
    assert actual == expected


def test_mul():
    a = [-1.0, -1.0, 1.0, 1.0]
    b = [1.0, -1.0, -1.0, 1.0]
    expected = [-1.0, 1.0, -1.0, 1.0]
    actual = arithmetic.mul(torch.tensor(a), torch.tensor(b)).tolist()
    assert actual == expected
