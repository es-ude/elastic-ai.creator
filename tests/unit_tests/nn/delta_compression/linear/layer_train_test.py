import pytest
import torch

from elasticai.creator.nn.delta_compression import DeltaType
from elasticai.creator.nn.delta_compression.linear import Linear


def _make_layer(
    in_features: int = 2,
    out_features: int = 1,
    total_bits: int = 8,
    frac_bits: int = 4,
    delta_bits: int = 4,
    delta_offset: int = 0,
    delta_type: DeltaType = DeltaType.CONSECUTIVE,
    clamp: bool = False,
    bias: bool = True,
) -> Linear:
    return Linear(
        in_features=in_features,
        out_features=out_features,
        total_bits=total_bits,
        frac_bits=frac_bits,
        delta_bits=delta_bits,
        delta_offset=delta_offset,
        delta_type=delta_type,
        clamp=clamp,
        bias=bias,
    )


class TestLinearForward:
    def test_output_shape(self):
        layer = _make_layer(in_features=3, out_features=2)
        assert layer(torch.zeros(3)).shape == (2,)

    def test_output_shape_for_batch_input(self):
        layer = _make_layer(in_features=3, out_features=2)
        assert layer(torch.zeros(5, 3)).shape == (5, 2)

    def test_forward_without_bias(self):
        """Equal weights produce zero consecutive deltas and survive compression unchanged."""
        layer = _make_layer(in_features=2, out_features=1, bias=False)
        layer.weight.data = torch.tensor([[1.0, 1.0]])
        output = layer(torch.tensor([1.0, 2.0]))
        assert output.tolist() == [3.0]

    def test_forward_with_bias(self):
        """A single-element bias has no delta tail, so it passes through compress/inflate as its absolute integer value."""
        layer = _make_layer(in_features=2, out_features=1)
        layer.weight.data = torch.tensor([[1.0, 1.0]])
        layer.bias.data = torch.tensor([0.5])
        output = layer(torch.tensor([1.0, 2.0]))
        assert output.tolist() == [3.5]

    def test_output_clamped_to_fixed_point_range(self):
        """Matmul result exceeding the max representable value is clamped."""
        layer = _make_layer(
            in_features=1, out_features=1, total_bits=4, frac_bits=1, bias=False
        )
        layer.weight.data = torch.tensor([[2.0]])
        # 4-bit 1-frac: max = (2**3 - 1) * 0.5 = 3.5; matmul = 2.0 * 4.0 = 8.0
        output = layer(torch.tensor([4.0]))
        assert output.tolist() == [3.5]


class TestLinearGetParams:
    def test_get_params_returns_raw_float_weights_and_bias(self):
        layer = _make_layer(in_features=2, out_features=1)
        layer.weight.data = torch.tensor([[0.5, -0.5]])
        layer.bias.data = torch.tensor([1.0])
        weights, bias = layer.get_params()
        assert weights == [[0.5, -0.5]]
        assert bias == [1.0]

    def test_get_params_returns_zero_bias_when_bias_is_disabled(self):
        """_bias property substitutes a zero vector when no bias parameter exists."""
        layer = _make_layer(in_features=2, out_features=1, bias=False)
        _, bias = layer.get_params()
        assert bias == [0.0]

    def test_get_params_compressed_first_weight_is_absolute_integer(self):
        """First element in the compressed sequence is the quantized integer, not a delta."""
        layer = _make_layer(in_features=2, out_features=1, bias=False)
        layer.weight.data = torch.tensor([[1.0, 1.0625]])
        c_weights, _ = layer.get_params_compressed()
        # 1.0 * 16 (= 2**frac_bits) = 16
        assert c_weights[0][0] == 16.0

    def test_get_params_compressed_subsequent_weights_are_consecutive_deltas(self):
        """Weights after the first are replaced by their consecutive difference in integer space."""
        layer = _make_layer(in_features=2, out_features=1, bias=False)
        # Integer values: 1.0 → 16, 1.0625 → 17; delta = 17 - 16 = 1
        layer.weight.data = torch.tensor([[1.0, 1.0625]])
        c_weights, _ = layer.get_params_compressed()
        assert c_weights == [[16.0, 1.0]]

    def test_get_params_compressed_bias(self):
        """A single-element bias is stored as its absolute integer value with no delta."""
        layer = _make_layer(in_features=2, out_features=1)
        layer.weight.data = torch.tensor([[1.0, 1.0625]])
        layer.bias.data = torch.tensor([0.5])
        # 0.5 / 0.0625 (step) = 8
        _, c_bias = layer.get_params_compressed()
        assert c_bias == [8.0]


@pytest.mark.slow
def test_training_loop_completes_without_error():
    layer = _make_layer(in_features=4, out_features=4)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(layer.parameters())
    x = torch.rand(8, 4)
    for _ in range(3):
        y = layer(x)
        loss = criterion(y, torch.zeros_like(y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TestCreateDesign:
    def test_create_design_raises_not_implemented(self):
        layer = _make_layer()
        with pytest.raises(NotImplementedError):
            layer.create_design("test_design")
