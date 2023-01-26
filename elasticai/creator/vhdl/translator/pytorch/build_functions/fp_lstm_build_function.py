import torch

from elasticai.creator.nn.lstm import FixedPointLSTM
from elasticai.creator.vhdl.translator.abstract.layers import LSTMModule


def build_fixed_point_lstm(
    layer: FixedPointLSTM,
    layer_id: str,
    sigmoid_resolution: tuple[float, float, int],
    tanh_resolution: tuple[float, float, int],
    work_library_name: str,
) -> LSTMModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    return LSTMModule(
        weights_ih=[to_list(layer.cell.linear_ih.weight)],
        weights_hh=[to_list(layer.cell.linear_hh.weight)],
        biases_ih=[to_list(layer.cell.linear_ih.bias)],
        biases_hh=[to_list(layer.cell.linear_hh.bias)],
        layer_id=layer_id,
        fixed_point_factory=layer.fixed_point_factory,
        sigmoid_resolution=sigmoid_resolution,
        tanh_resolution=tanh_resolution,
        work_library_name=work_library_name,
    )
