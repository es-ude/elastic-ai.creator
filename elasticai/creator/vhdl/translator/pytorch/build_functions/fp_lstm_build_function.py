import torch

from elasticai.creator.nn.lstm import FixedPointLSTMWithHardActivations
from elasticai.creator.vhdl.translator.abstract.layers import LSTMModule


def build_fixed_point_lstm(
    layer: FixedPointLSTMWithHardActivations,
    layer_id: str,
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
        work_library_name=work_library_name,
    )
