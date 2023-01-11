from collections.abc import Iterator

import torch

from elasticai.creator.vhdl.number_representations import FixedPointFactory
from elasticai.creator.vhdl.translator.abstract.layers import LSTMModule


def build_lstm(
    layer: torch.nn.LSTM,
    layer_id: str,
    fixed_point_factory: FixedPointFactory,
    sigmoid_resolution: tuple[float, float, int],
    tanh_resolution: tuple[float, float, int],
    work_library_name: str,
) -> LSTMModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    def get_weights(weight_prefix: str) -> Iterator[list]:
        for i in range(layer.num_layers):
            yield to_list(getattr(layer, f"{weight_prefix}_l{i}"))

    return LSTMModule(
        weights_ih=list(get_weights("weight_ih")),
        weights_hh=list(get_weights("weight_hh")),
        biases_ih=list(get_weights("bias_ih")),
        biases_hh=list(get_weights("bias_hh")),
        layer_id=layer_id,
        fixed_point_factory=fixed_point_factory,
        sigmoid_resolution=sigmoid_resolution,
        tanh_resolution=tanh_resolution,
        work_library_name=work_library_name,
    )
