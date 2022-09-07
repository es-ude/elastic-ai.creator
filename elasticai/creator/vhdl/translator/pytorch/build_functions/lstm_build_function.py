from collections.abc import Iterator

import torch

from elasticai.creator.vhdl.translator.abstract.layers import LSTMModule


def build_lstm(lstm: torch.nn.LSTM) -> LSTMModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    def get_weights(weight_prefix: str) -> Iterator[list]:
        for i in range(lstm.num_layers):
            yield to_list(getattr(lstm, f"{weight_prefix}_l{i}"))

    return LSTMModule(
        weights_ih=list(get_weights("weight_ih")),
        weights_hh=list(get_weights("weight_hh")),
        biases_ih=list(get_weights("bias_ih")),
        biases_hh=list(get_weights("bias_hh")),
    )
