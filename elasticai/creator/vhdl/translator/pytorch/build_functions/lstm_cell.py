import torch

from elasticai.creator.vhdl.translator.abstract.layers import LSTMCell


def _extract_weights(
    lstm_cell: torch.nn.LSTMCell,
) -> dict[str, list[list[float]] | list[float], ...]:
    hidden_size = lstm_cell.hidden_size

    def split_weight(
        weight: torch.Tensor, names: list[str]
    ) -> dict[str, list[list[float]] | list[float]]:
        weight = weight.detach().numpy()

        lstm_weights = dict()
        for i, name in enumerate(names):
            lstm_weights[name] = weight[
                hidden_size * i : hidden_size * (i + 1)
            ].tolist()

        return lstm_weights

    weights_i = split_weight(
        lstm_cell.weight_ih, ["weights_ii", "weights_if", "weights_ig", "weights_io"]
    )
    weights_h = split_weight(
        lstm_cell.weight_hh, ["weights_hi", "weights_hf", "weights_hg", "weights_ho"]
    )
    bias_i = split_weight(
        lstm_cell.bias_ih, ["bias_ii", "bias_if", "bias_ig", "bias_io"]
    )
    bias_h = split_weight(
        lstm_cell.bias_hh, ["bias_hi", "bias_hf", "bias_hg", "bias_ho"]
    )

    return dict(**weights_i, **weights_h, **bias_i, **bias_h)


def build_lstm_cell(lstm_cell: torch.nn.LSTMCell) -> LSTMCell:
    return LSTMCell(**_extract_weights(lstm_cell))
