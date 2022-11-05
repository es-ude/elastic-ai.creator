import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from elasticai.creator.examples.assets.determine_fixed_point.pems_dataset import (
    PeMSDataset,
)
from elasticai.creator.resource_utils import PathType
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers import (
    Linear1dTranslationArgs,
    LSTMTranslationArgs,
)
from elasticai.creator.vhdl.translator.pytorch import translator
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)


def read_commandline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir", required=True, type=Path)
    return parser.parse_args()


class LSTMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=20, batch_first=True)
        self.linear = torch.nn.Linear(in_features=20, out_features=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(self.lstm(x)[0])

    def get_weights_and_bias(self) -> list[np.ndarray]:
        return [
            self.lstm.weight_ih_l0.detach().numpy().squeeze(),
            self.lstm.weight_hh_l0.detach().numpy(),
            self.lstm.bias_ih_l0.detach().numpy(),
            self.lstm.bias_hh_l0.detach().numpy(),
            self.linear.weight.data.detach().numpy().squeeze(),
            self.linear.bias.data.detach().numpy(),
        ]


def determine_optimal_fixed_point_factory(
    model: LSTMModel,
    saved_model_path: PathType,
    data_path: PathType,
    sensor_idx: int,
    batch_size: int,
    datapoints_per_sample: int,
) -> Callable[[float | int], FixedPoint]:
    def train_test_split(
        dataset_path: PathType, datapoints_per_sample: int, sensor_idx: int
    ) -> tuple[PeMSDataset, PeMSDataset]:
        n_three_week_datapoints = 12 * 24 * 7 * 3
        train = PeMSDataset(
            dataset_path,
            datapoints_per_sample,
            sensor_idx,
            raw_data_slice=(0, n_three_week_datapoints),
        )
        test = PeMSDataset(
            dataset_path,
            datapoints_per_sample,
            sensor_idx,
            raw_data_slice=(n_three_week_datapoints, None),
        )
        return train, test

    _, test_data = train_test_split(
        dataset_path=data_path,
        datapoints_per_sample=datapoints_per_sample,
        sensor_idx=sensor_idx,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    weights_and_bias = model.get_weights_and_bias()

    total_bits = 16
    frac_bits_choices = [4, 6, 8, 10, 12, 14, 16]
    frac_bits_mse = []

    for frac_bits in frac_bits_choices:
        fixed_point_one = 2**frac_bits
        fp_args = dict(total_bits=total_bits, frac_bits=frac_bits)

        @np.vectorize
        def to_fixed_point(value: float) -> int:
            return FixedPoint.get_factory(**fp_args)(value).to_signed_int()

        @np.vectorize
        def to_float(value: int) -> float:
            return float(FixedPoint.from_int(value, **fp_args, signed_int=True))

        def fixed_point_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (np.matmul(a, b) * 1.0 / fixed_point_one).astype(int)

        #        def to_fixed_point(x: np.ndarray) -> np.ndarray:
        #            return (x * fixed_point_one).astype(int)
        #
        #        def to_float(x: np.ndarray) -> np.ndarray:
        #            return x * 1.0 / fixed_point_one
        #
        #        def fixed_point_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        #            return (np.matmul(a, b) * 1.0 / fixed_point_one).astype(int)

        def sigmoid(x: np.ndarray) -> np.ndarray:
            return to_fixed_point(1 / (1 + np.exp(-to_float(x))))

        def tanh(x: np.ndarray) -> np.ndarray:
            return to_fixed_point(np.tanh(to_float(x)))

        def qlstm_cell(
            x: np.ndarray,
            h_prev: np.ndarray,
            c_prev: np.ndarray,
            weights_and_bias: list[np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            x = to_fixed_point(np.ones([1, 1]) * x)

            w_ii = to_fixed_point(np.expand_dims(weights_and_bias[0][:20], axis=1))
            w_if = to_fixed_point(np.expand_dims(weights_and_bias[0][20:40], axis=1))
            w_ig = to_fixed_point(np.expand_dims(weights_and_bias[0][40:60], axis=1))
            w_io = to_fixed_point(np.expand_dims(weights_and_bias[0][60:80], axis=1))

            w_hi = to_fixed_point(weights_and_bias[1][:20])
            w_hf = to_fixed_point(weights_and_bias[1][20:40])
            w_hg = to_fixed_point(weights_and_bias[1][40:60])
            w_ho = to_fixed_point(weights_and_bias[1][60:80])

            b_ii = to_fixed_point(np.expand_dims(weights_and_bias[2][:20], axis=1))
            b_if = to_fixed_point(np.expand_dims(weights_and_bias[2][20:40], axis=1))
            b_ig = to_fixed_point(np.expand_dims(weights_and_bias[2][40:60], axis=1))
            b_io = to_fixed_point(np.expand_dims(weights_and_bias[2][60:80], axis=1))

            b_hi = to_fixed_point(np.expand_dims(weights_and_bias[3][:20], axis=1))
            b_hf = to_fixed_point(np.expand_dims(weights_and_bias[3][20:40], axis=1))
            b_hg = to_fixed_point(np.expand_dims(weights_and_bias[3][40:60], axis=1))
            b_ho = to_fixed_point(np.expand_dims(weights_and_bias[3][60:80], axis=1))

            f_t = sigmoid(
                fixed_point_matmul(w_if, x)
                + fixed_point_matmul(w_hf, h_prev)
                + b_if
                + b_hf
            )
            i_t = sigmoid(
                fixed_point_matmul(w_ii, x)
                + fixed_point_matmul(w_hi, h_prev)
                + b_ii
                + b_hi
            )
            g_t = tanh(
                fixed_point_matmul(w_ig, x)
                + fixed_point_matmul(w_hg, h_prev)
                + b_ig
                + b_hg
            )
            o_t = sigmoid(
                fixed_point_matmul(w_io, x)
                + fixed_point_matmul(w_ho, h_prev)
                + b_io
                + b_ho
            )

            c_t = f_t * c_prev / fixed_point_one + i_t * g_t / fixed_point_one
            h_t = o_t * tanh(c_t) / fixed_point_one

            return h_t, c_t

        def q_inference(
            weights_and_bias: list[np.ndarray],
            inputs: np.ndarray,
            hidden_state: np.ndarray,
            cell_state: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            inputs = np.array(inputs)

            for i in range(len(inputs[0])):
                hidden_state, cell_state = qlstm_cell(
                    inputs[0][i], hidden_state, cell_state, weights_and_bias
                )

            linear_w = to_fixed_point(np.expand_dims(weights_and_bias[4], axis=0))
            linear_b = to_fixed_point(np.expand_dims(weights_and_bias[5], axis=0))

            output = np.matmul(linear_w, hidden_state) / fixed_point_one + linear_b

            return output, hidden_state, cell_state

        try:
            test_mse = []
            for test_inputs, test_labels in test_loader:
                hidden_state = np.zeros([20, 1])
                cell_state = np.zeros([20, 1])

                test_outputs, hidden_state, cell_state = q_inference(
                    weights_and_bias, test_inputs, hidden_state, cell_state
                )

                mse = (to_float(test_outputs) - np.array(test_labels)) ** 2
                test_mse.append(mse)

            avg_mse = np.mean(test_mse)
            frac_bits_mse.append(avg_mse)
        except ValueError:
            break

    min_mse_idx = np.argmin(frac_bits_mse)
    best_frac_bits = frac_bits_choices[min_mse_idx]

    return FixedPoint.get_factory(total_bits=total_bits, frac_bits=best_frac_bits)


def main() -> None:
    args = read_commandline_args()

    model = LSTMModel()

    work_library_name = "xil_defaultlib"
    fixed_point_factory = determine_optimal_fixed_point_factory(
        model=model,
        saved_model_path="elasticai/creator/examples/assets/determine_fixed_point/model_v66",
        data_path="elasticai/creator/examples/assets/determine_fixed_point/pems-4w.csv",
        sensor_idx=4291,
        batch_size=1,
        datapoints_per_sample=6,
    )
    translation_args = dict(
        LSTM=LSTMTranslationArgs(
            fixed_point_factory=fixed_point_factory,
            sigmoid_resolution=(-2.5, 2.5, 256),
            tanh_resolution=(-1, 1, 256),
            work_library_name=work_library_name,
        ),
        Linear=Linear1dTranslationArgs(
            fixed_point_factory=fixed_point_factory,
            work_library_name=work_library_name,
        ),
    )

    code_repr = translator.translate_model(
        model=model,
        translation_args=translation_args,
        build_function_mapping=DEFAULT_BUILD_FUNCTION_MAPPING,
    )

    translator.save_code(code_repr=code_repr, path=args.build_dir)


if __name__ == "__main__":
    main()
