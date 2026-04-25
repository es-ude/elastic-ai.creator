import pytest
import torch

from elasticai.creator_plugins.lutron_filter.nn import Binarize
from elasticai.creator_plugins.lutron_filter.precompute.truth_table_generation import (
    generate_input_tensor_1d as generate_1d_input_tensor,
)
from elasticai.creator_plugins.lutron_filter.tensor_conversion import (
    torch1d_input_tensor_to_grouped_strings,
)


@pytest.mark.parametrize(
    "in_channels, kernel_size, groups, expected",
    [
        (1, 1, 1, [[[0]], [[1]]]),
        (1, 2, 1, [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]),
        (2, 1, 1, [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]),
        (2, 1, 2, [[[0], [0]], [[1], [1]]]),
        (
            2,
            2,
            2,
            [[[0, 0], [0, 0]], [[0, 1], [0, 1]], [[1, 0], [1, 0]], [[1, 1], [1, 1]]],
        ),
    ],
)
def test_input_tensor_generation(in_channels, kernel_size, groups, expected):
    expected = (torch.tensor(expected) * 2 - 1).tolist()
    result = (
        generate_1d_input_tensor(in_channels, kernel_size, groups)
        .to(torch.int)
        .tolist()
    )
    assert result == expected


def test_inference_on_truth_tables_and_pytorch_leads_to_same_result():
    channels = 6
    conv = torch.nn.Conv1d(
        in_channels=channels, out_channels=channels, kernel_size=4, groups=channels
    )
    input_tensor = generate_1d_input_tensor(channels, 4, channels)
    input_tensor.requires_grad = False
    bin = Binarize()
    result = bin(conv(input_tensor))

    def to_logics(x):
        return ((x + 1) / 2).to(torch.int)

    input_strings = torch1d_input_tensor_to_grouped_strings(
        to_logics(input_tensor), groups=channels
    )
    result_strings = torch1d_input_tensor_to_grouped_strings(
        to_logics(result), groups=channels
    )
    kernel_maps = [
        dict(zip(inputs, outputs))
        for inputs, outputs in zip(input_strings, result_strings)
    ]
    expected_result_tensor = bin(
        conv(
            torch.tensor(
                [
                    [-1.0, 1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0, -1],
                    [1.0, -1.0, 1.0, -1],
                    [1.0, -1.0, -1, 1],
                    [1.0, 1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0],
                ]
            )
        )
    )
    expected = "".join(map(str, to_logics(expected_result_tensor).flatten().tolist()))
    for kernel_map in kernel_maps:
        print("{")
        for input, output in kernel_map.items():
            print("\t", input, ":", output)
        print("}")
    actual_result = list(
        kernel_maps[i][x]
        for i, x in enumerate(("0111", "0110", "1010", "1001", "1110", "0000"))
    )
    actual_result = "".join(actual_result)
    assert expected == actual_result
