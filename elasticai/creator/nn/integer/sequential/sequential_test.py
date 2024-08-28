import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
)
from elasticai.creator.nn.integer.relu.relu import ReLU
from elasticai.creator.nn.integer.sequential.sequential import Sequential


@pytest.fixture
def linear_layer_0():
    linear_layer_0 = Linear(
        name="linear_0", in_features=3, out_features=10, bias=True, quant_bits=8
    )

    linear_layer_0.weight.data = torch.tensor(
        [
            [0.5, -0.5, 0.3],
            [0.2, -0.2, 0.1],
            [0.4, -0.4, 0.2],
            [0.6, -0.6, 0.3],
            [0.7, -0.7, 0.4],
            [0.8, -0.8, 0.5],
            [0.9, -0.9, 0.6],
            [1.0, -1.0, 0.7],
            [1.1, -1.1, 0.8],
            [1.2, -1.2, 0.9],
        ],
        dtype=torch.float32,
    )
    linear_layer_0.bias.data = torch.tensor(
        [0.5, -0.5, 0.3, 0.2, -0.2, 0.1, 0.4, -0.4, 0.2, 0.6], dtype=torch.float32
    )
    return linear_layer_0


@pytest.fixture
def relu_layer_0():
    return ReLU(name="relu_0", quant_bits=8)


@pytest.fixture
def linear_layer_1():
    linear_layer_1 = Linear(
        name="linear_1", in_features=10, out_features=1, bias=True, quant_bits=8
    )

    linear_layer_1.weight.data = torch.tensor(
        [[0.5, -0.5, 0.3, 0.2, -0.2, 0.1, 0.4, -0.4, 0.2, 0.6]],
        dtype=torch.float32,
    )

    linear_layer_1.bias.data = torch.tensor([0.5], dtype=torch.float32)
    return linear_layer_1


@pytest.fixture
def sequential_instance(linear_layer_0, relu_layer_0, linear_layer_1):
    layers = nn.ModuleList()
    layers.append(linear_layer_0)
    layers.append(relu_layer_0)
    layers.append(linear_layer_1)

    return Sequential(*layers, name="network", quant_data_file_dir=None)


@pytest.fixture
def inputs():
    return torch.tensor(
        [
            [-1.0, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def eps():
    return torch.tensor(
        (torch.finfo(torch.float32).eps),
        dtype=torch.float32,
        requires_grad=False,
    )


@pytest.fixture
def q_inputs(inputs, eps):
    min_float = inputs.min()
    max_float = inputs.max()
    min_quant = torch.tensor([-128], dtype=torch.int32)
    max_quant = torch.tensor([127], dtype=torch.int32)

    scale_factor, zero_point, min_float, max_float = calculate_asymmetric_quant_params(
        min_float, max_float, min_quant, max_quant, eps
    )

    q_inputs = inputs / scale_factor + zero_point
    q_inputs = q_inputs.round_().clamp(min=min_quant.item(), max=max_quant.item())
    q_inputs = q_inputs.to(torch.int32)
    return q_inputs


@pytest.fixture
def dq_outputs(inputs, q_inputs, linear_layer_0, relu_layer_0, linear_layer_1):
    linear_layer_0.train()
    tmp = linear_layer_0.forward(inputs)
    linear_layer_0.eval()
    linear_layer_0.precompute()
    tmp_q = linear_layer_0.int_forward(q_inputs)

    relu_layer_0.train()
    relu_layer_0.forward(tmp)
    relu_layer_0.eval()
    tmp_q = relu_layer_0.int_forward(tmp_q)

    linear_layer_1.train()
    tmp = linear_layer_1.forward(tmp)
    linear_layer_1.eval()
    linear_layer_1.precompute()
    tmp_q = linear_layer_1.int_forward(tmp_q)

    dq_outputs = linear_layer_1.outputs_QParams.dequantize(tmp_q)
    return dq_outputs


def test_QParams_transfering_in_forward(sequential_instance, inputs):
    sequential_instance.train()
    _ = sequential_instance(inputs)

    assert (
        sequential_instance.submodules[1].inputs_QParams
        == sequential_instance.submodules[0].outputs_QParams
    )
    assert (
        sequential_instance.submodules[2].inputs_QParams
        == sequential_instance.submodules[1].outputs_QParams
    )


def test_precompute(sequential_instance, inputs):
    sequential_instance.train()
    _ = sequential_instance(inputs)
    sequential_instance.eval()
    sequential_instance.precompute()
    for submodule in sequential_instance.submodules:
        if hasattr(submodule, "precompute"):
            assert submodule.precomputed == True


def test_quantize_inputs(sequential_instance, inputs, q_inputs):
    sequential_instance.train()
    _ = sequential_instance(inputs)
    sequential_instance.eval()
    sequential_instance.precompute()
    actual_q_inputs = sequential_instance.quantize_inputs(inputs)

    expected_q_inputs = q_inputs
    assert torch.allclose(actual_q_inputs, expected_q_inputs)


def test_dequantize_outputs(sequential_instance, inputs, dq_outputs):
    sequential_instance.train()
    _ = sequential_instance(inputs)
    sequential_instance.eval()
    sequential_instance.precompute()
    q_inputs = sequential_instance.quantize_inputs(inputs)
    q_outputs = sequential_instance.int_forward(q_inputs)
    actual_dq_outputs = sequential_instance.dequantize_outputs(q_outputs)

    expected_dq_outputs = dq_outputs
    assert torch.allclose(actual_dq_outputs, expected_dq_outputs, atol=1e-3)


def test_save_quant_data_in_int_forward(
    linear_layer_0, relu_layer_0, linear_layer_1, inputs, q_inputs
):
    layers = nn.ModuleList([linear_layer_0, relu_layer_0, linear_layer_1])

    with tempfile.TemporaryDirectory() as tmp_quant_data_file_dir:
        quant_data_file_dir = Path(tmp_quant_data_file_dir) / "quant_data"
        quant_data_file_dir.mkdir(parents=True, exist_ok=True)

        sequential_instance_with_dir = Sequential(
            *layers, name="network", quant_data_file_dir=quant_data_file_dir
        )
        sequential_instance_with_dir.train()
        sequential_instance_with_dir(inputs)
        sequential_instance_with_dir.eval()
        sequential_instance_with_dir.precompute()
        sequential_instance_with_dir.int_forward(q_inputs)

        files = {}
        for file_path in quant_data_file_dir.iterdir():
            if file_path.is_file():
                try:
                    with file_path.open("r") as file:
                        files[file_path.name] = file.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

        expected_files = {
            "network_q_x.txt",
            "linear_0_q_x.txt",
            "linear_0_q_y.txt",
            "relu_0_q_x.txt",
            "relu_0_q_y.txt",
            "linear_1_q_x.txt",
            "linear_1_q_y.txt",
            "network_q_y.txt",
        }

        actual_files = set(files.keys())
        assert expected_files == actual_files
