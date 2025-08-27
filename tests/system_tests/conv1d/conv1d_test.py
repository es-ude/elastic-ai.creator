import subprocess
import tomllib
from pathlib import Path

import pytest
import serial
import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Conv1d
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5
from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port
from tests.system_tests.helper.parse_tensors_to_bytearray import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)


def create_vhd_files(
    output_dir: str,
    num_inputs: int,
    num_outputs: int,
    num_in_channels: int,
    num_out_channels: int,
    kernel_size: int,
    total_bits: int,
    frac_bits: int,
    skeleton_id: list[int],
) -> Sequential:
    nn = Sequential(
        Conv1d(
            total_bits=total_bits,
            frac_bits=frac_bits,
            in_channels=num_in_channels,
            out_channels=num_out_channels,
            signal_length=num_inputs,
            kernel_size=kernel_size,
        )
    )
    nn[0].weight.data = torch.ones_like(nn[0].weight) * 2
    nn[0].bias.data = torch.ones_like(nn[0].bias) * -1
    destination = OnDiskPath(output_dir)
    my_design = nn.create_design("nn")
    my_design.save_to(destination.create_subpath("srcs"))

    firmware = FirmwareENv5(
        network=my_design,
        x_num_values=num_inputs,
        y_num_values=num_outputs,
        id=skeleton_id,
        skeleton_version="v2",
    )
    firmware.save_to(destination)
    return nn


def vivado_build_binfile(input_dir: str, binfile_dir: str):
    print(f"Building binfile in {binfile_dir}")
    with open("./tests/system_tests/vivado_build_server_conf.toml", "rb") as f:
        config = tomllib.load(f)
    out = subprocess.run(
        [
            "bash",
            "./utils/autobuild_binfile_vivado2021.1.sh",
            config["username"],
            config["ip"],
            input_dir,
            binfile_dir,
        ],
        capture_output=True,
    )
    print(f"{out.stdout=}")


@pytest.mark.hardware
def test_conv1d_runtime():
    # dev_address = "COM8"
    dev_address = get_env5_port()
    vhdl_dir = "./tests/system_tests/conv1d/build_dir"
    binfile_dir = "./tests/system_tests/conv1d/build_dir_output"
    binfile_path = Path(binfile_dir + "/output/env5_top_reconfig.bin")
    skeleton_id = list(range(15, -1, -1))
    skeleton_id_as_bytearray = bytearray()
    for x in skeleton_id:
        skeleton_id_as_bytearray.extend(
            x.to_bytes(length=1, byteorder="little", signed=False)
        )

    total_bits = 8
    frac_bits = 2
    num_inputs = 5
    kernel_size = 3
    num_in_channels = 1
    num_out_channels = 1
    # batches = 2
    num_outputs = num_inputs - kernel_size + 1
    nn = create_vhd_files(
        output_dir=vhdl_dir,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_in_channels=num_in_channels,
        num_out_channels=num_out_channels,
        kernel_size=kernel_size,
        total_bits=total_bits,
        frac_bits=frac_bits,
        skeleton_id=skeleton_id,
    )

    fxp_conf = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )

    # inputs = fxp_conf.as_rational(fxp_conf.as_integer(torch.rand(batches, num_in_channels, num_inputs)))
    inputs = fxp_conf.as_rational(
        fxp_conf.cut_as_integer(
            torch.Tensor(
                [
                    [[0.0, 0.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0, 1.0]],
                    [[2.0, 0.0, 0.0, 0.0, 0.0]],
                    [[1.0, 1.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0, 1.0]],
                    [[3.0, 0.0, 0.0, 0.0, 0.0]],
                    [[1.0, 2.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 2.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 2.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0, 2.0]],
                    [[1.0, 1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0, 1.0, 1.0]],
                    [[-1.0, 0.0, 0.0, 0.0, 0.0]],
                    [[0.0, -1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, -1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, -1.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0, -1.0]],
                    [[2.0, 0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 2.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 2.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 2.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0, 2.0]],
                ]
            )
        )
    )

    expected_outputs = nn(inputs)
    vivado_build_binfile(vhdl_dir, binfile_dir)

    with serial.Serial(dev_address) as serial_con:
        flash_start_address = 0

        urc = UserRemoteControl(device=serial_con)
        urc.send_and_deploy_model(
            binfile_path, flash_start_address, skeleton_id_as_bytearray
        )
        # urc.deploy_model(flash_start_address, skeleton_id_as_bytearray)

        batch_data = parse_fxp_tensor_to_bytearray(inputs, total_bits, frac_bits)
        inference_result = list()
        for i, sample in enumerate(batch_data):
            batch_result = urc.inference_with_data(
                sample, num_outputs * num_out_channels
            )
            my_result = parse_bytearray_to_fxp_tensor(
                [batch_result],
                total_bits,
                frac_bits,
                (1, num_out_channels, num_outputs),
            )
            print(
                f"Batch {i}: {my_result} == {expected_outputs[i]}, for input"
                f" {inputs[i]}"
            )

            inference_result.append(batch_result)
        actual_result = parse_bytearray_to_fxp_tensor(
            inference_result, total_bits, frac_bits, expected_outputs.shape
        )

        assert torch.equal(actual_result, expected_outputs)
