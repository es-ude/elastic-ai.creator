import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest
import serial  # type: ignore
import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point.linear import Linear
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5
from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port  # type: ignore
from tests.system_tests.helper.parse_tensors_to_bytearray import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)


def create_vhd_files(
    output_dir: str,
    num_inputs: int,
    num_outputs: int,
    total_bits: int,
    frac_bits: int,
    skeleton_id: list[int],
) -> Sequential:
    nn = Sequential(
        Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=True,
            total_bits=total_bits,
            frac_bits=frac_bits,
        )
    )
    nn[0].weight.data = torch.ones_like(nn[0].weight) * 2
    nn[0].bias.data = torch.Tensor([1.0, 2.0, -1.0])
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


def vivado_build_binfile(input_dir: str, binfile_dir: str) -> None:
    print(f"Building binfile in {binfile_dir}")
    time.sleep(5)
    with open("./tests/system_tests/vivado_build_server_conf.toml", "rb") as f:
        config = tomllib.load(f)
    subprocess.run(
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
    # print(f"{out.stdout=}")


@pytest.mark.hardware
def test_linear_layer_env5():
    # --- Settings
    # dev_address = "COM8"
    dev_address = get_env5_port()
    vivado_build = True

    # torch.manual_seed(1)
    total_bits = 8
    frac_bits = 2
    num_inputs = 5
    num_outputs = 3
    skeleton_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # --- Processing
    # --- Creating the dummy
    input_tensor = torch.Tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]])
    for idx_array in range(0, num_inputs):
        for value in np.arange(-2, 2, 0.5):
            list_zeros = [0.0 for idx in range(0, num_inputs)]
            list_zeros[idx_array] = value
            input_tensor = torch.cat(
                (input_tensor, torch.Tensor([[list_zeros]])), dim=0
            )

    skeleton_id_as_bytearray = bytearray()
    for x in skeleton_id:
        skeleton_id_as_bytearray.extend(
            x.to_bytes(length=1, byteorder="little", signed=False)
        )

    # --- Preparing the VHDL code generation
    vhdl_dir = "./tests/system_tests/linear_layer/build_dir"
    binfile_dir = "./tests/system_tests/linear_layer/build_dir_output"
    binfile_path = Path(binfile_dir + "/output/env5_top_reconfig.bin")
    nn = create_vhd_files(
        vhdl_dir, num_inputs, num_outputs, total_bits, frac_bits, skeleton_id
    )

    fxp_params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    fxp_conf = FxpArithmetic(fxp_params)
    inputs = fxp_conf.as_rational(fxp_conf.cut_as_integer(input_tensor))
    expected_outputs = nn(inputs)

    # --- Building the code
    if vivado_build:
        vivado_build_binfile(vhdl_dir, binfile_dir)

    # --- Open Serial Communication to Device
    with serial.Serial(dev_address) as serial_con:
        flash_start_address = 0
        urc = UserRemoteControl(device=serial_con)
        urc.send_and_deploy_model(
            binfile_path, flash_start_address, skeleton_id_as_bytearray
        )
        # urc.deploy_model(flash_start_address, skeleton_id_as_bytearray)

        # --- Doing the test
        batch_data = parse_fxp_tensor_to_bytearray(inputs, total_bits, frac_bits)
        inference_result = list()
        state = False
        urc.fpga_leds(True, False, False, False)
        for i, sample in enumerate(batch_data):
            urc.fpga_leds(True, False, False, state)
            state = False if state else True

            batch_result = urc.inference_with_data(sample, num_outputs)
            my_result = parse_bytearray_to_fxp_tensor(
                [batch_result], total_bits, frac_bits, (1, 1, 3)
            )

            dev_inp = my_result
            dev_out = expected_outputs.data[i].view((1, 1, 3))
            if not torch.equal(dev_inp, dev_out):
                print(
                    f"Batch #{i:02d}: \t{dev_inp} == {dev_out}, (Delta ="
                    f" {dev_inp - dev_out}) \t\t\t\tfor input {inputs[i]}"
                )
                if i % 4 == 3:
                    print("\n")

            inference_result.append(batch_result)

        urc.fpga_leds(False, False, False, False)
        actual_result = parse_bytearray_to_fxp_tensor(
            inference_result, total_bits, frac_bits, expected_outputs.shape
        )

        assert torch.equal(actual_result, expected_outputs)
