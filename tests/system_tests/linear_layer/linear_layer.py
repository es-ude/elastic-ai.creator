import atexit
import subprocess
import time
import tomllib

import serial  # type: ignore
import torch
from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port  # type: ignore

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.fixed_point.linear import Linear
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5
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
    nn[0].weight.data = torch.ones_like(nn[0].weight)*3
    nn[0].bias.data = torch.ones_like(nn[0].bias)
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
    print(out.stdout)


def send_binfile(
    local_urc: UserRemoteControl, binfile_address: int, file_dir: str
) -> bool:
    print(f"Sending binfile to {binfile_address=}")
    with open(file_dir + "output/env5_top_reconfig.bin", "rb") as file:
        binfile: bytes = file.read()
    finished = local_urc.send_data_to_flash(binfile_address, bytearray(binfile))
    print(f"Sending binfile to {binfile_address=}: {finished=}")
    return finished


def exit_handler(cdc_port: serial.Serial):
    cdc_port.close()
    print(f"closed {cdc_port.port=}")


if __name__ == "__main__":
    #torch.manual_seed(1)
    total_bits = 8
    frac_bits = 2
    num_inputs = 4
    num_outputs = 2
    batches = 4
    skeleton_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    skeleton_id_as_bytearray = bytearray()
    for x in skeleton_id:
        skeleton_id_as_bytearray.extend(
            x.to_bytes(length=1, byteorder="little", signed=False)
        )

    vhdl_dir = "./tests/system_tests/linear_layer/build_dir"
    binfile_dir = "./tests/system_tests/linear_layer/build_dir_output/"
    nn = create_vhd_files(
        vhdl_dir, num_inputs, num_outputs, total_bits, frac_bits, skeleton_id
    )
    #vivado_build_binfile(vhdl_dir, binfile_dir)

    fxp_conf = FixedPointConfig(total_bits, frac_bits)
    inputs = fxp_conf.as_rational(
        fxp_conf.as_integer(torch.rand(batches, 1, num_inputs))
    )
    inputs = fxp_conf.as_rational(fxp_conf.as_integer(torch.Tensor([[[0.0, 0.0, 0.0, 0.0]],
                                                                    [[1.0, 0.0, 0.0, 0.0]],
                                                                    [[0.0, 1.0, 0.0, 0.0]],
                                                                    [[0.0, 0.0, 1.0, 0.0]],
                                                                    [[0.0, 0.0, 0.0, 1.0]],])))
    expected_outputs = nn(inputs)

    serial_con = serial.Serial("/dev/tty.usbmodem2101")
    atexit.register(exit_handler, serial_con)

    binfile_address = 0
    urc = UserRemoteControl(device=serial_con)

    #send_binfile(urc, binfile_address, binfile_dir)
    urc.enV5RCP.fpga_power(True)
    time.sleep(0.1)
    inference_data = parse_fxp_tensor_to_bytearray(inputs, total_bits, frac_bits)
    inference_result = list()
    for batch_data in inference_data:
        batch_result = urc.inference_with_data(
            batch_data, num_outputs, binfile_address, skeleton_id_as_bytearray
        )
        inference_result.append(batch_result)
    actual_result = parse_bytearray_to_fxp_tensor(
        inference_result, total_bits, frac_bits, expected_outputs.shape
    )
    skeleton_id = urc.enV5RCP.read_skeleton_id()
    print(f"{skeleton_id=}")
    print(f"{skeleton_id_as_bytearray=}")
    print(f"{inputs=}")
    # print(f"")
    # print(f"{nn[0].weight=}")
    # print(f"{nn[0].bias=}")
    # print(f"weights = {[[0, 0.25, -0.5], [-0.5, -0.25, 0.25]]}")
    # print(f"bias = {[0.0, 0.5]}")
    print(f"{actual_result=}")
    print(f"{expected_outputs=}")
    assert torch.equal(actual_result, expected_outputs)
