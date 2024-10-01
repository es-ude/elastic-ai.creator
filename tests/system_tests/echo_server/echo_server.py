import atexit
import pathlib
import subprocess
import time
import tomllib

import serial  # type: ignore
import torch
from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port  # type: ignore

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.vhdl.system_integrations.plug_and_play_solution_ENV5 import (
    FirmwareEchoServerSkeletonV2,
)
from tests.system_tests.helper.parse_tensors_to_bytearray import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)


def build_vhdl_source(output_dir: str, num_inputs: int) -> bytearray:
    f = FirmwareEchoServerSkeletonV2(num_inputs=num_inputs, bitwidth=8)
    f.save_to(OnDiskPath(output_dir))
    skeleton_id_as_bytearray = bytearray()
    for x in f.skeleton_id:
        skeleton_id_as_bytearray.extend(
            x.to_bytes(length=1, byteorder="little", signed=False)
        )
    return skeleton_id_as_bytearray


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
    print(f"Sending binfile to {binfile_address=}: {finished=}")
    return finished


def exit_handler(cdc_port: serial.Serial):
    cdc_port.close()
    print(f"closed {cdc_port.port=}")


if __name__ == "__main__":
    output_dir = vhdl_dir = "./tests/system_tests/echo_server/build_dir"
    binfile_dir = "./tests/system_tests/echo_server/build_dir_output/"
    binfile_path = pathlib.Path(binfile_dir + "output/env5_top_reconfig.bin")

    total_bits = 8
    frac_bits = 2
    batches = 3
    num_inputs = 4
    num_outputs = num_inputs

    skeleton_id_as_bytearray = build_vhdl_source(output_dir, num_inputs)

    vivado_build_binfile(output_dir, binfile_dir)

    # serial_con = serial.Serial(get_env5_port())
    serial_con = serial.Serial("/dev/tty.usbmodem101")
    atexit.register(exit_handler, serial_con)
    binfile_address = 0

    urc = UserRemoteControl(device=serial_con)

    print("Send and deploy model")
    successful = urc.send_and_deploy_model(
        binfile_path, binfile_address, skeleton_id_as_bytearray
    )
    successful = urc.deploy_model(binfile_address, skeleton_id_as_bytearray)
    print(f"Model deployed {successful=}")

    fxp_conf = FixedPointConfig(total_bits, frac_bits)
    x = torch.randn(batches, 1, num_inputs)
    inputs = fxp_conf.as_rational(fxp_conf.as_integer(x))
    inference_data = parse_fxp_tensor_to_bytearray(inputs, total_bits, frac_bits)

    print(f"Start inference")
    inference_result = list()
    for batch_data in inference_data:
        batch_result = urc.inference_with_data(batch_data, num_outputs)
        inference_result.append(batch_result)

    actual_result = parse_bytearray_to_fxp_tensor(
        inference_result, total_bits, frac_bits, inputs.shape
    )
    print(f"{actual_result=}")
    print(f"{inputs + 1/(2**frac_bits)=}")
    assert torch.equal(actual_result, inputs + 1 / (2**frac_bits))
    print("Test successful")
