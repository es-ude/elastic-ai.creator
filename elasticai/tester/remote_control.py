import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from serial import Serial

from elasticai.tester.io_stream import IOStream
from elasticai.tester.remote_control_protocol import RemoteControlProtocol


class RemoteControl:
    def __init__(self, device: IOStream):
        self._rcp = RemoteControlProtocol(device)
        self._rcp.request_flash_chunk_size()

    def deploy_model(self, flash_sector: int, path_to_bitstream_dir: str):
        Path(path_to_bitstream_dir)
        # with open(directory / "meta.json", "r") as f:
        #     meta = json.load(f)
        # skeleton_id = bytes.fromhex(meta["skeleton_id"])
        self._rcp.deploy_model(
            flash_sector, bytes.fromhex("00000000000000000000000000000000")
        )
        print(self._rcp.read_skeleton_id())

    def upload_bitstream(self, flash_sector: int, path_to_bitstream_dir: str):
        directory = Path(path_to_bitstream_dir)
        with open(directory, "r+b") as f:
            config = f.read()
        self._rcp.write_to_flash(sector=flash_sector, data=config)

    def read_from_flash(self, flash_sector: int, num_bytes: int) -> bytes:
        raise NotImplementedError

    def predict(self, data: bytes, result_size: int) -> bytes:
        return self._rcp.predict(data, result_size)

    def read_skeleton_id(self) -> str:
        return self._rcp.read_skeleton_id().hex(sep=" ")

    def fpga_power_on(self) -> None:
        self._rcp.fpga_power_on()

    def fpga_power_off(self) -> None:
        self._rcp.fpga_power_off()

    def fpga_leds(self, leds: tuple[bool, bool, bool]):
        self._rcp.set_fpga_leds(leds)

    def mcu_leds(self, leds: tuple[bool, bool, bool]):
        self._rcp.set_mcu_leds(leds)

    def send_command(self, cmd_id: int, data: str, result_size: int):
        return self._rcp.send_custom_command(
            cmd_id, payload=bytes.fromhex(data), response_size=result_size
        )


def setup_arg_parser(parser: ArgumentParser):
    def upload(subparsers):
        p = subparsers.add_parser("upload")
        p.add_argument(
            "-a", "--address", type=int, help="Destination address in flash", default=0
        )
        p.add_argument("binfile", type=Path, help="Binary file")

        def fn(rc: RemoteControl, args) -> None:
            rc.upload_bitstream(args.address, args.binfile)

        p.set_defaults(func=fn)

    def read_skeleton_id(subparsers):
        p = subparsers.add_parser("get_id")

        def fn(rc: RemoteControl, _) -> None:
            print(rc.read_skeleton_id())

        p.set_defaults(func=fn)

    def predict(subparsers):
        p = subparsers.add_parser("predict")
        p.add_argument("-rs", "--result_size", type=int)
        p.add_argument("data", type=str)

        def read_from_stdin() -> bytes:
            return sys.stdin.buffer.read()

        def read_from_cmd_line(args) -> None:
            return bytes.fromhex(args.data)

        def fn(rc: RemoteControl, args):
            if args.data == "--":
                data = read_from_stdin()
            else:
                data = read_from_cmd_line(args)

            result = rc.predict(data, args.result_size)
            print(result)

        p.set_defaults(func=fn)

    def deploy(subparsers):
        p = subparsers.add_parser("deploy")
        p.add_argument("address", type=int)
        p.add_argument("path_to_bitstream_dir")

        def fn(rc: RemoteControl, args) -> None:
            rc.deploy_model(args.address, args.path_to_bitstream_dir)

        p.set_defaults(func=fn)

    def mcu_leds(subparsers):
        p = subparsers.add_parser("mcu_leds")
        p.add_argument("leds", type=str)

        def fn(rc: RemoteControl, args) -> None:
            rc.mcu_leds([led == "1" for led in args.leds])

        p.set_defaults(func=fn)

    def fpga_leds(subparsers):
        p = subparsers.add_parser("fpga_leds")
        p.add_argument("leds", type=str)

        def fn(rc: RemoteControl, args) -> None:
            rc.fpga_leds([led == "1" for led in args.leds])

        p.set_defaults(func=fn)

    def fpga_on(subparsers):
        p = subparsers.add_parser("fpga_on")

        def fn(rc: RemoteControl, _):
            rc.fpga_power_on()

        p.set_defaults(func=fn)

    def fpga_off(subparsers):
        p = subparsers.add_parser("fpga_off")

        def fn(rc: RemoteControl, _):
            rc.fpga_power_off()
            print("powering off")

        p.set_defaults(func=fn)

    def command(subparsers):
        p = subparsers.add_parser("command")
        p.add_argument("-rs", "--result_size", type=int)
        p.add_argument("-id", "--command_id", type=int)
        p.add_argument("data", type=str)

        def read_from_stdin() -> bytes:
            return sys.stdin.buffer.read()

        def read_from_cmd_line(args) -> None:
            return args.data

        def fn(rc: RemoteControl, args):
            if args.data == "--":
                data = read_from_stdin()
            else:
                data = read_from_cmd_line(args)

            result = rc.send_command(args.command_id, data, args.result_size)
            print(result)

        p.set_defaults(func=fn)

    actions = (
        deploy,
        predict,
        mcu_leds,
        upload,
        read_skeleton_id,
        fpga_on,
        fpga_off,
        fpga_leds,
        command,
    )

    arg_parser.add_argument("-p", "--port", type=str, help="serial port", required=True)
    arg_parser.add_argument("-v", "--verbose", help="debug msgs", action="store_true")
    subparsers = arg_parser.add_subparsers(required=True, help="available actions")
    for submenu in actions:
        submenu(subparsers)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    setup_arg_parser(arg_parser)
    args = arg_parser.parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            force=True,
            format="{levelname}:: {pathname}:{lineno}\n\t{message}",
            style="{",
        )
    with Serial(args.port, timeout=5) as device:
        rc = RemoteControl(device)
        args.func(rc, args)
