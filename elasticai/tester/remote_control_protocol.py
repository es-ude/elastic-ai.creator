from elasticai.tester.commands import Command
from elasticai.tester.io_stream import IOStream
from elasticai.tester.message_builder import MessageBuilder
from elasticai.tester.message_chunks_receiver import MessageChunksReceiver
from elasticai.tester.message_io import MessageIO


class RemoteControlProtocol:
    def __init__(self, device: IOStream) -> None:
        self._msg_builder = MessageBuilder()
        self._msg_builder.command = Command.NAK
        self._nak = next(self._msg_builder.build())
        self._msg_builder.command = Command.ACK
        self._ack = next(self._msg_builder.build())
        self._device = MessageIO(
            device,
            max_trials=5,
            ack=self._ack,
            nak=self._nak,
            byte_order=self._msg_builder.byte_order,
        )
        self._chunk_receiver = MessageChunksReceiver(
            self._device, byte_order=self._msg_builder.byte_order
        )

    def request_flash_chunk_size(self) -> None:
        self._msg_builder.command = Command.GET_FLASH_CHUNK_SIZE
        self._send()
        response = self._device.read()
        self._msg_builder.flash_chunk_size = response.payload_as_uint

    def fpga_power_on(self):
        self._msg_builder.command = Command.FPGA_POWER
        self._msg_builder.data = b"\xff"
        self._send()

    def fpga_power_off(self):
        self._msg_builder.command = Command.FPGA_POWER
        self._msg_builder.data = b"\x00"
        self._send()

    def predict(self, _input: bytes, output_length: int) -> bytes:
        self._msg_builder.command = Command.INFERENCE
        self._msg_builder.data = _input
        self._msg_builder.expected_response_size = output_length
        self._send()
        msg = self._device.read()
        return msg.payload

    def send_custom_command(
        self, cmd_id: int, payload: bytes, response_size: int = 0
    ) -> bytes:
        self._msg_builder.command = cmd_id
        self._msg_builder.data = payload
        self._msg_builder.expected_response_size = response_size
        self._send()
        if response_size > 0:
            msg = self._device.read()
            return msg.payload
        return None

    def read_skeleton_id(self) -> bytes:
        self._msg_builder.command = Command.READ_SKELETON_ID
        self._send()
        response = self._device.read()
        return response.payload

    def set_fpga_leds(self, leds: tuple[bool, bool, bool]) -> None:
        self._msg_builder.command = Command.FPGA_LEDS
        self._msg_builder.data = self._build_led_byte(leds)
        self._send()

    def set_mcu_leds(self, leds: tuple[bool, bool, bool]) -> None:
        self._msg_builder.command = Command.MCU_LEDS
        self._msg_builder.data = self._build_led_byte(leds)
        self._send()

    def deploy_model(self, sector: int, skeleton_id: bytes):
        self._msg_builder.flash_address = sector * self._msg_builder.flash_chunk_size
        self._msg_builder.command = Command.DEPLOY_MODEL
        self._msg_builder.data = skeleton_id
        self._send()

    def write_to_flash(self, sector: int, data: bytes):
        self._msg_builder.flash_address = sector * self._msg_builder.flash_chunk_size
        self._msg_builder.command = Command.WRITE_TO_FLASH
        self._msg_builder.data = data
        self._send()

    def read_from_flash(self, sector: int, result_size: int) -> bytes:
        self._msg_builder.command = Command.READ_FROM_FLASH
        self._msg_builder.flash_address = sector * self._msg_builder.flash_chunk_size
        self._msg_builder.num_read_bytes = result_size
        self._send()
        self._chunk_receiver.receive()
        return self._chunk_receiver.data

    def _send(self) -> None:
        for i, msg in enumerate(self._msg_builder.build()):
            self._device.write(msg)

    def _build_led_byte(self, leds: tuple[bool, bool, bool]) -> bytes:
        led_byte = 0
        for led in leds:
            led = 1 if led else 0
            led_byte = (led_byte << 1) | led
        return led_byte.to_bytes(length=1, byteorder="big", signed=False)

    def __enter__(self):
        self.request_flash_chunk_size()
        return self

    def __exit__(self):
        pass
