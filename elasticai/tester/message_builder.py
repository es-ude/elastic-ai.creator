from typing import Iterable

from elasticai.tester.commands import Command
from elasticai.tester.message import Message


def _batched_bytes(iterable: bytes | bytearray, batch_size: int):
    b = bytearray()
    for i in iterable:
        b.append(i)
        if len(b) == batch_size:
            yield b
            b.clear()
    if len(b) > 0:
        yield b


class MessageBuilder:
    def __init__(self) -> None:
        self.flash_chunk_size = 256
        self.flash_address = 0
        self.data = bytes()
        self.num_read_bytes = 1
        self.byte_order = "big"
        self._NUM_BYTES_FOR_LENGTH = 4
        self.command = Command.NAK
        self.expected_response_size = 1

    @property
    def payload(self) -> bytes:
        return self.data

    @payload.setter
    def payload(self, v: bytes) -> None:
        self.data = v

    def build(self) -> Iterable[Message]:
        match self.command:
            case (
                Command.NAK
                | Command.ACK
                | Command.READ_SKELETON_ID
                | Command.GET_FLASH_CHUNK_SIZE
            ):
                yield self._command_without_payload()
            case Command.WRITE_TO_FLASH:
                yield from self._write_to_flash()
            case Command.READ_FROM_FLASH:
                yield self._read_from_flash()
            case Command.FPGA_LEDS | Command.MCU_LEDS | Command.FPGA_POWER:
                yield self._simple_message_with_payload()
            case Command.INFERENCE:
                yield from self._command_with_payload_and_response()
            case Command.DEPLOY_MODEL:
                yield self._deploy_model()
            case _:
                if self.expected_response_size == 0 and len(self.data) == 0:
                    yield from self._command_without_payload()
                elif self.expected_response_size == 0:
                    yield from self._simple_message_with_payload()
                else:
                    yield from self._command_with_payload_and_response()

    def _get_number_in_bytes(self, number: int) -> bytes:
        return number.to_bytes(
            length=self._NUM_BYTES_FOR_LENGTH, byteorder=self.byte_order, signed=False
        )

    def _new_msg(self, data: bytes) -> Message:
        return Message(self.command, data, self.byte_order)

    def _command_without_payload(self) -> Message:
        return Message(self.command, bytes())

    def _simple_message_with_payload(self):
        return self._new_msg(self.data)

    @property
    def _address_in_bytes(self) -> bytes:
        return self._get_number_in_bytes(self.flash_address)

    @property
    def _data_size_in_bytes(self) -> bytes:
        return self._get_number_in_bytes(len(self.data))

    @property
    def _num_read_bytes_in_bytes(self) -> bytes:
        return self._get_number_in_bytes(len(self.num_read_bytes))

    def _generate_message_chunks(self):
        for chunk in _batched_bytes(self.data, self.flash_chunk_size):
            yield self._new_msg(chunk)

    @property
    def inference_input_length(self) -> int:
        return len(self.data)

    def _deploy_model(self) -> Message:
        return self._new_msg(
            b"".join((self._get_number_in_bytes(self.flash_address), self.data))
        )

    def _command_with_payload_and_response(self) -> Iterable[Message]:
        yield self._new_msg(
            b"".join(
                map(
                    self._get_number_in_bytes,
                    (
                        self.inference_input_length,
                        self.expected_response_size,
                    ),
                )
            )
        )
        yield from self._generate_message_chunks()

    def _write_to_flash(self) -> Iterable[Message]:
        yield self._new_msg(
            b"".join((self._address_in_bytes, self._data_size_in_bytes)),
        )
        yield from self._generate_message_chunks()

    def _read_from_flash(self) -> Message:
        yield self._new_msg(
            b"".join((self._address_in_bytes, self._num_read_bytes_in_bytes)),
        )
