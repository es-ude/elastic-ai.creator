import logging

from elasticai.tester.commands import Command
from elasticai.tester.message_io import MessageIO


class MessageChunksReceiver:
    def __init__(self, stream: MessageIO, byte_order: str) -> None:
        self._stream = stream
        self.command = Command.NAK
        self.data: bytes | bytearray = bytes()
        self._byte_order = byte_order
        self._logger = logging.getLogger(__name__)

    def receive(self):
        msg = self._stream.read()
        expected_num_bytes = msg.payload_as_uint
        received_num_bytes = 0
        self.command = msg.command
        self.data = bytearray()
        while received_num_bytes < expected_num_bytes:
            msg = self._stream.read()
            received_num_bytes += msg.payload_size
            self.data.extend(msg.payload)
            self._logger.debug(f"got {msg.payload}")
