import logging

from elasticai.tester.commands import Command
from elasticai.tester.io_stream import IOStream
from elasticai.tester.message import Message


class MessageIO:
    def __init__(
        self,
        io_stream: IOStream,
        byte_order: str,
        ack: Message,
        nak: Message,
        max_trials: int = 5,
    ) -> None:
        self._stream = io_stream
        self._byte_order = byte_order
        self._ACK = ack
        self._NAK = nak
        self._last_msg = ack
        self._last_checksum = ack.checksum
        self._max_trials = max_trials
        self._logger = logging.getLogger(__name__)

    def _receive_new_msg(self) -> Message:
        raw_command = self._do_read(Message.NUM_BYTES_COMMAND)
        raw_length = self._do_read(Message.NUM_BYTES_PAYLOAD_SIZE)
        length = int.from_bytes(raw_length, self._byte_order)
        command = int.from_bytes(raw_command, self._byte_order)
        if command in Command:
            command = Command(command)
        data = self._do_read(length + Message.NUM_BYTES_CHECKSUM)
        data, self._last_checksum = (
            data[:-1],
            data[-1:],
        )  # data[i] returns int instead of byte, so we do data[i:]
        self._last_msg = Message(command, data, self._byte_order)
        self._logger.debug(f"received new message: {self._last_msg}", stacklevel=3)

    def _checksum_was_valid(self) -> bool:
        return self._last_checksum == self._last_msg.checksum

    def _send_ack(self):
        self._do_write(self._ACK.to_bytes())

    def _send_nak(self):
        self._do_write(self._NAK.to_bytes())

    def _acknowledge_required(self):
        return self._last_msg.command not in (Command.NAK, Command.ACK)

    def read(self) -> Message:
        self._receive_new_msg()
        if self._acknowledge_required():
            if self._checksum_was_valid():
                self._send_ack()
            else:
                self._send_nak()
        if not self._checksum_was_valid():
            self._logger.debug("invalid checksum", stacklevel=1)

        return self._last_msg

    def _do_write(self, data: bytes) -> None:
        self._logger.debug(f"writing raw data {data}", stacklevel=2)
        self._stream.write(data)

    def _do_read(self, num_bytes: int) -> bytes:
        self._logger.debug(f"reading {num_bytes} bytes", stacklevel=2)
        data = self._stream.read(num_bytes)
        self._logger.debug(f"read data {data}", stacklevel=2)
        return data

    def write(self, msg: Message) -> None:
        self._logger.debug(f"sending {msg}", stacklevel=3)
        b = msg.to_bytes()
        self._do_write(b)
        rec = self._NAK
        for t in range(self._max_trials):
            rec = self.read()
            if rec == self._ACK:
                self._logger.debug(f"received valid ack on trial {t}")
                break
            elif t < self._max_trials - 1:
                self._logger.debug("retry sending", stacklevel=2)
                self._do_write(b)
        if rec == self._NAK:
            raise IOError(
                f"number of trials exceeded for writing message {msg.to_bytes()}"
            )
