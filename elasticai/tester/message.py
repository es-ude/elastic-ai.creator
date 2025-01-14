from elasticai.tester.commands import Command


class Message:
    NUM_BYTES_COMMAND: int = 1
    NUM_BYTES_PAYLOAD_SIZE: int = 4
    NUM_BYTES_CHECKSUM: int = 1
    PAYLOAD_OFFSET: int = NUM_BYTES_COMMAND + NUM_BYTES_PAYLOAD_SIZE
    MINIMUM_SIZE: int = NUM_BYTES_PAYLOAD_SIZE + NUM_BYTES_CHECKSUM + NUM_BYTES_COMMAND

    def __init__(self, command: Command, data: bytes, byte_order: str = "big") -> None:
        self.command = command
        self.payload = data
        self.byte_order = byte_order

    @classmethod
    def from_bytes(cls, raw_data: bytes) -> "Message":
        reconstructed = cls(
            Command.from_bytes(raw_data[0:1]),
            raw_data[cls.PAYLOAD_OFFSET : -cls.NUM_BYTES_CHECKSUM],
        )
        return reconstructed

    @property
    def payload_as_uint(self) -> int:
        return int.from_bytes(self.payload, byteorder=self.byte_order, signed=False)

    @classmethod
    def get_size_from_header(cls, header: bytes) -> int:
        reconstructed = cls.from_bytes(header)
        return reconstructed.payload_size

    @property
    def _message_without_checksum(self) -> bytes:
        return b"".join(
            [self._command_in_bytes, self._payload_size_in_bytes, self.payload]
        )

    def _int_to_bytes(self, data: int, length: int) -> bytes:
        return data.to_bytes(length, byteorder=self.byte_order, signed=False)

    @property
    def _payload_size_in_bytes(self) -> bytes:
        return self._int_to_bytes(self.payload_size, self.NUM_BYTES_PAYLOAD_SIZE)

    @property
    def _command_in_bytes(self) -> bytes:
        return self._int_to_bytes(self.command, self.NUM_BYTES_COMMAND)

    @property
    def payload_size(self) -> int:
        return len(self.payload)

    @property
    def checksum(self) -> bytes:
        data = self._message_without_checksum
        checksum = 0
        for b in data:
            checksum ^= b
        return self._int_to_bytes(checksum, self.NUM_BYTES_CHECKSUM)

    def __repr__(self):
        command_name = (
            self.command.name
            if hasattr(self.command, "name")
            else f"custom <{self.command}>"
        )
        return (
            f"Message(command={command_name}, payload={'...'.join([str(self.payload[:10]), str(self.payload[-10:])]) if self.payload_size > 20 else self.payload}, checksum={self.checksum},"
            f" payload_size={self.payload_size})"
        )

    def to_bytes(self) -> bytes:
        return b"".join([self._message_without_checksum, self.checksum])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        else:
            return (
                self.payload == other.payload
                and self.command == other.command
                and self.payload == other.payload
            )
