import pytest

from .commands import Command
from .io_stream import IOStream
from .message import Message
from .message_builder import MessageBuilder
from .message_io import MessageIO


@pytest.fixture
def get_flash_chunk_size_msg():
    return b"\x03\x00\x00\x00\x04\x00\x00\x02\x00\x05"


class DummyIO(IOStream):
    def __init__(self):
        self.tx = bytearray()
        self.current_read_pos = 0
        self.rx = bytearray()

    def read(self, n_bytes: int) -> bytes:
        old_pos = self.current_read_pos
        self.current_read_pos += n_bytes
        v = self.tx[old_pos : self.current_read_pos]
        return v

    def write(self, data: bytes):
        self.rx.extend(data)


@pytest.fixture
def dummy():
    return DummyIO()


@pytest.fixture
def msg_builder():
    return MessageBuilder()


@pytest.fixture
def ack(msg_builder):
    b = msg_builder
    b.command = Command.ACK
    return next(iter(b.build()))


@pytest.fixture
def nak(msg_builder):
    b = msg_builder
    b.command = Command.NAK
    return next(iter(b.build()))


@pytest.fixture
def cut(dummy, ack, nak):
    return MessageIO(io_stream=dummy, ack=ack, nak=nak, byte_order="big", max_trials=2)


def test_GET_FLASH_CHUNK_SIZE_is_acknowledged(
    get_flash_chunk_size_msg: bytes, cut: MessageIO, ack: Message, dummy: DummyIO
):
    dummy.tx.extend(get_flash_chunk_size_msg)
    dummy.tx.extend(ack.to_bytes())
    cut.read()
    assert dummy.rx == ack.to_bytes()


def test_ACK_is_not_acknowledged(ack: Message, cut: MessageIO, dummy: DummyIO):
    dummy.tx.extend(ack.to_bytes())
    cut.read()
    assert len(dummy.rx) == 0
