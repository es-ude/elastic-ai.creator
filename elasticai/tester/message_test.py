from .message import Message


def test_to_and_from_bytes_are_equal():
    raw = b"\x03\x00\x00\x00\x04\x00\x00\x02\x00\x05"
    msg = Message.from_bytes(raw)
    actual = msg.to_bytes()
    assert actual == raw


def test_get_flash_chunk_size_response_checksum_is_0():
    response_from_env5 = b"\x03\x00\x00\x00\x04\x00\x00\x02\x00\x05"
    msg = Message.from_bytes(response_from_env5)
    assert msg.checksum == b"\x05"
