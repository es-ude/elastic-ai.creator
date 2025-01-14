from elasticai.tester.commands import Command
from elasticai.tester.message import Message
from elasticai.tester.message_builder import MessageBuilder


def test_mcu_led2_creates_correct_message():
    b = MessageBuilder()
    b.command = Command.MCU_LEDS
    b.payload = b"\x02"
    expected = Message(Command.MCU_LEDS, b"\x02", byte_order="big")
    actual = next(iter(b.build()))
    assert actual == expected


def test_mcu_led2_yields_correct_byte_sequence():
    b = MessageBuilder()
    b.command = Command.MCU_LEDS
    b.payload = b"\x02"
    expected = b"\x08\x00\x00\x00\x01\x02\x0b"
    actual = next(iter(b.build())).to_bytes()
    assert actual == expected
