from elasticai.creator.hdl.code_generation.code_generation import to_hex


def test_convert_1_to_1bit_hex():
    assert to_hex(1, 1) == "1"


def test_convert_1_to_5bit_hex():
    assert to_hex(1, 5) == "01"


def test_convert_11_to_4bit_hex():
    assert to_hex(11, 4) == "b"


def test_convert_254_to_9bit_hex():
    assert to_hex(254, 9) == "0fe"
