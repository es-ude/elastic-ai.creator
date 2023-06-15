from elasticai.creator.vhdl.code_generation.code_abstractions import to_vhdl_hex_string


def test_convert_1_to_1bit_hex():
    assert to_vhdl_hex_string(1, 1) == "'x1'"


def test_convert_1_to_5bit_hex():
    assert to_vhdl_hex_string(1, 5) == "'x01'"


def test_convert_11_to_4bit_hex():
    assert to_vhdl_hex_string(11, 4) == "'xb'"


def test_convert_254_to_9bit_hex():
    assert to_vhdl_hex_string(254, 9) == "'x0fe'"
