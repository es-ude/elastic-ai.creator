from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    AssignmentList,
    Signal,
    SignalDefinitionList,
)


def test_signal_definition_list_with_default():
    expected = {Signal("a", 0)}
    actual = SignalDefinitionList.from_code("signal a: std_logic := '0';").signals
    assert actual == expected


def test_vector_signal():
    expected = {Signal("v", 5)}
    actual = SignalDefinitionList.from_code(
        "signal v: std_logic_vector(5 - 1 downto 0) := (others => '0');"
    ).signals
    assert actual == expected


def test_alternative_vector_signal():
    expected = {Signal("v", 5)}
    actual = SignalDefinitionList.from_code(
        "signal v: std_logic_vector(4 downto 0) := (others => '0');"
    ).signals
    assert actual == expected


def test_two_vector_signals():
    expected = {Signal("v", 5), Signal("a", 4)}
    actual = SignalDefinitionList.from_code(
        [
            "signal v: std_logic_vector(4 downto 0) := (others => '0');",
            "signal a: std_logic_vector(3 downto 0) := (others => '0');",
        ]
    ).signals
    assert actual == expected


def test_layer_connections():
    expected = AssignmentList.from_dict({"a": "b", "c": "d"})
    actual = AssignmentList.from_code("some other stuf...; a <= b; c <= d;")
    assert actual == expected
