from typing import Any

from pytest import fixture

from elasticai.creator.graph.base_graph import BaseGraph
from elasticai.creator.ir2vhdl import (
    Edge,
    Implementation,
    Instance,
    LogicSignal,
    LogicVectorSignal,
    NullDefinedLogicSignal,
    PortMap,
    Shape,
    Signal,
    VhdlNode,
    vhdl_node,
)


@fixture
def impl() -> Implementation:
    impl: Implementation[VhdlNode, Edge] = Implementation(
        name="conv1",
        type="conv",
        data={"a": 1},
    )
    impl.add_node(
        VhdlNode(name="x", data=dict(type="y")),
    )
    return impl


@fixture
def data() -> dict[str, Any]:
    return {
        "name": "conv1",
        "type": "conv",
        "nodes": {
            "x": {"type": "y"},
        },
        "edges": {},
        "a": 1,
    }


def test_store_as_dict(data, impl):
    assert data == impl.as_dict()


def test_load_from_dict(data):
    assert data == Implementation(graph=BaseGraph()).load_from_dict(data).as_dict()


def test_can_access_attributes_of_vhdl_node():
    n = VhdlNode(name="a", data={"type": "b", "implementation": "c", "stride": 2})
    n.input_shape = Shape(1)
    n.output_shape = Shape(2)
    assert n.attributes["stride"] == 2


class TestInstance:
    def test_instantiate(self):
        n = vhdl_node(
            name="my_component",
            type="my_type",
            implementation="my_implementation",
            input_shape=Shape(2, 8),
            output_shape=Shape(2),
            attributes={},
        )
        my_instance = Instance(
            n,
            dict(data_depth="8"),
            dict(
                clk=NullDefinedLogicSignal("clk"),
                valid_in=LogicSignal("valid_in"),
                d_in=LogicVectorSignal("data_in", 16),
            ),
        )
        expected = (
            "my_component: entity work.my_implementation(rtl) ",
            "generic map (",
            "  DATA_DEPTH => 8",
            "  )",
            "  port map (",
            "    clk => clk,",
            "    valid_in => valid_in_my_component,",
            "    d_in => data_in_my_component",
            "  );",
        )

        assert expected == tuple(my_instance.instantiate())

    def test_can_instantiate_with_two_generics(self):
        n = vhdl_node(
            name="my_component",
            type="my_type",
            implementation="my_implementation",
            input_shape=Shape(2, 8),
            output_shape=Shape(2),
            attributes={},
        )
        my_instance = Instance(
            n,
            dict(data_depth="8", other="10"),
            dict(),
        )
        expected = (
            "my_component: entity work.my_implementation(rtl) ",
            "generic map (",
            "  DATA_DEPTH => 8,",
            "  OTHER => 10",
            "  )",
            "  port map (",
            "  );",
        )

        assert expected == tuple(my_instance.instantiate())


class TestLogicSignal:
    def test_can_create_from_code_with_default_val(self):
        code = "signal clk: std_logic := '0';"
        signal = Signal.from_code(code)
        assert signal.name == "clk"

    def test_can_create_from_code(self):
        code = "signal clk : std_logic;"
        signal = Signal.from_code(code)
        assert signal.name == "clk"

    def test_can_define_signal(self):
        code = ("signal clk : std_logic := '0';",)
        signal = LogicSignal("clk")
        assert code == tuple(signal.define())

    def test_make_instance_specific(self):
        code = ("signal clk_my_component : std_logic := '0';",)
        signal = LogicSignal("clk")
        signal = signal.make_instance_specific("my_component")
        assert code == tuple(signal.define())


class TestLogicVectorSignal:
    def test_can_create_from_code_with_default_val(self):
        code = 'signal d_in: std_logic_vector(3 downto 0) := "0000";'
        signal = Signal.from_code(code)
        assert signal.name == "d_in"
        assert signal.width == 4

    def test_can_create_from_signal_with_arith_expr(self):
        code = 'signal d_in: std_logic_vector(3 - 1 downto 0) := "000";'
        signal = Signal.from_code(code)
        assert signal.width == 3

    def test_can_define(self):
        code = ("signal d_in : std_logic_vector(4 - 1 downto 0) := (others => '0');",)
        signal = LogicVectorSignal("d_in", 4)
        assert code == tuple(signal.define())


class TestPortMap:
    def test_can_get_as_dict(self):
        p = PortMap(
            dict(
                d_in=LogicVectorSignal("d_in", 4),
                clk=LogicSignal("clk"),
            )
        )
        assert p.as_dict() == {
            "d_in": "signal d_in : std_logic_vector(4 - 1 downto 0) := (others => '0');",
            "clk": "signal clk : std_logic := '0';",
        }

    def test_can_get_from_dict(self):
        p = PortMap(
            dict(
                d_in=LogicVectorSignal("d_in", 4),
                clk=LogicSignal("clk"),
            )
        )
        assert p == PortMap.from_dict(
            {
                "d_in": "signal d_in : std_logic_vector(4 - 1 downto 0) := (others => '0');",
                "clk": "signal clk : std_logic := '0';",
            }
        )
