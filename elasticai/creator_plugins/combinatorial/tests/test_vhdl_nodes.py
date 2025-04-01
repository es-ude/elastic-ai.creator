import pytest

from elasticai.creator.ir2vhdl import Shape, VhdlNode

from ..vhdl_nodes.node_factory import (
    InstanceFactoryForCombinatorial,
)


@pytest.fixture(scope="class")
def node(raw_node):
    return InstanceFactoryForCombinatorial(raw_node)


def new_node(
    name: str,
    type: str,
    implementation: str,
    input_shape: Shape,
    output_shape: Shape,
    attributes: dict | None = None,
) -> VhdlNode:
    if attributes is None:
        attributes = {}
    n = VhdlNode(
        dict(name=name, type=type, implementation=implementation, **attributes)
    )
    n.input_shape = input_shape
    n.output_shape = output_shape

    return n


class TestStridingShiftRegister:
    @pytest.fixture(scope="class")
    def raw_node(self):
        conv0_channels = 2
        conv1_kernel_size = 3
        conv0_out_shape = Shape(conv0_channels, 1)
        conv1_in_shape = Shape(conv0_channels, conv1_kernel_size)
        n = new_node(
            name="a",
            type="striding_shift_register",
            implementation="striding_shift_register",
            input_shape=conv0_out_shape,
            output_shape=conv1_in_shape,
        )
        n.data["stride"] = 2
        return n

    def test_can_instantiate(self, node, raw_node):
        """
        data width is the size of a single time step, which
        translates to number of channels or depth of the preceding convolution

        num points is the number of time steps for the succeeding which corresponds to
        the kernel size of the succeeding convolution.
        """
        stride = raw_node.attributes["stride"]
        expected = (
            "a: entity work.striding_shift_register(rtl)",
            "generic map (",
            f"DATA_WIDTH => {raw_node.input_shape.depth},",
            f"NUM_POINTS => {raw_node.output_shape.width},",
            f"STRIDE => {stride}",
            ")",
            "port map (",
            "clk => clk,",
            "rst => rst,",
            "d_in => d_in_a,",
            "d_out => d_out_a,",
            "valid_in => valid_in_a,",
            "valid_out => valid_out_a",
            ");",
        )
        actual = tuple(line.strip() for line in node.instantiate())
        assert actual == expected

    def test_can_define_signals(self, node, raw_node):
        signals = tuple(line for line in node.define_signals())
        expected = (
            f"signal d_in_a : std_logic_vector({raw_node.input_shape.size()} - 1 downto 0) := (others => '0');",
            f"signal d_out_a : std_logic_vector({raw_node.output_shape.size()} - 1 downto 0) := (others => '0');",
            "signal valid_in_a : std_logic := '0';",
            "signal valid_out_a : std_logic := '0';",
        )
        assert signals == expected


class TestShiftRegister:
    @pytest.fixture(scope="class")
    def raw_node(self):
        conv0_channels = 2
        conv1_kernel_size = 3
        conv0_out_shape = Shape(conv0_channels, 1)
        conv1_in_shape = Shape(conv0_channels, conv1_kernel_size)
        n = new_node(
            name="a",
            type="shift_register",
            implementation="impl",
            input_shape=conv0_out_shape,
            output_shape=conv1_in_shape,
        )
        return n

    def test_can_instantiate(self, node, raw_node):
        expected = (
            "a: entity work.impl(rtl)",
            "generic map (",
            f"DATA_WIDTH => {raw_node.input_shape.depth},",
            f"NUM_POINTS => {raw_node.output_shape.width}",
            ")",
            "port map (",
            "clk => clk,",
            "rst => rst,",
            "d_in => d_in_a,",
            "d_out => d_out_a,",
            "valid_in => valid_in_a,",
            "valid_out => valid_out_a",
            ");",
        )
        actual = tuple(line.strip() for line in node.instantiate())
        assert actual == expected

    def test_can_define_signals(self, node):
        expected = (
            "signal d_in_a : std_logic_vector(2 - 1 downto 0) := (others => '0');",
            "signal d_out_a : std_logic_vector(6 - 1 downto 0) := (others => '0');",
            "signal valid_in_a : std_logic := '0';",
            "signal valid_out_a : std_logic := '0';",
        )
        signals = tuple(line for line in node.define_signals())

        assert signals == expected


class TestSlidingWindow:
    @pytest.fixture(scope="class")
    def raw_node(self):
        return new_node(
            name="a",
            type="sliding_window",
            input_shape=Shape(4),
            output_shape=Shape(2),
            implementation="impl",
            attributes={"stride": 2},
        )

    def test_can_instantiate(self, node):
        expected = (
            "a: entity work.impl(rtl)",
            "generic map (",
            "STRIDE => 2,",
            "INPUT_WIDTH => 4,",
            "OUTPUT_WIDTH => 2",
            ")",
            "port map (",
            "clk => clk,",
            "rst => rst,",
            "d_in => d_in_a,",
            "d_out => d_out_a,",
            "valid_in => valid_in_a,",
            "valid_out => valid_out_a",
            ");",
        )
        actual = tuple(line.strip() for line in node.instantiate())
        assert actual == expected

    def test_can_define_signals(self, node):
        expected = (
            "signal d_in_a : std_logic_vector(4 - 1 downto 0) := (others => '0');",
            "signal d_out_a : std_logic_vector(2 - 1 downto 0) := (others => '0');",
            "signal valid_in_a : std_logic := '0';",
            "signal valid_out_a : std_logic := '0';",
        )
        signals = tuple(line for line in node.define_signals())

        assert signals == expected
