import unittest
from functools import partial

from elasticai.creator.vhdl.hw_equivalent_layers.hw_blocks import (
    BaseHWBlock,
    BufferedBaseHWBlock,
)


class NetworkHWComponentTest(unittest.TestCase):
    """
    Tests:
    - signals are uniquely named
    - signal names can be traced back to their emitting graph nodes and/or layers
    - all layers from model are instantiated
    - layer names correspond to signal and folder names
        -> node names need to be propagated to each module as well as network component
    - constants are correctly generated for
        - data width per layer connection
        - address width per layer connection

    Notes:
     - network needs to know in (x) and out (y) address widths for layers
     - network needs graph and component names for each graph node
     - network needs to know data width

    Questions for Chao:
        - Answer: address_width == number of bits to count number of inputs
            - max(1, math.ceil(math.log2(num_items)))
    """


class HWBlockInstantiationTest(unittest.TestCase):
    def test_creates_instance(self):
        block = BaseHWBlock(x_width=0, y_width=0)
        lines = block.instantiation("my_comp")
        first_line = list(lines)[0]
        expected = "my_comp : entity work.my_comp(rtl)"
        self.assertEqual(expected, first_line)

    def test_opens_portmap(self):
        block = BaseHWBlock(x_width=0, y_width=0)
        lines = block.instantiation("my_comp")
        second_line = list(lines)[1]
        expected = "port map("
        self.assertEqual(expected, second_line)

    def test_assigns_all_ports(self):
        block = BaseHWBlock(x_width=0, y_width=0)
        lines = partial(block.instantiation, "my_comp")
        for port in ("enable", "clock", "x", "y"):
            with self.subTest(port):
                self.assertTrue(
                    any((line.startswith(f"{port} => ") for line in lines())),
                    f"expected '{port} =>',\nfound: {list(lines())}",
                )

    def test_ports_are_connected_to_correctly_named_signals(self):
        block = BaseHWBlock(x_width=0, y_width=0)
        lines = block.instantiation("some_other_name")
        port_to_signal_connections = list(lines)[2:-1]
        for line in port_to_signal_connections:
            with self.subTest(line):
                line = line.strip(",")
                port = line.split(" => ")[0]
                signal = line.split(" => ")[1]
                self.assertEqual(signal, f"some_other_name_{port}")


class HWBlockSignals(unittest.TestCase):
    def constructor(self, name, data_width, *args):
        return BaseHWBlock(x_width=data_width, y_width=data_width).signal_definitions(
            name
        )

    def test_clock_signal_is_generated(self):
        signals = self.constructor("other_name", 2)
        self.assertTrue("signal other_name_clock : std_logic := '0';" in list(signals))

    def test_component_name_is_generated(self):
        signals = self.constructor("other_name", 2)
        self.assertTrue("signal other_name_enable : std_logic := '0';" in list(signals))

    def check_signals_contain_logic_vector(self, prefix, signals, suffix, width):
        expected = f"signal {prefix}_{suffix} : std_logic_vector({width - 1} downto 0);"
        signals = list(signals)
        self.assertTrue(
            expected in signals,
            msg=f"expected: {expected}\nfound: {signals}",
        )

    def test_generates_correct_number_of_lines(self):
        signals = self.constructor("other_name", 2)
        self.assertEqual(4, len(tuple(signals)))


class HWBlockSignals(HWBlockSignals):
    def constructor(self, name, data_width, *args):
        x_address_width = 1
        y_address_width = 1
        y_data_width = data_width
        x_data_width = data_width
        if len(args) == 2:
            x_address_width = args[0]
            y_address_width = args[1]
        return BufferedBaseHWBlock(
            x_width=x_data_width,
            y_width=y_data_width,
            x_address_width=x_address_width,
            y_address_width=y_address_width,
        ).signal_definitions(name)

    def test_5_downto_0_generated_for_x_addr_width_6(self):
        signals = self.constructor("other_name", 4, 6, 2)
        self.check_signals_contain_logic_vector("other_name", signals, "x_address", 6)

    def test_8_downto_0_generated_for_y_addr_width_9(self):
        signals = self.constructor("other_name", 4, 6, 9)
        self.check_signals_contain_logic_vector("other_name", signals, "y_address", 9)

    def test_generates_correct_number_of_lines(self):
        signals = self.constructor("other_name", 2, 2, 4)
        self.assertEqual(7, len(tuple(signals)))
