import unittest

from elasticai.creator.vhdl.components.network_component import ComponentInstantiation


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
     - how do i know what the adress width of a layer is?
    """


class ComponentInstantiationTest(unittest.TestCase):
    def test_creates_instance(self):
        instantiation = ComponentInstantiation("my_comp")

        first_line = list(instantiation.code())[0]
        expected = "my_comp : entity work.my_comp(rtl)"
        self.assertEqual(expected, first_line)

    def test_opens_portmap(self):
        instantiation = ComponentInstantiation("my_comp")
        second_line = list(instantiation.code())[1]
        expected = "port map("
        self.assertEqual(expected, second_line)

    def test_assigns_all_ports(self):
        instantiation = ComponentInstantiation("my_comp")
        for port in ("enable", "clock", "x", "y"):
            with self.subTest(port):
                self.assertTrue(
                    any(
                        (
                            line.startswith(f"{port} => ")
                            for line in instantiation.code()
                        )
                    )
                )

    def test_ports_are_connected_to_correctly_named_signals(self):
        instantiation = ComponentInstantiation("some_other_name")
        port_to_signal_connections = list(instantiation.code())[2:-1]
        for line in port_to_signal_connections:
            with self.subTest(line):
                line = line.strip(",")
                port = line.split(" => ")[0]
                signal = line.split(" => ")[1]
                self.assertEqual(signal, f"some_other_name_{port}")
