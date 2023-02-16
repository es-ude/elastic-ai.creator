import unittest

from elasticai.creator.vhdl.language.ports import PortImpl as Port
from elasticai.creator.vhdl.language.ports import PortMap
from elasticai.creator.vhdl.language.signals import SignalBuilder


class PortMapTestCase(unittest.TestCase):
    def setUp(self) -> None:
        b = SignalBuilder()
        self.x = b.id("x").accepted_names(["x", "y"]).build()
        self.y = b.id("y").build()
        return super().setUp()

    def single_in_signal_portmap(self) -> PortMap:
        return Port(in_signals=(self.x,), out_signals=tuple()).build_port_map(
            id="in_map"
        )

    def test_code_generation(self):
        map = Port(in_signals=[self.x], out_signals=[self.y]).build_port_map(id="map")
        self.assertEqual(
            [
                "map : entity work.map(rtl)",
                "port map (",
                "x => map_x,",
                "y => map_y",
                ");",
            ],
            list(map.instantiation()),
        )

    def test_signal_definitions(self):
        logic = SignalBuilder().id("l").build()
        vector = SignalBuilder().id("v").width(1).build()

        map = Port(in_signals=[logic, vector], out_signals=[]).build_port_map(
            id="my_id"
        )
        expected = sorted(
            [
                "signal my_id_l : std_logic;",
                "signal my_id_v : std_logic_vector(0 downto 0);",
            ]
        )
        self.assertEqual(expected, sorted(list(map.signal_definitions())))
