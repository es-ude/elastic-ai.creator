import unittest

from elasticai.creator.vhdl.ports.port_map_impl import PortMapImpl as PortMap
from elasticai.creator.vhdl.signals import SignalBuilder


class PortMapTestCase(unittest.TestCase):
    def setUp(self) -> None:
        b = SignalBuilder()
        self.x = b.id("x").accepted_names(["x", "y"]).build()
        self.y = b.id("y").build()
        return super().setUp()

    def single_in_signal_portmap(self) -> PortMap:
        return PortMap(id="in_map", in_signals=(self.x,), out_signals=tuple())

    def test_code_generation(self):
        map = PortMap(id="map", in_signals=[self.x], out_signals=[self.y])
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

        map = PortMap(id="my_id", in_signals=[logic, vector], out_signals=[])
        expected = sorted(
            [
                "signal my_id_l : std_logic;",
                "signal my_id_v : std_logic_vector(0 downto 0);",
            ]
        )
        self.assertEqual(expected, sorted(list(map.signal_definitions())))


if __name__ == "__main__":
    unittest.main()
