import unittest

from elasticai.creator.vhdl.ports.port import PortMapImpl as PortMap
from elasticai.creator.vhdl.signals import SignalBuilder


class PortMapTestCase(unittest.TestCase):
    def setUp(self) -> None:
        b = SignalBuilder()
        self.x = b.id("x").accepted_names(["x", "y"]).build()
        self.y = b.id("y").build()
        return super().setUp()

    def single_in_signal_portmap(self) -> PortMap:
        return PortMap(id="in_map", in_signals=(self.x,), out_signals=tuple())

    def test_connecting_two_single_signal_portmaps_connects_their_signals(self):
        in_map = self.single_in_signal_portmap()
        out_map = PortMap(
            id="out_map",
            in_signals=tuple(),
            out_signals=(self.x,),
        )
        in_map.connect(out_map)
        self.assertFalse(out_map.is_missing_inputs())

    def test_connects_in_both_directions(self):
        out_map = PortMap(id="out", in_signals=tuple(), out_signals=(self.x,))
        in_map = PortMap(id="in", in_signals=(self.x,), out_signals=tuple())
        out_map.connect(in_map)
        self.assertFalse(out_map.is_missing_inputs())

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
            list(map.code()),
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

    def test_code_generation_for_connected_signals_returns_map_to_other_map_connection(
        self,
    ):
        map = PortMap(id="map", in_signals=[self.x], out_signals=[])
        other_map = PortMap(id="other", in_signals=[], out_signals=[self.x])
        map.connect(other_map)
        self.assertEqual(["other_x <= map_x;"], list(map.connections()))

    def test_code_for_unconnected_signals_is_empty(self):
        map = PortMap(id="map", in_signals=[self.x], out_signals=[])
        self.assertEqual([], list(map.connections()))


if __name__ == "__main__":
    unittest.main()
