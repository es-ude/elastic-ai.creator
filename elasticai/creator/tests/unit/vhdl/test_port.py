import unittest

from elasticai.creator.vhdl.port import PortMap
from elasticai.creator.vhdl.signals import LogicInSignal, LogicOutSignal


class PortMapTestCase(unittest.TestCase):
    @staticmethod
    def single_in_signal_portmap() -> PortMap:
        return PortMap(
            id="in_map", in_signals=(LogicInSignal("x"),), out_signals=tuple()
        )

    def test_connecting_two_single_signal_portmaps_connects_their_signals(self):
        in_map = self.single_in_signal_portmap()
        out_map = PortMap(
            id="out_map",
            in_signals=tuple(),
            out_signals=(LogicOutSignal(basename="x"),),
        )
        in_map.connect(out_map)
        signal = in_map.in_signals()[0]
        self.assertFalse(signal.is_missing_inputs())

    def test_signals_are_prefixed_with_portmap_id(self):
        for id in ("my_map", "my_other_map"):
            with self.subTest(id):
                port_map = PortMap(
                    id=id,
                    in_signals=(LogicInSignal("x"),),
                    out_signals=(LogicOutSignal("y"),),
                )
                self.assertEqual(f"{id}_x", port_map.in_signals()[0].id())
                self.assertEqual(f"{id}_y", port_map.out_signals()[0].id())

    def test_connects_in_both_directions(self):
        out_map = PortMap(
            id="out", in_signals=tuple(), out_signals=(LogicOutSignal("x"),)
        )
        in_map = PortMap(id="in", in_signals=(LogicInSignal("x"),), out_signals=tuple())
        out_map.connect(in_map)
        self.assertFalse(in_map.in_signals()[0].is_missing_inputs())

    def test_code_generation(self):
        map = PortMap(
            id="map", in_signals=[LogicInSignal("x")], out_signals=[LogicOutSignal("y")]
        )
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

    def test_code_generation_for_connected_signals_returns_map_to_other_map_connection(
        self,
    ):
        map = PortMap(id="map", in_signals=[LogicInSignal("x")], out_signals=[])
        other_map = PortMap(
            id="other", in_signals=[], out_signals=[LogicOutSignal("x")]
        )
        map.connect(other_map)
        self.assertEqual(["map_x <= other_x;"], list(map.code_for_signal_connections()))

    def test_code_for_unconnected_signals_is_empty(self):
        map = PortMap(id="map", in_signals=[LogicInSignal("x")], out_signals=[])
        self.assertEqual([], list(map.code_for_signal_connections()))


class PortTestCase(unittest.TestCase):
    """
    Tests:
      - port can generate portmap
      - port signal connections are not used for portmap

    Open Questions:
      - where do default values come from?

    Thoughts:
      - port map dictates the signal prefixes. Port dictates signal types (in,out, vector, logic)
    """


if __name__ == "__main__":
    unittest.main()
