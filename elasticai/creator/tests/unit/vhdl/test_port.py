import unittest

from elasticai.creator.vhdl.hardware_description_language.ports import Port as Port
from elasticai.creator.vhdl.hardware_description_language.ports import PortMap
from elasticai.creator.vhdl.hardware_description_language.signals import Signal


class PortMapTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x = Signal(id="x", accepted_names=["x", "y"], width=0)
        self.y = Signal(id="y", accepted_names=[], width=0)
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
        logic = Signal(id="l", accepted_names=[], width=0)
        vector = Signal(id="v", accepted_names=[], width=1)

        map = Port(in_signals=[logic, vector], out_signals=[]).build_port_map(
            id="my_id"
        )
        expected = sorted(
            [
                "signal my_id_l : std_logic := '0';",
                "signal my_id_v : std_logic_vector(0 downto 0) := (other => '0');",
            ]
        )
        self.assertEqual(expected, sorted(list(map.signal_definitions())))
