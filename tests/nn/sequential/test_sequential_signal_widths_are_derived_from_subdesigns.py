import pytest

from elasticai.creator.nn.sequential.design import Sequential
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.design_base.ports import Port
from elasticai.creator.vhdl.savable import Path


class DummyDesign(Design):
    def __init__(
        self, name: str, x_width: int, y_width: int, x_count: int = 0, y_count: int = 0
    ):
        super().__init__(name)
        self._port = create_port(
            x_width=x_width, y_width=y_width, x_count=x_count, y_count=y_count
        )

    @property
    def port(self) -> Port:
        return self._port

    def save_to(self, destination: Path) -> None:
        pass


@pytest.fixture
def port_of_mixed_sequential():
    return Sequential(
        [
            DummyDesign("dd_0", x_width=4, y_width=3),
            DummyDesign("dd_1", x_width=3, x_count=2, y_width=6, y_count=3),
            DummyDesign("dd_2", x_width=6, y_width=2),
        ],
        name="sequential_0",
    ).port


def test_x_width_matches(port_of_mixed_sequential: Port):
    assert port_of_mixed_sequential["x"].width == 4


def test_y_width_matches(port_of_mixed_sequential: Port):
    assert port_of_mixed_sequential["y"].width == 2