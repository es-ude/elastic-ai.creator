from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from lut.singular_lut.utils import vhd_lut_cases

from elasticai.creator.vhdl.design.design import Design


@dataclass(eq=True, frozen=True)
class Signal:
    name: str
    width: int


@dataclass(eq=True, frozen=True)
class Port:
    incoming: tuple[Signal, ...]
    outgoing: tuple[Signal, ...]


class LUT(Design):
    def __init__(
        self,
        name: str,
        in_bits: int,
        out_bits: int,
        io_pairs,
    ):
        super().__init__(name)
        self._io_pairs = io_pairs
        self._in_bits = in_bits
        self._out_bits = out_bits
        self._port = Port(
            incoming=(Signal(name="x", width=in_bits), Signal(name="enable", width=0)),
            outgoing=(Signal(name="y", width=out_bits), Signal(name="clock", width=0)),
        )

    @property
    def port(self) -> Port:
        return self._port

    def save_to(self, destination: Path):
        template = SimpleNamespace(
            content=list(
                """library ieee;
use ieee.std_logic_1164.all;

entity $name is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x   : in std_logic_vector($x_width-1 downto 0);
        y  : out std_logic_vector($y_width-1 downto 0);
    );
end $name;

architecture rtl of $name is
begin
    process (x)
    begin
        case x is
            $cases
        end case;
    end process;
end rtl;""".splitlines()
            ),
            parameters={
                "cases": vhd_lut_cases(zip(*self._io_pairs)),
                "name": self.name,
                "y_width": str(self._out_bits),
                "x_width": str(self._in_bits),
            },
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)
