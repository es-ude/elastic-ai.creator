from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from creator.file_generation.template_writer import TemplateWriter
from creator.vhdl.code_generation.code_abstractions import to_vhdl_binary_string
from lut.singular_lut.utils import vhd_when

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
    def __init__(self, name: str, in_bits: int, out_bits: int, outputs: Iterable[int]):
        super().__init__(name)
        self._in_bits = in_bits
        self._out_bits = out_bits
        self.outputs = outputs
        self._port = Port(
            incoming=(Signal(name="x", width=in_bits), Signal(name="enable", width=0)),
            outgoing=(Signal(name="y", width=out_bits), Signal(name="clock", width=0)),
        )

    @property
    def port(self) -> Port:
        return self._port

    def _input_to_string(self, input: int):
        return to_vhdl_binary_string(input, self._in_bits)

    def _output_to_string(self, output: int):
        return to_vhdl_binary_string(output, self._out_bits)

    def _gen_cases(self):
        counter = 0
        for output in self.outputs:
            input = self._input_to_string(counter)
            counter += 1
            yield vhd_when((input, self._output_to_string(output)))

    def save_to(self, destination: Path):
        template = SimpleNamespace(
            content=list(
                """library ieee;
use ieee.std_logic_1164.all;

entity $name is
    port (
        enable : in std_logic;
        clock : in std_logic;
        x : in std_logic_vector($x_width-1 downto 0);
        y : out std_logic_vector($y_width-1 downto 0);
    );
end;

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
                "cases": self._gen_cases(),
                "name": self.name,
                "y_width": str(self._out_bits),
                "x_width": str(self._in_bits),
            },
        )
        with destination.create_subpath(f"{self.name}.vhd").open("w") as f:
            writer = TemplateWriter(f)
            writer.write(template)
