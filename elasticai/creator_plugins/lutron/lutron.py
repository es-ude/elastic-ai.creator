from itertools import chain
from typing import TypeAlias

from elasticai.creator.ir2vhdl import Code, Implementation, type_handler

from .language import Port, VHDLEntity

IOPair: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]


@type_handler
def lutron(lowered: Implementation) -> Code:
    def _iter():
        name: str = lowered.name
        d_in_width: int = lowered.attributes["input_size"]
        d_out_width: int = lowered.attributes["output_size"]
        io_pairs: tuple[IOPair, ...] = lowered.attributes["truth_table"]
        entity = VHDLEntity(
            name=name,
            port=Port(
                inputs=dict(d_in=logic_vector(d_in_width)),
                outputs=dict(d_out=logic_vector(d_out_width)),
            ),
            generics=dict(),
        )

        def rtl():
            yield from [
                f"architecture rtl of {name} is",
                "begin",
                "  process (d_in) is",
                "  begin",
                "    case d_in is",
            ]
            for _in, out in io_pairs:
                yield f'      when b"{_in}" => d_out <= b"{out}";'
            yield "      when others => d_out <= (others => 'X');"
            yield from ("    end case;", "  end process;", "end architecture;")

        yield from chain(entity.generate_entity(), ("",), rtl())

    return lowered.name, _iter()


def logic_vector(width: int | str) -> str:
    return f"std_logic_vector({width} - 1 downto 0)"
