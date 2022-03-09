from itertools import chain
from typing import Iterable

from elasticai.creator.vhdl.language import (
    ContextClause,
    LibraryClause,
    UseClause,
    Entity,
    InterfaceVariable,
    DataType,
    Mode,
    Architecture,
    InterfaceSignal
)


class MacAsync:
    def __init__(self, component_name, data_width, frac_width):
        self.component_name = component_name
        self.architecture_name = "{component_name}_rtl".format(component_name=self.component_name)
        self.data_width = data_width
        self.frac_width = frac_width

    def __call__(self) -> Iterable[str]:
        library = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.numeric_std.all",
                ]
            ),
        )
        entity = Entity(self.component_name)
        entity.generic_list.append(
            InterfaceVariable(
                identifier="DATA_WIDTH", identifier_type=DataType.INTEGER, value=self.data_width
            )
        )
        entity.generic_list.append(
            InterfaceVariable(
                identifier="FRAC_WIDTH", identifier_type=DataType.INTEGER, value=self.frac_width
            )
        )
        entity.port_list.append(
            InterfaceVariable(
                identifier="x1",
                mode=Mode.IN,
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        entity.port_list.append(
            InterfaceVariable(
                identifier="x2",
                mode=Mode.IN,
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        entity.port_list.append(
            InterfaceVariable(
                identifier="w1",
                mode=Mode.IN,
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        entity.port_list.append(
            InterfaceVariable(
                identifier="w2",
                mode=Mode.IN,
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        entity.port_list.append(
            InterfaceVariable(
                identifier="b",
                mode=Mode.IN,
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        entity.port_list.append(
            InterfaceVariable(
                identifier="y",
                mode=Mode.OUT,
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )

        architecture = Architecture(
            identifier=self.architecture_name,
            design_unit=self.component_name,
        )
        architecture.architecture_declaration_list.append(
            InterfaceSignal(
                identifier="product_1",
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        architecture.architecture_declaration_list.append(
            InterfaceSignal(
                identifier="product_2",
                range="DATA_WIDTH-1 downto 0",
                identifier_type=DataType.SIGNED,
            )
        )
        architecture.architecture_statement_part.append(
            [
                "product_1 <= shift_right((x1 * w1), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
                "product_2 <= shift_right((x2 * w2), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
                "y <= product_1 + product_2 + b;"
            ]
        )

        code = chain(chain(library(), entity()), architecture())
        return code
