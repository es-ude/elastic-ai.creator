from itertools import chain
from typing import Iterable

from elasticai.creator.vhdl.generator.generator_functions import precomputed_logic_function_process
from elasticai.creator.vhdl.language import Architecture, Entity, InterfaceConstrained, Mode, DataType, UseClause, \
    LibraryClause, ContextClause
from elasticai.creator.vhdl.number_representations import BitVector


class NaiveLUTBasedConv():
    def __init__(self, component_name: str, inputs: Iterable[list[BitVector]], outputs: Iterable[list[BitVector]], input_width:int, output_width:int
                 ):
        self.identifier = component_name
        self.inputs = inputs
        self.outputs = outputs
        self.input_width = input_width
        self.output_width = output_width

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
        entity = Entity("NaiveLUTConv")
        entity.port_list.append(InterfaceConstrained(
            identifier="x",
            mode=Mode.IN,
            range=f"INPUT_WIDTH-1 downto 0",
            variable_type=DataType.STD_LOGIC_VECTOR,
        ))
        entity.port_list.append(InterfaceConstrained(
            identifier="y",
            mode=Mode.OUT,
            range=f"OUTPUT_WIDTH-1 downto 0",
            variable_type=DataType.STD_LOGIC_VECTOR,
        ))
        entity.generic_list = [
            f"INPUT_WIDTH : integer := {self.input_width}",
            f"OUTPUT_WIDTH : integer := {self.output_width}",
        ]
        architecture = Architecture(
            identifier=self.identifier,
            design_unit="NaiveLUTConv",
        )
        architecture.architecture_statement_part = lambda :precomputed_logic_function_process(
                x_list=self.inputs, y_list=self.outputs
            )
        code = chain(chain(library(), entity()), architecture())
        return code
