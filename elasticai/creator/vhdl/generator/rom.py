from io import StringIO
from itertools import chain

from elasticai.creator.vhdl.generator.functions.generate_lstm_cell_vhd import component_name
from elasticai.creator.vhdl.language import ContextClause, LibraryClause, UseClause, Entity, InterfaceVariable, \
    DataType, Mode, Architecture, Process


class Rom:
    def __init__(self, rom_name, data_width, addr_width, array_value):
        self.rom_name = rom_name
        self.architecture_name = f'{rom_name}_rtl'.format(rom_name=rom_name)
        self.rom_name_arrat_t = f'{rom_name}_array_t'.format(rom_name=rom_name)
        self.data_width = data_width
        self.addr_width = addr_width
        self.array_value = array_value

    def __call__(self):
        library = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.std_logic_unsigned.all",
                ]
            ),
        )
        entity = Entity(identifier=self.rom_name)
        entity.port_list.append(
            InterfaceVariable(identifier="clk",
                              identifier_type=DataType.STD_LOGIC,
                              mode=Mode.IN)
        )
        entity.port_list.append(
            InterfaceVariable(identifier="en",
                              identifier_type=DataType.STD_LOGIC,
                              mode=Mode.IN)
        )
        entity.port_list.append(
            InterfaceVariable(identifier="addr",
                              identifier_type=DataType.STD_LOGIC_VECTOR,
                              mode=Mode.IN,
                              range="{addr_width}-1 downto 0".format(addr_width=self.addr_width))
        )
        entity.port_list.append(
            InterfaceVariable(identifier="data",
                              identifier_type=DataType.STD_LOGIC_VECTOR,
                              mode=Mode.OUT,
                              range="{data_width}-1 downto 0".format(data_width=self.data_width))
        )
        architecture = Architecture(
            identifier=self.architecture_name,
            design_unit=self.rom_name)

        architecture.architecture_declaration_list.append(
            "type rom_bi_array_t is array (0 to 2**{addr_width}-1) of std_logic_vector({data_width}-1 downto 0)".format(
                addr_width=self.addr_width,
                data_width=self.data_width)
        )
        architecture.architecture_declaration_list.append(
            "signal ROM : {rom_name_arrat_t}:=({array_value})".format(rom_name_arrat_t=self.rom_name_arrat_t,
                                                                      array_value=self.array_value)
        )
        architecture.architecture_declaration_list.append(
            "attribute rom_style : string"
        )
        architecture.architecture_declaration_list.append(
            'attribute rom_style of ROM : signal is "block"'
        )

        rom_process = Process(identifier="ROM", input_name="clk")
        rom_process.process_statements_list.append(
            "if rising_edge(clk) then\nif (en = '1') then\ndata <= ROM(conv_integer(addr))"
        )
        rom_process.process_statements_list.append("end if")
        rom_process.process_statements_list.append("end if")

        architecture.architecture_statement_part = rom_process

        code = chain(chain(library(), entity()), architecture())

        return code
