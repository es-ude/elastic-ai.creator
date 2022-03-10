from itertools import chain
from typing import Iterable

from elasticai.creator.vhdl.language import (
    ContextClause,
    LibraryClause,
    UseClause,
    Entity,
    Architecture,
    ComponentDeclaration,
    Process,
    PortMap,
)
from elasticai.creator.vhdl.language_testbench import TestCasesLSTM


class LSTMCommonGateTestBench:
    def __init__(
        self,
        data_width: int,
        frac_width: int,
        vector_len_width: int,
        x_mem_list_for_testing: list,
        w_mem_list_for_testing: list,
        b_list_for_testing: list,
        y_list_for_testing: list[int],
        component_name: str = None,
    ):
        self.component_name = self._get_lower_case_class_name_or_component_name(
            component_name=component_name
        )
        self.data_width = data_width
        self.frac_width = frac_width
        self.vector_len_width = vector_len_width
        self.x_mem_list_for_testing = x_mem_list_for_testing
        self.w_mem_list_for_testing = w_mem_list_for_testing
        self.y_list_for_testing = y_list_for_testing
        self.b_list_for_testing = b_list_for_testing

    @classmethod
    def _get_lower_case_class_name_or_component_name(cls, component_name):
        if component_name is None:
            return cls.__name__.lower()
        return component_name

    @property
    def file_name(self) -> str:
        return f"{self.component_name}_tb.vhd"

    def __call__(self) -> Iterable[str]:
        library = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.numeric_std.all",
                    "ieee.math_real.all",
                ]
            ),
        )

        entity = Entity(self.component_name + "_tb")
        entity.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
            f"VECTOR_LEN_WIDTH : integer := {self.vector_len_width}",
        ]
        entity.port_list = [
            "clk : out std_logic",
        ]

        component = ComponentDeclaration(identifier=self.component_name)
        component.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
            f"VECTOR_LEN_WIDTH : integer := {self.vector_len_width}",
        ]
        component.port_list = [
            "reset : in std_logic",
            "clk : in std_logic",
            "x : in signed(DATA_WIDTH-1 downto 0)",
            "w : in signed(DATA_WIDTH-1 downto 0)",
            "b : in signed(DATA_WIDTH-1 downto 0)",
            "vector_len : in unsigned(VECTOR_LEN_WIDTH-1 downto 0)",
            "idx : out unsigned(VECTOR_LEN_WIDTH-1 downto 0)",
            "ready : out std_logic",
            "y : out signed(DATA_WIDTH-1 downto 0)",
        ]

        process = Process(
            identifier="clock",
        )
        process.process_statements_list = [
            "clk <= '0'",
            "wait for clk_period/2",
            "clk <= '1'",
            "wait for clk_period/2",
        ]

        uut_port_map = PortMap(map_name="uut", component_name=self.component_name)
        uut_port_map.signal_list.append("reset => reset")
        uut_port_map.signal_list.append("clk => clock")
        uut_port_map.signal_list.append("x => x")
        uut_port_map.signal_list.append("w => w")
        uut_port_map.signal_list.append("b => b")
        uut_port_map.signal_list.append("vector_len => vector_len")
        uut_port_map.signal_list.append("idx => idx")
        uut_port_map.signal_list.append("ready => ready")
        uut_port_map.signal_list.append("y => y")

        test_cases = TestCasesLSTM(
            x_mem_list_for_testing=self.x_mem_list_for_testing,
            w_mem_list_for_testing=self.w_mem_list_for_testing,
            b_list_for_testing=self.b_list_for_testing,
            y_list_for_testing=self.y_list_for_testing,
        )
        test_process = Process(identifier="test")
        test_process.process_test_case_list = test_cases

        architecture = Architecture(
            design_unit=self.component_name + "_tb",
        )

        architecture.architecture_declaration_list.append(
            "type RAM_ARRAY is array (0 to 9) of signed(DATA_WIDTH-1 downto 0)"
        )

        architecture.architecture_declaration_list.append(
            "signal clk_period : time := 2 ps"
        )
        architecture.architecture_declaration_list.append("signal clock : std_logic")
        architecture.architecture_declaration_list.append(
            "signal reset, ready : std_logic:='0'"
        )
        architecture.architecture_declaration_list.append(
            "signal X_MEM : RAM_ARRAY :=(others=>(others=>'0'))"
        )
        architecture.architecture_declaration_list.append(
            "signal W_MEM : RAM_ARRAY:=(others=>(others=>'0'))"
        )
        architecture.architecture_declaration_list.append(
            "signal x, w, y, b : signed(DATA_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal vector_len : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal idx : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_component_list.append(component)
        architecture.architecture_process_list.append(process)
        architecture.architecture_port_map_list.append(uut_port_map)
        architecture.architecture_assignment_at_end_of_declaration_list.append(
            "x <= X_MEM(to_integer(idx))"
        )
        architecture.architecture_assignment_at_end_of_declaration_list.append(
            "w <= W_MEM(to_integer(idx))"
        )
        architecture.architecture_statement_part = test_process

        code = chain(chain(library(), entity()), architecture())
        return code
