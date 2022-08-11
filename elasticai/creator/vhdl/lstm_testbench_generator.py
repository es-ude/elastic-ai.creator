import math
from itertools import chain

from elasticai.creator.vhdl.language import (
    Architecture,
    Code,
    ContextClause,
    Entity,
    Keywords,
    LibraryClause,
    PortMap,
    Procedure,
    Process,
    UseClause,
    hex_representation,
)
from elasticai.creator.vhdl.language_testbench import TestBenchBase
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    infer_total_and_frac_bits,
)


class LSTMCellTestBench:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        test_x_h_data: list[FixedPoint],
        test_c_data: list[FixedPoint],
        h_out: list[FixedPoint],
        component_name: str = None,
    ):
        self.component_name = self._get_lower_case_class_name_or_component_name(
            component_name=component_name
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_h_addr_width = math.ceil(math.log2(input_size + hidden_size))
        self.hidden_addr_width = max(1, math.ceil(math.log2(hidden_size)))
        self.w_addr_width = math.ceil(
            math.log2((input_size + hidden_size) * hidden_size)
        )
        self.data_width, self.frac_width = infer_total_and_frac_bits(
            test_x_h_data, test_c_data, h_out
        )

        def to_hex(value: FixedPoint) -> str:
            return hex_representation(value.to_hex())

        self.test_x_h_data = list(map(to_hex, test_x_h_data))
        self.test_c_data = list(map(to_hex, test_c_data))
        self.h_out = h_out

    @classmethod
    def _get_lower_case_class_name_or_component_name(cls, component_name):
        if component_name is None:
            return cls.__name__.lower()
        return component_name

    @property
    def file_name(self) -> str:
        return f"{self.component_name}_tb.vhd"

    def __call__(self) -> Code:
        library = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee", "work"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.numeric_std.all",
                    "work.lstm_common.all",
                ]
            ),
        )

        entity = Entity(self.component_name + "_tb")
        entity.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
            f"INPUT_SIZE : integer := {self.input_size}",
            f"HIDDEN_SIZE : integer := {self.hidden_size}",
            f"X_H_ADDR_WIDTH : integer := {self.x_h_addr_width}",
            f"HIDDEN_ADDR_WIDTH : integer := {self.hidden_addr_width}",
            f"W_ADDR_WIDTH : integer := {self.w_addr_width}",
        ]
        entity.port_list = [
            "clk : out std_logic",
        ]

        procedure_0 = Procedure(identifier="send_x_h_data")
        procedure_0.declaration_list = [
            "addr_in : in std_logic_vector(X_H_ADDR_WIDTH-1 downto 0)",
            "data_in : in std_logic_vector(DATA_WIDTH-1 downto 0)",
            "signal clock : in std_logic",
            "signal wr : out std_logic",
            "signal addr_out : out std_logic_vector(X_H_ADDR_WIDTH-1 downto 0)",
        ]
        procedure_0.declaration_list_with_is = [
            "signal data_out : out std_logic_vector(DATA_WIDTH-1 downto 0))",
        ]
        procedure_0.statement_list = [
            "addr_out <= addr_in",
            "data_out <= data_in",
            "wait until clock='0'",
            "wr <= '1'",
            "wait until clock='1'",
            "wait until clock='0'",
            "wr <= '0'",
            "wait until clock='1'",
        ]
        procedure_1 = Procedure(identifier="send_c_data")
        procedure_1.declaration_list = [
            "addr_in : in std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0)",
            "data_in : in std_logic_vector(DATA_WIDTH-1 downto 0)",
            "signal clock : in std_logic",
            "signal wr : out std_logic",
            "signal addr_out : out std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0)",
        ]
        procedure_1.declaration_list_with_is = [
            "signal data_out : out std_logic_vector(DATA_WIDTH-1 downto 0))",
        ]
        procedure_1.statement_list = [
            "addr_out <= addr_in",
            "data_out <= data_in",
            "wait until clock='0'",
            "wr <= '1'",
            "wait until clock='1'",
            "wait until clock='0'",
            "wr <= '0'",
            "wait until clock='1'",
        ]

        process = Process(
            identifier="clock",
        )
        process.process_statements_list = [
            "clock <= '0'",
            "wait for clk_period/2",
            "clock <= '1'",
            "wait for clk_period/2",
        ]

        uut_port_map = PortMap(
            map_name="uut",
            component_name="entity work." + self.component_name + "(rtl)",
        )
        uut_port_map.generic_map_list = (
            "DATA_WIDTH => DATA_WIDTH",
            "FRAC_WIDTH => FRAC_WIDTH",
            "INPUT_SIZE => INPUT_SIZE",
            "HIDDEN_SIZE => HIDDEN_SIZE",
            "X_H_ADDR_WIDTH => X_H_ADDR_WIDTH",
            "HIDDEN_ADDR_WIDTH => HIDDEN_ADDR_WIDTH",
            "W_ADDR_WIDTH => W_ADDR_WIDTH",
        )
        uut_port_map.signal_list.append("clock => clock")
        uut_port_map.signal_list.append("reset => reset")
        uut_port_map.signal_list.append("enable => enable")
        uut_port_map.signal_list.append("x_h_we => x_config_en")
        uut_port_map.signal_list.append("x_h_data => x_config_data")
        uut_port_map.signal_list.append("x_h_addr => x_config_addr")
        uut_port_map.signal_list.append("c_we => c_config_en")
        uut_port_map.signal_list.append("c_data_in => c_config_data")
        uut_port_map.signal_list.append("c_addr_in => c_config_addr")
        uut_port_map.signal_list.append("done => done")
        uut_port_map.signal_list.append("h_out_en => h_out_en")
        uut_port_map.signal_list.append("h_out_data => h_out_data")
        uut_port_map.signal_list.append("h_out_addr => h_out_addr")

        test_cases = TestCasesLSTMCell(
            reference_h_out=self.h_out,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        test_process = Process(identifier="test")
        test_process.process_statements_list = [t for t in test_cases()]

        architecture = Architecture(
            design_unit=self.component_name + "_tb",
        )
        architecture.architecture_declaration_list.append(
            "signal clk_period : time := 10 ns"
        )

        architecture.architecture_declaration_list.append("signal clock : std_logic")
        architecture.architecture_declaration_list.append("signal enable : std_logic")
        architecture.architecture_declaration_list.append(
            "signal reset: std_logic:='0'"
        )
        architecture.architecture_declaration_list.append(
            "signal x_config_en: std_logic:='0'"
        )
        architecture.architecture_declaration_list.append(
            "signal x_config_data:std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal x_config_addr:std_logic_vector(X_H_ADDR_WIDTH-1 downto 0) :=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal h_config_en: std_logic:='0'"
        )
        architecture.architecture_declaration_list.append(
            "signal h_config_data:std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal h_config_addr:std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0) :=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal c_config_en: std_logic:='0'"
        )
        architecture.architecture_declaration_list.append(
            "signal c_config_data:std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal c_config_addr:std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0) :=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal done :  std_logic:='0'"
        )
        architecture.architecture_declaration_list.append("signal h_out_en : std_logic")
        architecture.architecture_declaration_list.append(
            "signal h_out_addr : std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0) :=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "signal h_out_data : std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0')"
        )
        architecture.architecture_declaration_list.append(
            "type X_H_ARRAY is array (0 to 511) of signed(16-1 downto 0)"
        )
        architecture.architecture_declaration_list.append(
            "type C_ARRAY is array (0 to 511) of signed(16-1 downto 0)"
        )
        architecture.architecture_declaration_list.append(
            f"signal test_x_h_data : X_H_ARRAY := ({','.join(self.test_x_h_data)},others=>(others=>'0'))"
        )
        architecture.architecture_declaration_list.append(
            f"signal test_c_data : C_ARRAY := ({','.join(self.test_c_data)},others=>(others=>'0'))"
        )
        architecture.architecture_component_list.append(procedure_0)
        architecture.architecture_component_list.append(procedure_1)
        architecture.architecture_process_list.append(process)
        architecture.architecture_port_map_list.append(uut_port_map)
        architecture.architecture_assignment_at_end_of_declaration_list.append(
            "clk <= clock"
        )
        architecture.architecture_statement_part = test_process

        code = chain(library(), entity(), architecture())
        return code


class TestCasesLSTMCommonGate(TestBenchBase):
    def __init__(
        self,
        x_mem_list_for_testing: list[FixedPoint],
        w_mem_list_for_testing: list[FixedPoint],
        b_list_for_testing: list[FixedPoint],
        y_list_for_testing: list[FixedPoint],
        y_variable_name: str = "y",
    ):
        assert (
            len(x_mem_list_for_testing)
            == len(w_mem_list_for_testing)
            == len(b_list_for_testing)
            == len(y_list_for_testing)
        )

        def to_hex(value: FixedPoint) -> str:
            return hex_representation(value.to_hex())

        self.x_mem_list_for_testing = list(map(to_hex, x_mem_list_for_testing))
        self.w_mem_list_for_testing = list(map(to_hex, w_mem_list_for_testing))
        self.b_list_for_testing = list(map(to_hex, b_list_for_testing))
        self.y_list_for_testing = list(map(int, y_list_for_testing))
        self.y_variable_name = y_variable_name

    def _body(self) -> Code:
        counter = 0
        yield f"vector_len <= to_unsigned(10, VECTOR_LEN_WIDTH)"
        for x_mem_value, w_mem_value, b, y_value in zip(
            self.x_mem_list_for_testing,
            self.w_mem_list_for_testing,
            self.b_list_for_testing,
            self.y_list_for_testing,
        ):
            yield f"X_MEM <= ({x_mem_value})"
            yield f"W_MEM <= ({w_mem_value})"
            yield f"b <= {b}"
            yield from (
                "reset <= '1'",
                "wait for 2*clk_period",
                "wait until clock = '0'",
                "reset <= '0'",
                "wait until ready = '1'",
            )

            yield f"report \"expected output is {y_value}, value of '{self.y_variable_name}' is \" & integer'image(to_integer(signed({self.y_variable_name})))"
            yield f'assert {self.y_variable_name}={y_value} report "The {counter}. test case fail" severity error'
            yield "reset <= '1'"
            yield "wait for 1*clk_period"
            counter = counter + 1

    def __call__(self) -> Code:
        yield from iter(self)


class TestCasesLSTMCell(TestBenchBase):
    def __init__(
        self,
        reference_h_out: list[FixedPoint],
        input_size: int = 0,
        hidden_size: int = 0,
    ):
        self.reference_h_out = list(map(int, reference_h_out))

        assert (input_size != 0) and (
            hidden_size != 0
        ), "hidden_size and input_size is not set yet"

        self.len_of_x_h_vector = input_size + hidden_size
        self.len_of_cell_vector = hidden_size
        self.len_of_h_vector = hidden_size

    def _body(self) -> Code:
        yield f"reset <= '1'"
        yield f"h_out_en <= '0'"
        yield f"wait for 2*clk_period"
        yield f"reset <= '0'"
        yield f"for ii {Keywords.IN.value} 0 to {str(self.len_of_x_h_vector-1)} loop send_x_h_data(std_logic_vector(to_unsigned(ii, X_H_ADDR_WIDTH)), std_logic_vector(test_x_h_data(ii)), clock, x_config_en, x_config_addr, x_config_data)"
        yield f"wait for 10 ns"
        yield f"{Keywords.END.value} loop"
        yield f"for ii {Keywords.IN.value} 0 to {str(self.len_of_cell_vector-1)} loop send_c_data(std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH)), std_logic_vector(test_c_data(ii)), clock, c_config_en, c_config_addr, c_config_data)"
        yield f"wait for 10 ns"
        yield f"{Keywords.END.value} loop"
        yield f"enable <= '1'"
        yield f"wait until done = '1'"
        yield f"wait for 1*clk_period"
        yield f"enable <= '0'"
        yield f"-- reference h_out: {str(self.reference_h_out)}"
        yield f"for ii in 0 to {str(self.len_of_h_vector-1)} loop h_out_addr <= std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH))"
        yield f"h_out_en <= '1'"
        yield f"wait for 2*clk_period"
        yield f'report "The value of h_out(" & integer\'image(ii)& ") is " & integer\'image(to_integer(signed(h_out_data)))'
        yield f"{Keywords.END.value} loop"
        yield f"wait for 10*clk_period"

    def __call__(self) -> Code:
        yield from iter(self)
