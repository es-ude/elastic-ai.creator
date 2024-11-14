from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath

from .design import LinearDesign


@pytest.fixture
def linear_design() -> LinearDesign:
    return LinearDesign(
        name="linear",
        in_feature_num=3,
        out_feature_num=2,
        total_bits=16,
        frac_bits=8,
        weights=[[1, 1, 1], [1, 1, 1]],
        bias=[1, 1],
    )


def save_design(design: LinearDesign) -> dict[str, str]:
    destination = InMemoryPath("linear", parent=None)
    design.save_to(destination)
    files = cast(list[InMemoryFile], list(destination.children.values()))
    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(linear_design: LinearDesign) -> None:
    saved_files = save_design(linear_design)

    expected_files = {"linear_w_rom.vhd", "linear_b_rom.vhd", "linear.vhd"}
    actual_files = set(saved_files.keys())

    assert expected_files == actual_files


def test_weight_rom_code_generated_correctly(linear_design: LinearDesign) -> None:
    expected_code = """library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity linear_w_rom is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(3-1 downto 0);
        data : out std_logic_vector(16-1 downto 0)
    );
end entity linear_w_rom;
architecture rtl of linear_w_rom is
    type linear_w_rom_array_t is array (0 to 2**3-1) of std_logic_vector(16-1 downto 0);
    signal ROM : linear_w_rom_array_t:=("0000000000000001","0000000000000001","0000000000000001","0000000000000001","0000000000000001","0000000000000001","0000000000000000","0000000000000000");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(clk)
    begin
        if rising_edge(clk) then
            if (en = '1') then
                data <= ROM(conv_integer(addr));
            end if;
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear_w_rom.vhd"]
    assert expected_code == actual_code


def test_bias_rom_code_generated_correctly(linear_design: LinearDesign) -> None:
    expected_code = """library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity linear_b_rom is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(1-1 downto 0);
        data : out std_logic_vector(16-1 downto 0)
    );
end entity linear_b_rom;
architecture rtl of linear_b_rom is
    type linear_b_rom_array_t is array (0 to 2**1-1) of std_logic_vector(16-1 downto 0);
    signal ROM : linear_b_rom_array_t:=("0000000000000001","0000000000000001");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(clk)
    begin
        if rising_edge(clk) then
            if (en = '1') then
                data <= ROM(conv_integer(addr));
            end if;
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear_b_rom.vhd"]
    assert expected_code == actual_code


def test_linear_code_generated_correctly(linear_design: LinearDesign) -> None:
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

library work;
use work.all;

entity linear is -- layer_name is for distinguish same type of layers (with various weights) in one module
    generic (
        DATA_WIDTH   : integer := 16;
        FRAC_WIDTH   : integer := 8;
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 1;
        IN_FEATURE_NUM : integer := 3;
        OUT_FEATURE_NUM : integer := 2;
        RESOURCE_OPTION : string := "auto" -- can be "distributed", "block", or  "auto"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);

        x   : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH-1 downto 0);

        done   : out std_logic
    );
end linear;

architecture rtl of linear is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    -- macc
    function multiply_accumulate(w : in signed(DATA_WIDTH-1 downto 0);
                    x : in signed(DATA_WIDTH-1 downto 0);
                    y_0 : in signed(2*DATA_WIDTH-1 downto 0)
            ) return signed is

        variable TEMP : signed(DATA_WIDTH*2-1 downto 0) := (others=>'0');
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin
        TEMP := w * x;

        return TEMP+y_0;
    end function;

    function cut_down(x: in signed(2*DATA_WIDTH-1 downto 0))return signed is
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin

        TEMP2 := x(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP3 := x(FRAC_WIDTH-1 downto 0);
        if TEMP2(DATA_WIDTH-1) = '1' and TEMP3 /= 0 then
            TEMP2 := TEMP2 + 1;
        end if;

        if x>0 and TEMP2<0 then
            TEMP2 := ('0', others => '1');
        elsif x<0 and TEMP2>0 then
            TEMP2 := ('1', others => '0');
        end if;
        return TEMP2;
    end function;

    -- Log2 function is for calculating the bitwidth of the address lines
    -- for bias and weights rom
    function log2(val : INTEGER) return natural is
        variable res : natural;
    begin
        for i in 1 to 31 loop
            if (val <= (2 ** i)) then
                res := i;
                exit;
            end if;
        end loop;
        return res;
    end function log2;

    -----------------------------------------------------------
    -- Signals
    -----------------------------------------------------------
    constant FXP_ZERO : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    constant FXP_ONE : signed(DATA_WIDTH-1 downto 0) := to_signed(2**FRAC_WIDTH,DATA_WIDTH);

    type t_state is (s_stop, s_forward, s_idle);

    signal n_clock : std_logic;
    signal w_in : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal b_in : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');

    signal addr_w : std_logic_vector(log2(IN_FEATURE_NUM*OUT_FEATURE_NUM)-1 downto 0) := (others=>'0');
    --signal addr_b : std_logic_vector((log2(OUT_FEATURE_NUM)-1) downto 0) := (others=>'0');
    signal addr_b : std_logic_vector(Y_ADDR_WIDTH-1 downto 0) := (others=>'0');

    signal fxp_x, fxp_w, fxp_b, fxp_y : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal macc_sum : signed(2*DATA_WIDTH-1 downto 0) := (others=>'0');

    signal reset : std_logic := '0';
    signal state : t_state;

    -- simple solution for the output buffer
    type t_y_array is array (0 to OUT_FEATURE_NUM) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_ram : t_y_array;
    attribute rom_style : string;
    attribute rom_style of y_ram : signal is RESOURCE_OPTION;

begin

    -- connecting signals to ports
    n_clock <= not clock;

    fxp_w <= signed(w_in);
    fxp_x <= signed(x);
    fxp_b <= signed(b_in);

    -- connects ports
    reset <= not enable;

    linear_main : process (clock, enable, reset)
        variable current_neuron_idx : integer range 0 to OUT_FEATURE_NUM-1 := 0;
        variable current_input_idx : integer  range 0 to IN_FEATURE_NUM-1 := 0;
        variable var_addr_w : integer range 0 to OUT_FEATURE_NUM*IN_FEATURE_NUM-1 := 0;
        variable var_sum, var_y : signed(2*DATA_WIDTH-1 downto 0);
        variable var_w, var_x : signed(DATA_WIDTH-1 downto 0);
        variable y_write_en : std_logic;
        variable var_y_write_idx : integer;
    begin

        if (reset = '1') then
            state <= s_stop;
            done <= '0';

            current_neuron_idx := 0;
            current_input_idx := 0;
            var_addr_w := 0;

        elsif rising_edge(clock) then

            if state=s_stop then
                state <= s_forward;

                -- first add b accumulated sum
                var_y := (others=>'0');
                var_x := fxp_b;
                var_w := FXP_ONE;
            elsif state=s_forward then

                -- remapping to x and w
                var_y := macc_sum;
                var_x := fxp_x;
                var_w := fxp_w;

                if current_input_idx<IN_FEATURE_NUM-1 then
                    current_input_idx := current_input_idx + 1;
                    var_addr_w := var_addr_w + 1;
                else
                    current_input_idx := 0;

                    y_write_en := '1';
                    var_y_write_idx := current_neuron_idx;

                    if current_neuron_idx<OUT_FEATURE_NUM-1 then
                        current_neuron_idx := current_neuron_idx + 1;
                        var_addr_w := var_addr_w + 1;
                        state <= s_stop;
                    else
                        state <= s_idle;
                        done <= '1';
                    end if;

                end if;
            end if;

            var_sum := multiply_accumulate(var_w, var_x, var_y);
            macc_sum <= var_sum;

            if y_write_en='1'then
                y_ram(var_y_write_idx) <= std_logic_vector(cut_down(var_sum));
                y_write_en := '0';
            end if;

        end if;

        x_address <= std_logic_vector(to_unsigned(current_input_idx, x_address'length));
        addr_w <= std_logic_vector(to_unsigned(var_addr_w, addr_w'length));
        addr_b <= std_logic_vector(to_unsigned(current_neuron_idx, addr_b'length));
    end process linear_main;

    y_reading : process (clock, state)
    begin
        if (state=s_idle) or (state=s_stop) then
            if falling_edge(clock) then
                -- After the layer in at idle mode, y is readable
                -- but it only update at the rising edge of the clock
                y <= y_ram(to_integer(unsigned(y_address)));
            end if;
        end if;
    end process y_reading;

    -- Weights
    rom_w : entity work.linear_w_rom(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_w,
        data => w_in
    );

    -- Bias
    rom_b : entity work.linear_b_rom(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_b,
        data => b_in
    );

end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear.vhd"]
    assert expected_code == actual_code
