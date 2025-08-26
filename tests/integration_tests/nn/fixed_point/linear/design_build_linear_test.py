from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign


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
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;

entity linear_w_rom is
    generic (
        ROM_ADDR_WIDTH : integer := 3;
        ROM_DATA_WIDTH : integer := 16
    );
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(ROM_ADDR_WIDTH-1 downto 0);
        data : out std_logic_vector(ROM_DATA_WIDTH-1 downto 0)
    );
end entity linear_w_rom;
architecture rtl of linear_w_rom is
    type linear_w_rom_array_t is array (0 to 2**ROM_ADDR_WIDTH-1) of std_logic_vector(ROM_DATA_WIDTH-1 downto 0);
    signal ROM : linear_w_rom_array_t:=("0000000000000001", "0000000000000001", "0000000000000001", "0000000000000001", "0000000000000001", "0000000000000001", "0000000000000000", "0000000000000000");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(addr)
    begin
        if (en = '1') then
            data <= ROM(to_integer(unsigned(addr)));
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear_w_rom.vhd"]
    assert expected_code == actual_code


def test_bias_rom_code_generated_correctly(linear_design: LinearDesign) -> None:
    expected_code = """library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;

entity linear_b_rom is
    generic (
        ROM_ADDR_WIDTH : integer := 1;
        ROM_DATA_WIDTH : integer := 16
    );
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(ROM_ADDR_WIDTH-1 downto 0);
        data : out std_logic_vector(ROM_DATA_WIDTH-1 downto 0)
    );
end entity linear_b_rom;
architecture rtl of linear_b_rom is
    type linear_b_rom_array_t is array (0 to 2**ROM_ADDR_WIDTH-1) of std_logic_vector(ROM_DATA_WIDTH-1 downto 0);
    signal ROM : linear_b_rom_array_t:=("0000000000000001", "0000000000000001");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(addr)
    begin
        if (en = '1') then
            data <= ROM(to_integer(unsigned(addr)));
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear_b_rom.vhd"]
    assert expected_code == actual_code


def test_linear_code_generated_correctly(linear_design: LinearDesign) -> None:
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.all;

-- layer_name is for distinguish same type of layers (with various weights) in one module
-- MAC operator with one multiplier
entity linear is
    generic (
        DATA_WIDTH   : integer := 16;
        FRAC_WIDTH   : integer := 8;
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 1;
        IN_FEATURE_NUM : integer := 3;
        OUT_FEATURE_NUM : integer := 2;
        RESOURCE_OPTION : string := "auto"
        -- can be "distributed", "block", or  "auto"
    );
    port (
        enable      : in    std_logic;
        clock       : in    std_logic;
        x_address   : out   std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address   : in    std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        x           : in    std_logic_vector(DATA_WIDTH-1 downto 0);
        y           : out   std_logic_vector(DATA_WIDTH-1 downto 0);
        done        : out   std_logic
    );
end linear;

architecture rtl of linear is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    -- FXP_ROUNDING with clamping if range violation is available
    function FXP_ROUNDING(
        x0: in signed(2*DATA_WIDTH-1 downto 0)
    ) return signed is
        variable TEMP0 : signed(DATA_WIDTH-1 downto 0) := (others => '0');
        variable TEMP1 : signed(FRAC_WIDTH-1 downto 0) := (others => '0');
    begin
        TEMP0 := x0(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP1 := x0(FRAC_WIDTH-1 downto 0);

        if (x0(2*DATA_WIDTH-1) = '1' and TEMP0(DATA_WIDTH-1) = '0') then
            TEMP0 := ('1', others => '0');
        elsif (x0(2*DATA_WIDTH-1) = '0' and TEMP0(DATA_WIDTH-1) = '1') then
            TEMP0 := ('0', others => '1');
        else
            if TEMP0(DATA_WIDTH-1) = '1' and TEMP1 /= 0 then
                TEMP0 := TEMP0 + 1;
            end if;
        end if;

        return TEMP0;
    end function;

    -- log2 function is for calculating the bitwidth of the address lines
    function log2(
        val : INTEGER
    ) return natural is
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
    -- Process
    -----------------------------------------------------------
    signal addr_w   : unsigned(log2(IN_FEATURE_NUM*OUT_FEATURE_NUM)-1 downto 0) := (others => '0');
    signal addr_b   : unsigned(Y_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal addr_x   : unsigned(X_ADDR_WIDTH-1 downto 0) := (others => '0');

    signal w_in, b_in           : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal fxp_x, fxp_w, fxp_b  : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal buf_x, buf_w, buf_b  : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal mac_y                : signed(2*DATA_WIDTH-1 downto 0) := (others=>'0');
    signal enable_mac, reset_mac, done_int : std_logic;

    -- simple solution for the output buffer
    type t_y_array is array (0 to OUT_FEATURE_NUM) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_ram                    : t_y_array;
    attribute rom_style             : string;
    attribute rom_style of y_ram    : signal is RESOURCE_OPTION;
begin
    -- connecting signals to ports
    fxp_w <= signed(w_in);
    fxp_x <= signed(x);
    fxp_b <= signed(b_in);

    done <= done_int;
    x_address <= std_logic_vector(addr_x);
    enable_mac <= enable and not done_int;

    -- Pipelined MAC operator and saving into buffer
    mac: process(clock)
    begin
        if rising_edge(clock) then
            if (enable_mac = '0') then
                buf_x <= (others => '0');
                buf_w <= (others => '0');
                buf_b <= (others => '0');
                mac_y <= (others => '0');
            else
                if (reset_mac = '1') then
                    buf_x <= (others => '0');
                    buf_w <= (others => '0');
                    buf_b <= (others => '0');
                    mac_y <= (others => '0');
                    y_ram(to_integer(unsigned(addr_b))) <= std_logic_vector(FXP_ROUNDING(mac_y + buf_w * buf_x + SHIFT_LEFT(RESIZE(buf_b, 2*DATA_WIDTH), FRAC_WIDTH)));
                else
                    buf_x <= fxp_x;
                    buf_w <= fxp_w;
                    buf_b <= fxp_b;
                    mac_y <= mac_y + (buf_w * buf_x);
                end if;
            end if;
        end if;
    end process mac;

    -- Counter Operator for controlling the linear layer
    control : process (clock)
    begin
        if rising_edge(clock) then
            if (enable = '0') then
                done_int <= '0';
                addr_x <= (others => '0');
                addr_w <= (others => '0');
                addr_b <= (others => '0');
                reset_mac <= '0';
            else
                if (done_int <= '0') then
                    if (addr_x = IN_FEATURE_NUM-1) then
                        if (reset_mac = '0') then
                            reset_mac <= '1';
                        else
                            reset_mac <= '0';

                            addr_x <= (others => '0');
                            if (addr_b = OUT_FEATURE_NUM-1) then
                                addr_b <= (others => '0');
                                addr_w <= (others => '0');
                                done_int <= '1';
                            else
                                addr_b <= addr_b + 1;
                                addr_w <= addr_w + 1;
                                done_int <= '0';
                            end if;
                        end if;
                    else
                        done_int <= '0';
                        addr_x <= addr_x + 1;
                        addr_b <= addr_b;
                        addr_w <= addr_w + 1;
                    end if;
                else
                    done_int <= '1';
                    addr_x <= (others => '0');
                    addr_w <= (others => '0');
                    addr_b <= (others => '0');
                end if;
            end if;
        end if;
    end process control;

    -- Reading operator
    y_reading : process (clock)
    begin
        if rising_edge(clock) then
            if (done_int = '1') then
                y <= y_ram(to_integer(unsigned(y_address)));
            end if;
        end if;
    end process y_reading;

    -- Weights
    rom_w : entity work.linear_w_rom(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => std_logic_vector(addr_w),
        data => w_in
    );

    -- Bias
    rom_b : entity work.linear_b_rom(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => std_logic_vector(addr_b),
        data => b_in
    );
end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear.vhd"]
    assert expected_code == actual_code
