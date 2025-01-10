from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.relu.test_relu import inputs, relu_layer


@pytest.fixture
def saved_files(relu_layer, inputs):
    relu_layer.forward(inputs)
    relu_layer.eval()
    design = relu_layer.create_design("relu_0")

    destination = InMemoryPath("relu_0", parent=None)
    design.save_to(destination)
    files = cast(list[InMemoryFile], list(destination.children.values()))

    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(saved_files) -> None:
    expected_files = {
        "relu_0_tb.vhd",
        "relu_0.vhd",
    }
    actual_files = set(saved_files.keys())
    assert expected_files == actual_files


def test_relu_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["relu_0.vhd"]

    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity relu_0 is
generic (
    DATA_WIDTH : integer := 8;
    THRESHOLD : integer := -43;
    CLOCK_OPTION : boolean := false
);
port (
    enable : in std_logic;
	clock  : in std_logic;
	x  : in std_logic_vector(DATA_WIDTH - 1 downto 0);
	y : out std_logic_vector(DATA_WIDTH - 1 downto 0)
);
end entity relu_0;
architecture rtl of relu_0 is
    signal signed_x : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal signed_y : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    clocked: if CLOCK_OPTION generate
        main_process : process (enable, clock)
        begin
            if (enable = '0') then
                signed_y <= to_signed(THRESHOLD, DATA_WIDTH);
            elsif (rising_edge(clock)) then
                if signed_x < THRESHOLD then
                    signed_y <= to_signed(THRESHOLD, DATA_WIDTH);
                else
                    signed_y <= signed_x;
                end if;
            end if;
        end process;
    end generate;
    async: if (not CLOCK_OPTION) generate
        process (enable, signed_x)
        begin
            if enable = '0' then
                signed_y <= to_signed(THRESHOLD, DATA_WIDTH);
            else
                if signed_x < THRESHOLD then
                    signed_y <= to_signed(THRESHOLD, DATA_WIDTH);
                else
                    signed_y <= signed_x;
                end if;
            end if;
        end process;
    end generate;
end architecture;"""
    assert expected_code == actual_code


def test_relu_tb_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["relu_0_tb.vhd"]
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library work;
use work.all;
entity relu_0_tb is
    generic (
        DATA_WIDTH : integer := 8;
        THRESHOLD : integer := -43;
        CLOCK_OPTION : boolean := false
    );
port(
    clk : out std_logic
    );
end entity;
architecture rtl of relu_0_tb is
    constant C_CLK_PERIOD : time := 10 ns;
    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    signal uut_enable : std_logic := '0';
    signal x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
begin
    CLK_GEN : process
    begin
        clock <= '1';
        wait for C_CLK_PERIOD/2;
        clock <= '0';
        wait for C_CLK_PERIOD/2;
    end process CLK_GEN;
    RESET_GEN : process
    begin
        reset <= '1',
                '0' after 20.0*C_CLK_PERIOD;
    wait;
    end process RESET_GEN;
    clk <= clock;
    test_main : process
        constant file_inputs:      string := "./data/relu_0_q_x.txt";
        constant file_labels:      string := "./data/relu_0_q_y.txt";
        constant file_pred:      string := "./data/relu_0_out.txt";
        file fp_inputs:      text;
        file fp_labels:      text;
        file fp_pred:      text;
        variable line_content:  integer;
        variable line_num:      line;
        variable filestatus:    file_open_status;
        variable input_rd_cnt : integer := 0;
        variable output_rd_cnt : integer := 0;
    begin
        file_open (filestatus, fp_inputs, file_inputs, READ_MODE);
        report file_inputs & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_labels, file_labels, READ_MODE);
        report file_labels & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        file_open (filestatus, fp_pred, file_pred, WRITE_MODE);
        report file_pred & LF & HT & "file_open_status = " &
                    file_open_status'image(filestatus);
        assert filestatus = OPEN_OK
            report "file_open_status /= file_ok"
            severity FAILURE;
        uut_enable <= '0';
        wait until reset='0';
        wait for C_CLK_PERIOD;
        uut_enable <= '1';
        while not ENDFILE (fp_inputs) loop
            readline (fp_inputs, line_num);
            read (line_num, line_content);
            x_in <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
            wait for 2*C_CLK_PERIOD;
            readline (fp_labels, line_num);
            read (line_num, line_content);
            report "Correct/Simulated = " & integer'image(line_content) & "/" & integer'image(to_integer(signed(y_out))) & ", Differece = " & integer'image(line_content - to_integer(signed(y_out)));
            write (line_num, to_integer(signed(y_out)));
            writeline(fp_pred, line_num);
        end loop;
        wait until falling_edge(clock);
        file_close (fp_inputs);
        file_close (fp_labels);
        file_close (fp_pred);
        report "All files closed.";
        wait;
    end process ;
    uut: entity work.relu_0(rtl)
    port map (
        enable => uut_enable,
        clock  => clock,
        x  => x_in,
        y  => y_out
    );
end architecture;"""
    assert expected_code == actual_code
