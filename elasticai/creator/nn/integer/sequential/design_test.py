from typing import cast

import pytest
from torch import nn

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.sequential.sequential import Sequential
from elasticai.creator.nn.integer.sequential.test_sequential import (
    eps,
    inputs,
    linear_layer_0,
    linear_layer_1,
    q_inputs,
    relu_layer_0,
)


@pytest.fixture
def sequential_instance(linear_layer_0, relu_layer_0, linear_layer_1):
    layers = nn.ModuleList()
    layers.append(linear_layer_0)
    layers.append(relu_layer_0)
    layers.append(linear_layer_1)
    return Sequential(*layers, name="sequential", quant_data_file_dir=None)


@pytest.fixture
def saved_files(sequential_instance, inputs):
    sequential_instance.train()
    sequential_instance.forward(inputs)
    sequential_instance.eval()
    sequential_instance.precompute()
    design = sequential_instance.create_design("sequential")

    destination = InMemoryPath("source", parent=None)
    design.save_to(destination)

    network_folder = None
    for name, child in destination.children.items():
        if name == "sequential" and isinstance(child, InMemoryPath):
            network_folder = child
            break
        elif isinstance(child, InMemoryPath):
            for sub_name, sub_child in child.children.items():
                if sub_name == "sequential" and isinstance(sub_child, InMemoryPath):
                    network_folder = sub_child
                    break
            if network_folder:
                break

    if network_folder is None:
        raise ValueError("Network folder not found")

    files = cast(list[InMemoryFile], list(network_folder.children.values()))
    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(saved_files) -> None:
    expected_files = {
        "sequential.vhd",
        "network_tb.vhd",
    }
    actual_files = set(saved_files.keys())
    assert expected_files == actual_files


def test_network_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["sequential.vhd"]
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.all;
entity sequential is
    port (
        enable: in std_logic;
        clock: in std_logic;
        x_address: out std_logic_vector(2-1 downto 0);
        y_address: in std_logic_vector(1-1 downto 0);
        x: in std_logic_vector(8-1 downto 0);
        y: out std_logic_vector(8-1 downto 0);
        done: out std_logic
    );
end sequential;
architecture rtl of sequential is
    signal i_linear_0_clock : std_logic := '0';
    signal i_linear_0_done : std_logic := '0';
    signal i_linear_0_enable : std_logic := '0';
    signal i_linear_0_x : std_logic_vector(7 downto 0) := (others => '0');
    signal i_linear_0_x_address : std_logic_vector(1 downto 0) := (others => '0');
    signal i_linear_0_y : std_logic_vector(7 downto 0) := (others => '0');
    signal i_linear_0_y_address : std_logic_vector(3 downto 0) := (others => '0');
    signal i_linear_1_clock : std_logic := '0';
    signal i_linear_1_done : std_logic := '0';
    signal i_linear_1_enable : std_logic := '0';
    signal i_linear_1_x : std_logic_vector(7 downto 0) := (others => '0');
    signal i_linear_1_x_address : std_logic_vector(3 downto 0) := (others => '0');
    signal i_linear_1_y : std_logic_vector(7 downto 0) := (others => '0');
    signal i_linear_1_y_address : std_logic_vector(0 downto 0) := (others => '0');
    signal i_relu_0_clock : std_logic := '0';
    signal i_relu_0_enable : std_logic := '0';
    signal i_relu_0_x : std_logic_vector(7 downto 0) := (others => '0');
    signal i_relu_0_y : std_logic_vector(7 downto 0) := (others => '0');
begin
    done <= i_linear_1_done;
    i_linear_0_clock <= clock;
    i_linear_0_enable <= enable;
    i_linear_0_x <= x;
    i_linear_0_y_address <= i_linear_1_x_address;
    i_linear_1_clock <= clock;
    i_linear_1_enable <= i_linear_0_done;
    i_linear_1_x <= i_relu_0_y;
    i_linear_1_y_address <= y_address;
    i_relu_0_clock <= clock;
    i_relu_0_enable <= i_linear_0_done;
    i_relu_0_x <= i_linear_0_y;
    x_address <= i_linear_0_x_address;
    y <= i_linear_1_y;
    i_linear_0 : entity work.linear_0(rtl)
    port map(
      clock => i_linear_0_clock,
      done => i_linear_0_done,
      enable => i_linear_0_enable,
      x => i_linear_0_x,
      x_address => i_linear_0_x_address,
      y => i_linear_0_y,
      y_address => i_linear_0_y_address
    );
    i_relu_0 : entity work.relu_0(rtl)
    port map(
      clock => i_relu_0_clock,
      enable => i_relu_0_enable,
      x => i_relu_0_x,
      y => i_relu_0_y
    );
    i_linear_1 : entity work.linear_1(rtl)
    port map(
      clock => i_linear_1_clock,
      done => i_linear_1_done,
      enable => i_linear_1_enable,
      x => i_linear_1_x,
      x_address => i_linear_1_x_address,
      y => i_linear_1_y,
      y_address => i_linear_1_y_address
    );
end rtl;"""

    assert expected_code == actual_code


def test_network_tb_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["network_tb.vhd"]
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library work;
use work.all;
entity network_tb is
    generic (
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 1;
        DATA_WIDTH : integer := 8;
        IN_FEATURES : integer := 3;
        OUT_FEATURES : integer := 1
    );
port(
    clk : out std_logic
    );
end entity;
architecture rtl of network_tb is
    constant C_CLK_PERIOD : time := 10 ns;
    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    signal uut_enable : std_logic := '0';
    signal x_addr : std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
    signal x_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal y_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal done : std_logic;
    type t_array_x is array (0 to IN_FEATURES - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal x_arr : t_array_x := (others=>(others=>'0'));
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
    data_read : process( clock )
    begin
        if rising_edge(clock) then
            x_in <= x_arr(to_integer(unsigned(x_addr)));
        end if;
    end process ;
    test_main : process
        constant file_inputs:      string := "./data/network_q_x.txt";
        constant file_labels:      string := "./data/network_q_y.txt";
        constant file_pred:      string := "./data/network_out.txt";
        file fp_inputs:      text;
        file fp_labels:      text;
        file fp_pred:      text;
        variable line_content:  integer;
        variable line_num:      line;
        variable filestatus:    file_open_status;
        variable input_rd_cnt : integer := 0;
        variable output_rd_cnt : integer := 0;
        variable v_TIME : time := 0 ns;
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
        y_addr <= (others=>'0');
        uut_enable <= '0';
        wait until reset='0';
        wait for C_CLK_PERIOD;
        while not ENDFILE (fp_inputs) loop
            input_rd_cnt := 0;
            while input_rd_cnt < IN_FEATURES loop
                readline (fp_inputs, line_num);
                read (line_num, line_content);
                x_arr(input_rd_cnt) <= std_logic_vector(to_signed(line_content, DATA_WIDTH));
                input_rd_cnt := input_rd_cnt + 1;
            end loop;
            wait for C_CLK_PERIOD;
            v_TIME := now;
            uut_enable <= '1';
            wait for C_CLK_PERIOD;
            wait until done='1';
            v_TIME := now - v_TIME;
            output_rd_cnt := 0;
            while output_rd_cnt<OUT_FEATURES loop
                readline (fp_labels, line_num);
                read (line_num, line_content);
                y_addr <= std_logic_vector(to_unsigned(output_rd_cnt, y_addr'length));
                wait for 2*C_CLK_PERIOD;
                report "Correct/Simulated = " & integer'image(line_content) & "/" & integer'image(to_integer(signed(y_out))) & ", Differece = " & integer'image(line_content - to_integer(signed(y_out)));
                write (line_num, to_integer(signed(y_out)));
                writeline(fp_pred, line_num);
                output_rd_cnt := output_rd_cnt + 1;
            end loop;
            uut_enable <= '0';
        end loop;
        wait until falling_edge(clock);
        file_close (fp_inputs);
        file_close (fp_labels);
        file_close (fp_pred);
        report "All files closed.";
        report "Time taken for processing = " & time'image(v_TIME);
        report "Simulation done.";
        assert false report "Simulation done. The `assertion failure` is intended to stop this simulation." severity FAILURE;
    end process ;
    uut: entity work.sequential(rtl)
    port map (
        enable => uut_enable,
        clock  => clock,
        x_address => x_addr,
        y_address => y_addr,
        x   => x_in,
        y   => y_out,
        done   => done
    );
end architecture;"""

    assert expected_code == actual_code
