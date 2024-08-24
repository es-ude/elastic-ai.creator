from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.linear.test_linear import inputs, linear_layer


@pytest.fixture
def saved_files(linear_layer, inputs):
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()
    design = linear_layer.create_design("linear_0")

    destination = InMemoryPath("linear_0", parent=None)
    design.save_to(destination)
    files = cast(list[InMemoryFile], list(destination.children.values()))

    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(saved_files) -> None:
    # saved_files = get_saved_files(linear_layer)
    expected_files = {
        "linear_0_b_rom.vhd",
        "linear_0_ram.vhd",
        "linear_0_tb.vhd",
        "linear_0_w_rom.vhd",
        "linear_0.vhd",
    }
    actual_files = set(saved_files.keys())
    assert expected_files == actual_files


def test_linear_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["linear_0.vhd"]

    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.all;
entity linear_0 is
    generic (
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 4;
        DATA_WIDTH : integer := 8;
        IN_FEATURES : integer := 3;
        OUT_FEATURES : integer := 10;
        M_Q : integer := 236223;
        M_Q_SHIFT : integer := 25;
        Z_X : integer := 42;
        Z_W : integer := 0;
        Z_B : integer := 0;
        Z_Y : integer := 69;
        M_Q_DATA_WIDTH : integer := 19;
        Y_RESOURCE_OPTION : string := "auto"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x   : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end linear_0;
architecture rtl of linear_0 is
    function multiply_accumulate(
                    w : in signed(DATA_WIDTH downto 0);
                    x_in : in signed(DATA_WIDTH downto 0);
                    y_0 : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0)
            ) return signed is
        variable TMP : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    begin
        TMP := w * x_in;
        return TMP + y_0;
    end function;
    function scaling(x_in : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
    scaler_m : in signed(M_Q_DATA_WIDTH -1 downto 0);
    scaler_m_shift : in integer
    ) return signed is
    variable TMP_1 : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_2 : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable TMP_3 : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH -1 downto 0) := (others=>'0');
    variable is_negative : boolean := x_in(x_in'left) = '1';
    begin
        if is_negative then
            TMP_1 := -x_in * scaler_m;
        else
            TMP_1 := x_in * scaler_m;
        end if;
        TMP_2 := shift_right(TMP_1, scaler_m_shift);
        TMP_3 := TMP_2;
        if TMP_1(scaler_m_shift-1) = '1' then
            TMP_3 := TMP_2 + 1;
        end if;
        if is_negative then
            return -resize(TMP_3, DATA_WIDTH + 1);
        else
            return resize(TMP_3, DATA_WIDTH + 1);
        end if;
    end function;
    function log2(val : INTEGER) return natural is
        variable result : natural;
    begin
        for i in 1 to 31 loop
            if (val <= (2 ** i)) then
                result := i;
                exit;
            end if;
        end loop;
        return result;
    end function log2;
    signal M_Q_SIGNED:signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q, M_Q_DATA_WIDTH);
    signal n_clock : std_logic;
    signal reset : std_logic := '0';
    type t_layer_state is (s_stop, s_forward, s_finished);
    signal layer_state : t_layer_state;
    type t_mac_state is (s_stop, s_init, s_preload, s_accumulate, s_scaling, s_output, s_done);
    signal mac_state : t_mac_state;
    signal x_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal x_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal w_in : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal w_addr : std_logic_vector(log2(IN_FEATURES*OUT_FEATURES) - 1 downto 0) := (others=>'0');
    signal w_int : signed(DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal w_sub_z : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal b_in : std_logic_vector(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    signal b_addr : std_logic_vector(log2(OUT_FEATURES) - 1 downto 0) := (others=>'0');
    signal b_int : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
    signal y_store_en : std_logic;
    signal y_scaled : signed(DATA_WIDTH downto 0) := (others=>'0');
    signal y_store_addr : integer range 0 to OUT_FEATURES;
    signal y_store_addr_std : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal y_store_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal macc_sum : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others=>'0');
begin
    n_clock <= not clock;
    w_int <= signed(w_in);
    x_int <= signed(x);
    b_int <= signed(b_in);
    reset <= not enable;
    fsm : process (clock, reset)
    begin
        if (reset = '1') then
            layer_state <= s_stop;
            done <= '0';
        elsif rising_edge(clock) then
            if layer_state=s_stop then
                layer_state <= s_forward;
            elsif layer_state=s_forward then
                if mac_state=s_done then
                    layer_state <= s_finished;
                else
                    layer_state <= s_forward;
                end if;
            else
                done <= '1';
                layer_state <= s_finished;
            end if;
        end if;
    end process fsm;
    mac : process( clock, layer_state )
        variable neuron_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable input_idx : integer  range 0 to IN_FEATURES - 1 := 0;
        variable weight_idx : integer range 0 to OUT_FEATURES * IN_FEATURES-1 := 0;
        variable bias_idx : integer range 0 to OUT_FEATURES-1 := 0;
        variable output_idx : integer  range 0 to OUT_FEATURES - 1 := 0;
        variable mac_cnt : integer range 0 to IN_FEATURES+1 := 0;
        variable input_offset : integer;
        variable var_product : signed(DATA_WIDTH - 1 downto 0);
        variable var_b_add_z_b : integer;
        variable var_y_store : signed(DATA_WIDTH downto 0);
    begin
        if rising_edge(clock) then
            if layer_state=s_stop then
                mac_state <= s_init;
                neuron_idx := 0;
                input_idx := 0;
                weight_idx := 0;
                bias_idx := 0;
                output_idx := 0;
                input_offset :=0;
                mac_cnt :=0;
                y_store_en <= '0';
            elsif layer_state=s_forward then
                case mac_state is
                    when s_init =>
                        input_idx := input_offset;
                        mac_cnt :=0;
                        mac_state <= s_preload;
                        y_store_en <= '0';
                    when s_preload =>
                        x_sub_z <= to_signed(0, x_sub_z'length);
                        w_sub_z <= to_signed(0, w_sub_z'length);
                        var_b_add_z_b := to_integer(b_int) +Z_B;
                        macc_sum <= to_signed(var_b_add_z_b, macc_sum'length);
                        input_idx := input_idx + 1;
                        weight_idx := weight_idx + 1;
                        mac_state <= s_accumulate;
                    when s_accumulate =>
                        x_sub_z <= x_int - to_signed(Z_X, x_sub_z'length);
                        w_sub_z <= w_int - to_signed(Z_W, w_sub_z'length);
                        macc_sum <= multiply_accumulate(w_sub_z, x_sub_z, macc_sum);
                        mac_cnt := mac_cnt + 1;
                        if mac_cnt <= IN_FEATURES then
                            if mac_cnt < IN_FEATURES-1 then
                                input_idx := input_idx + 1;
                                weight_idx := weight_idx + 1;
                            end if;
                            mac_state <= s_accumulate;
                        else
                            mac_state <= s_scaling;
                        end if;
                    when s_scaling =>
                        y_scaled <= scaling(macc_sum, M_Q_SIGNED, M_Q_SHIFT);
                        mac_state <= s_output;
                    when s_output =>
                        var_y_store := y_scaled + to_signed(Z_Y, y_scaled'length);
                        y_store_data <= std_logic_vector(resize(var_y_store, y_store_data'length));
                        y_store_addr <= output_idx;
                        y_store_en <= '1';
                        if neuron_idx < OUT_FEATURES-1 then
                            neuron_idx := neuron_idx + 1;
                            weight_idx := weight_idx + 1;
                            bias_idx := bias_idx + 1;
                            mac_state <= s_init;
                            output_idx := output_idx + 1;
                        else
                            mac_state <= s_done;
                        end if;

                    when others =>
                        mac_state <= s_done;
                end case;
            else
                mac_state <= s_done;
                y_store_en <= '0';
            end if;
            x_address <= std_logic_vector(to_unsigned(input_idx, x_address'length));
            w_addr <= std_logic_vector(to_unsigned(weight_idx, w_addr'length));
            b_addr <= std_logic_vector(to_unsigned(bias_idx, b_addr'length));
        end if;
    end process ;
    y_store_addr_std <= std_logic_vector(to_unsigned(y_store_addr, y_store_addr_std'length));
    ram_y : entity work.linear_0_ram(rtl)
    generic map (
        RAM_WIDTH => DATA_WIDTH,
        RAM_DEPTH_WIDTH => Y_ADDR_WIDTH,
        RAM_PERFORMANCE => "LOW_LATENCY",
        RESOURCE_OPTION => Y_RESOURCE_OPTION,
        INIT_FILE => ""
    )
    port map  (
        addra  => y_store_addr_std,
        addrb  => y_address,
        dina   => y_store_data,
        clka   => clock,
        clkb   => clock,
        wea    => y_store_en,
        enb    => '1',
        rstb   => '0',
        regceb => '0',
        doutb  => y
    );
    rom_w : entity work.linear_0_w_rom(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => w_addr,
        data => w_in
    );
    rom_b : entity work.linear_0_b_rom(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => b_addr,
        data => b_in
    );
end architecture;"""
    assert expected_code == actual_code


def test_linear_tb_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["linear_0_tb.vhd"]
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library work;
use work.all;
entity linear_0_tb is
    generic (
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 4;
        DATA_WIDTH : integer := 8;
        IN_FEATURES : integer := 3;
        OUT_FEATURES : integer := 10
    );
port(
    clk : out std_logic
    );
end entity;
architecture rtl of linear_0_tb is
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
        constant file_inputs:      string := "./data/linear_0_q_x.txt";
        constant file_labels:      string := "./data/linear_0_q_y.txt";
        constant file_pred:      string := "./data/linear_0_out.txt";
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
        report  "all files closed.";
        report "Time taken for processing = " & time'image(v_TIME);
        report "Simulation done.";
        assert false report "Simulation done. The `assertion failure` is intended to stop this simulation." severity FAILURE;
    end process;
    uut: entity work.linear_0(rtl)
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


def test_weight_rom_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["linear_0_w_rom.vhd"]
    expected_code = """library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity linear_0_w_rom is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(5-1 downto 0);
        data : out std_logic_vector(8-1 downto 0)
    );
end entity linear_0_w_rom;
architecture rtl of linear_0_w_rom is
    type linear_0_w_rom_array_t is array (0 to 2**5-1) of std_logic_vector(8-1 downto 0);
    signal ROM : linear_0_w_rom_array_t:=("00110101","11001011","00100000","00010101","11101011","00001011","00101010","11010110","00010101","01000000","11000000","00100000","01001010","10110110","00101010","01010101","10101011","00110101","01100000","10100000","01000000","01101010","10010110","01001010","01110101","10001011","01010101","01111111","10000000","01100000","00000000","00000000");
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
    assert expected_code == actual_code


def test_bias_rom_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["linear_0_b_rom.vhd"]
    expected_code = """library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity linear_0_b_rom is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(4-1 downto 0);
        data : out std_logic_vector(18-1 downto 0)
    );
end entity linear_0_b_rom;
architecture rtl of linear_0_b_rom is
    type linear_0_b_rom_array_t is array (0 to 2**4-1) of std_logic_vector(18-1 downto 0);
    signal ROM : linear_0_b_rom_array_t:=("000010001101000111","111101110010111001","000001010100101011","000000111000011100","111111000111100100","000000011100001110","000001110000111001","111110001111000111","000000111000011100","000010101001010110","000000000000000000","000000000000000000","000000000000000000","000000000000000000","000000000000000000","000000000000000000");
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
    assert expected_code == actual_code


def test_ram_code_generated_correctly(saved_files) -> None:
    actual_code = saved_files["linear_0_ram.vhd"]
    expected_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
entity linear_0_ram is
    generic (
        RAM_WIDTH : integer := 64;
        RAM_DEPTH_WIDTH : integer := 8;
        RAM_PERFORMANCE : string := "LOW_LATENCY";
        RESOURCE_OPTION : string := "auto";
        INIT_FILE : string := ""
    );
    port (
        addra : in std_logic_vector((RAM_DEPTH_WIDTH-1) downto 0);
        addrb : in std_logic_vector((RAM_DEPTH_WIDTH-1) downto 0);
        dina  : in std_logic_vector(RAM_WIDTH-1 downto 0);
        clka  : in std_logic;
        clkb  : in std_logic;
        wea   : in std_logic;
        enb   : in std_logic;
        rstb  : in std_logic;
        regceb: in std_logic;
        doutb : out std_logic_vector(RAM_WIDTH-1 downto 0)
    );
end linear_0_ram;
architecture rtl of linear_0_ram is
    constant C_RAM_WIDTH : integer := RAM_WIDTH;
    constant C_RAM_DEPTH : integer := 2**RAM_DEPTH_WIDTH;
    constant C_RAM_PERFORMANCE : string := RAM_PERFORMANCE;
    constant C_INIT_FILE : string := INIT_FILE;
    signal doutb_reg : std_logic_vector(C_RAM_WIDTH-1 downto 0) := (others => '0');
    type ram_type is array (0 to C_RAM_DEPTH-1) of std_logic_vector(C_RAM_WIDTH-1 downto 0);
    signal ram_data : std_logic_vector(C_RAM_WIDTH-1 downto 0);
    function init_from_file_or_zeroes(ramfile : string) return ram_type is
    begin
        return (others => (others => '0'));
    end;
    signal ram_name : ram_type := init_from_file_or_zeroes(C_INIT_FILE);
    attribute ram_style : string;
    attribute ram_style of ram_name : signal is RESOURCE_OPTION;
begin
    process(clka)
    begin
        if rising_edge(clka) then
            if wea = '1' then
                ram_name(to_integer(unsigned(addra))) <= dina;
            end if;
        end if;
    end process;
    process(clkb)
    begin
        if rising_edge(clkb) then
            if enb = '1' then
                ram_data <= ram_name(to_integer(unsigned(addrb)));
            end if;
        end if;
    end process;
    no_output_register : if C_RAM_PERFORMANCE = "LOW_LATENCY" generate
        doutb <= ram_data;
    end generate;
    output_register : if C_RAM_PERFORMANCE = "HIGH_PERFORMANCE" generate
        process(clkb)
        begin
            if rising_edge(clkb) then
                if rstb = '1' then
                    doutb_reg <= (others => '0');
                elsif regceb = '1' then
                    doutb_reg <= ram_data;
                end if;
            end if;
        end process;
        doutb <= doutb_reg;
    end generate;
end architecture rtl;"""
    assert expected_code == actual_code
