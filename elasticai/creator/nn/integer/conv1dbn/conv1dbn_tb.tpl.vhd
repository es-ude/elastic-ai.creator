library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name}_tb is
    generic (
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        IN_CHANNELS : integer := ${in_channels};
        OUT_CHANNELS : integer := ${out_channels};
        KERNEL_SIZE : integer := ${kernel_size}
    );
    port(
        clk : out std_logic
    );
end entity;

architecture rtl of ${name}_tb is
    constant C_CLK_PERIOD : time := 10 ns;

    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    signal enable : std_logic := '0';

    signal x_address : std_logic_vector(X_ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal x : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal y_address : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal y : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal done : std_logic := '0';

    type t_x_input is array (0 to IN_CHANNELS * KERNEL_SIZE - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    type t_y_output is array (0 to OUT_CHANNELS - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);

    signal x_input : t_x_input := (others => (others => '0'));
    signal y_expected : t_y_output := (others => (others => '0'));

begin
    CLK_GEN : process
    begin
        clock <= '1';
        wait for C_CLK_PERIOD / 2;
        clock <= '0';
        wait for C_CLK_PERIOD / 2;
    end process;

    TEST_PROCESS : process
        file fp_inputs: text;
        file fp_outputs: text;
        variable line : line;
        variable temp_data : integer;
    begin
        -- Open input files
        file_open(fp_inputs, "./data/${name}_q_x.txt", READ_MODE);
        file_open(fp_outputs, "./data/${name}_q_y.txt", READ_MODE);

        -- Read inputs
        for i in 0 to IN_CHANNELS * KERNEL_SIZE - 1 loop
            readline(fp_inputs, line);
            read(line, temp_data);
            x_input(i) <= std_logic_vector(to_signed(temp_data, DATA_WIDTH));
        end loop;

        -- Apply inputs
        enable <= '1';
        wait for C_CLK_PERIOD;

        for i in 0 to OUT_CHANNELS - 1 loop
            y_address <= std_logic_vector(to_unsigned(i, Y_ADDR_WIDTH));
            wait for 2 * C_CLK_PERIOD;
            report "Output: " & integer'image(to_integer(signed(y)));
        end loop;

        enable <= '0';

        wait;
    end process;

    -- Instantiate UUT
    uut: entity ${work_library_name}.${name}(rtl)
    port map (
        enable => enable,
        clock => clock,
        x_address => x_address,
        x => x,
        y_address => y_address,
        y => y,
        done => done
    );
end architecture;
