library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

library ${work_library_name};
use ${work_library_name}.all;

entity ${name}_tb is
    generic (
        DATA_WIDTH : integer := ${data_width};
        IN_CHANNELS : integer := ${in_channels};
        OUT_CHANNELS : integer := ${out_channels};
        KERNEL_SIZE : integer := ${kernel_size}
    );
    port (
        clk : out std_logic
    );
end entity;

architecture rtl of ${name}_tb is
    constant CLK_PERIOD : time := 10 ns;

    signal clock : std_logic := '0';
    signal enable : std_logic := '0';
    signal reset : std_logic := '0';

    signal x : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal y : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal done : std_logic;

    type t_x_input is array (0 to IN_CHANNELS - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    type t_y_expected is array (0 to OUT_CHANNELS - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);

    signal x_input : t_x_input := (others => (others => '0'));
    signal y_expected : t_y_expected := (others => (others => '0'));

begin
    -- Clock Generation
    clk_process : process
    begin
        clock <= '1';
        wait for CLK_PERIOD / 2;
        clock <= '0';
        wait for CLK_PERIOD / 2;
    end process;

    -- Test Process
    test_process : process
        file fp_inputs : text;
        file fp_outputs : text;
        variable line : line;
        variable temp_data : integer;
    begin
        -- Open Input and Expected Output Files
        file_open(fp_inputs, "./data/${name}_q_x.txt", READ_MODE);
        file_open(fp_outputs, "./data/${name}_q_y.txt", READ_MODE);

        -- Read Input Data
        for i in 0 to IN_CHANNELS - 1 loop
            readline(fp_inputs, line);
            read(line, temp_data);
            x_input(i) <= std_logic_vector(to_signed(temp_data, DATA_WIDTH));
        end loop;

        -- Wait for Reset
        enable <= '0';
        wait for CLK_PERIOD * 2;
        enable <= '1';

        -- Wait for the UUT to Finish
        wait until done = '1';
        wait for CLK_PERIOD;

        -- Compare Outputs
        for i in 0 to OUT_CHANNELS - 1 loop
            readline(fp_outputs, line);
            read(line, temp_data);
            assert std_logic_vector(to_signed(temp_data, DATA_WIDTH)) = y
            report "Mismatch at output " & integer'image(i) & ": Expected = " & integer'image(temp_data)
                & ", Received = " & integer'image(to_integer(signed(y)))
            severity failure;
        end loop;

        -- End Simulation
        report "Testbench Completed Successfully.";
        wait;
    end process;

    -- Instantiate Unit Under Test (UUT)
    uut : entity ${work_library_name}.${name}(rtl)
        port map (
            enable => enable,
            clock  => clock,
            x      => x,
            y      => y,
            done   => done
        );
end architecture;
