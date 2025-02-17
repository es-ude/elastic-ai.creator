library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity striding_shift_register_tb is
    generic (runner_cfg : string);
end entity;

architecture behav of striding_shift_register_tb is
    constant KERNEL_SIZE : natural := 3;
    constant NUM_CHANNELS : natural := 3;
    constant NUM_STEPS : natural := 1;
    constant TOTAL_LENGTH : natural := (KERNEL_SIZE + NUM_STEPS - 1) * NUM_CHANNELS;
    signal clk : std_logic := '1';
    signal rst : std_logic := '0';
    signal x : std_logic_vector(NUM_CHANNELS - 1 downto 0) := (others => '0');
    signal y : std_logic_vector(KERNEL_SIZE*NUM_CHANNELS - 1 downto 0) := (others => '0');
    signal register_valid : std_logic := '0';
    signal input_valid : std_logic := '0';
    constant clk_period : time := 20 ps;
begin
    -- Clock generation
    clk <= not clk after clk_period/2;

    dut_i : entity work.striding_shift_register(rtl)
        generic map (
            DATA_WIDTH => NUM_CHANNELS,
            NUM_POINTS => KERNEL_SIZE,
            STRIDE => 2
        )
        port map (
            d_in => x,
            d_out => y,
            clk => clk,
            valid_out => register_valid,
            valid_in => input_valid,
            rst => rst
        );

    main: process is
        procedure reset_register is begin
            input_valid <= '0';
            rst <= '1';
            wait for clk_period;
            rst <= '0';
            wait for clk_period;
            wait until rising_edge(clk);
        end procedure;

        procedure feed_input(
            data: std_logic_vector(2 downto 0);
            expected: std_logic_vector
        ) is begin
            input_valid <= '1';
            x <= data;
            wait for clk_period;
            check_equal(y, expected); -- checks the result of previous feed action
        end procedure;

        procedure register_not_filled is begin
            check_equal(register_valid, '0', "expected output invalid before register is full");
        end procedure;

        procedure register_is_filled is begin
            check_equal(register_valid, '1', "expected output valid after register is full");
        end procedure;

    begin
        test_runner_setup(runner, runner_cfg);
            if run("test_striding_shift_register") then
                reset_register;
                
                register_not_filled;
                feed_input(
                    b"001",
                    std_logic_vector'(b"000000000")
                );
                register_not_filled;
                feed_input(
                    b"111",
                    std_logic_vector'(b"000000001")
                );
                register_not_filled;
                feed_input(
                    b"011",
                    std_logic_vector'(b"000000001")
                );
                register_not_filled;
                feed_input(
                    b"111",
                    std_logic_vector'(b"000001011")
                );
                register_not_filled;
                feed_input(
                    b"111",
                    std_logic_vector'(b"000001011")
                );
                register_not_filled;
                feed_input(
                    b"010",
                    std_logic_vector'(b"001011111")
                );
                register_is_filled;
                feed_input(
                    b"001",
                    std_logic_vector'(b"001011111")
                );
                register_is_filled;
              
                
            end if;
        test_runner_cleanup(runner);
    end process;
end architecture;
