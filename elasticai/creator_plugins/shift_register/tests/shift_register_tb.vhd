library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity shift_register_tb is
    generic (runner_cfg : string);
end entity;

architecture behav of shift_register_tb is
    type test_config_t is record
        kernel_size : natural;
        num_channels : natural;
    end record;

  

    constant CONFIG : test_config_t := (
        kernel_size => 3,
        num_channels => 3
    );

  
    signal clk : std_logic := '1';
    signal rst : std_logic := '0';
    signal valid_in : std_logic := '0';
    signal start : std_logic := '0';
    
    signal d_in : std_logic_vector(CONFIG.num_channels - 1 downto 0) := (others => '0');
    signal d_out : std_logic_vector(CONFIG.kernel_size*CONFIG.num_channels - 1 downto 0) := (others => '0');
    signal valid_out : std_logic;
    
  
    constant clk_period : time := 4 ps;
begin


    clk <= not clk after clk_period / 2;

    dut: entity work.shift_register(rtl)
        generic map (
            DATA_WIDTH => CONFIG.num_channels,
            NUM_POINTS => CONFIG.kernel_size
        )
        port map (
            clk => clk,
            rst => rst,
            d_in => d_in,
            d_out => d_out,
            valid_in => valid_in,
            valid_out => valid_out
        );




    test_sequence: process is
        procedure can_shift_next_three_bits (
            constant input: std_logic_vector(2 downto 0);
            constant expected: std_logic_vector(8 downto 0)
        ) is 
        begin
            valid_in <= '1';
            d_in <= input;
            wait for clk_period;
            valid_in <= '0';
            wait for clk_period;
            check_equal(d_out, expected);
        end procedure;

        procedure reset_sr is begin
            valid_in <= '0';
            rst <= '1';
            wait until rising_edge(clk);
            rst <= '0';
            wait until rising_edge(clk);
        end procedure;
        
        procedure check_sr_is_full is begin
            check_equal(valid_out, '1', "expected shift reg to be full");
        end procedure;

        procedure check_sr_is_not_full is begin
            check_equal(valid_out, '0', "expected shift reg to be not full yet");
        end procedure;
    begin
        test_runner_setup(runner, runner_cfg);
        while test_suite loop
            if run("test_shift_register") then
                reset_sr;

                check_sr_is_not_full;
                can_shift_next_three_bits(b"000", b"000000000");
                check_sr_is_not_full;
                can_shift_next_three_bits(b"111", b"000000111");
                check_sr_is_not_full;
                can_shift_next_three_bits(b"011", b"000111011");
                check_sr_is_full;
                can_shift_next_three_bits(b"101", b"111011101");
                check_sr_is_full;
                can_shift_next_three_bits(b"010", b"011101010");
                check_sr_is_full;
            end if;
            if run("ensure reset is working") then
                reset_sr;
                valid_in <= '1';
                d_in <= b"111";
                wait for (CONFIG.kernel_size - 1)* clk_period;
                reset_sr;
                can_shift_next_three_bits(b"001", b"000000001");
            end if;
            if run("check valid out behaves correctly") then
                reset_sr;
                valid_in <= '1';
                for i in 0 to CONFIG.kernel_size  loop
                    check_sr_is_not_full;
                    wait for clk_period;
                end loop;
                check_sr_is_full;
            end if;
        end loop;
        test_runner_cleanup(runner);
    end process;



end architecture;
