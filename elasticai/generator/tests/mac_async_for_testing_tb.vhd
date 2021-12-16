library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
use ieee.math_real.all;                 -- for the ceiling and log constant calculation function

entity mac_async_tb is
    port ( clk: out std_logic);
end entity ; -- mac_async_tb

architecture arch of mac_async_tb is

    component mac_async is
        generic (
                DATA_WIDTH : integer := 16;
                FRAC_WIDTH : integer := 8
            );
        port (
            x1 : in signed(DATA_WIDTH-1 downto 0);
            x2 : in signed(DATA_WIDTH-1 downto 0);
            w1 : in signed(DATA_WIDTH-1 downto 0);
            w2 : in signed(DATA_WIDTH-1 downto 0);
            b : in signed(DATA_WIDTH-1 downto 0);
            y : out signed(DATA_WIDTH-1 downto 0)
        );
    end component mac_async;

    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------
    signal test_X : signed(16-1 downto 0);
    signal test_h_in : signed(16-1 downto 0);
    signal test_W0 : signed(16-1 downto 0);
    signal test_W1 : signed(16-1 downto 0);
    signal test_b : signed(16-1 downto 0);
    signal test_mac_out : signed(16-1 downto 0);
    signal clk_period : time := 1 ns;
    signal product_1, product_2 : signed(16-1 downto 0);

begin

    clock_process : process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process; -- clock_process

    uut: mac_async
    port map (
    x1 => test_X,
    x2 => test_h_in,
    w1 => test_W0,
    w2 => test_W1,
    b => test_b,
    y => test_mac_out
    );

    test_process: process is
    begin
        Report "======Simulation start======" severity Note;

        test_X <=  to_signed(0,16);
        test_h_in <=  to_signed(0,16);
        test_W0 <=  to_signed(0,16);
        test_W1 <=  to_signed(0,16);
        test_b <=  to_signed(255,16);
        wait for 1*clk_period;
        report "The value of 'test_mac_out' is " & integer'image(to_integer(unsigned(test_mac_out)));
        assert test_mac_out = 255 report "The 1. test case fail" severity error;

        test_X <=  to_signed(256,16);
        test_h_in <= to_signed(1024,16);
        test_W0 <= to_signed(16,16);
        test_W1 <= to_signed(16,16);
        test_b <= to_signed(3,16);
        wait for 1*clk_period;
        -- report "The value of 'test_mac_out' is " & integer'image(to_integer(unsigned(test_mac_out)));
        assert test_mac_out = 83 report "The 2rd test case fail" severity error;

        -- if there is no error message, that means all test case are passed.
        report "======Simulation Success======" severity Note;
        report "Please check the output message." severity Note;

        -- wait forever
        wait;

    end process; -- test_process

end architecture ; -- arch