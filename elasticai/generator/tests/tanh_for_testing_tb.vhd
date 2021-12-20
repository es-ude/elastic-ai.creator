library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
use ieee.math_real.all;                 -- for the ceiling and log constant calculation function

entity tanh_tb is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8
        );
    port ( clk: out std_logic);
end entity ; -- tanh_tb

architecture arch of tanh_tb is

    component tanh is
        generic (
                DATA_WIDTH : integer := 16;
                FRAC_WIDTH : integer := 8
            );
        port (
            x : in signed(DATA_WIDTH-1 downto 0);
            y : out signed(DATA_WIDTH-1 downto 0)
        );
    end component tanh;

    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------
    signal clk_period : time := 1 ns;
    signal test_input : signed(16-1 downto 0):=(others=>'0');
    signal test_output : signed(16-1 downto 0);

begin

    clock_process : process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process; -- clock_process

    uut: tanh
    port map (
    x => test_input,
    y => test_output
    );

    test_process: process is
    begin
        Report "======Simulation start======" severity Note;

        test_input <=  to_signed(-1281,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output="1111111100000000" report "The test case -1281 fail" severity failure;

        test_input <=  to_signed(-1000,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output=-255 report "The test case -1000 fail" severity failure;

        test_input <=  to_signed(-500,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output=-246 report "The test case -500 fail" severity failure;

        test_input <=  to_signed(0,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output=0 report "The test case 0 fail" severity failure;

        test_input <=  to_signed(500,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output=245 report "The test case 500 fail" severity failure;

        test_input <=  to_signed(800,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output=254 report "The test case 800 fail" severity failure;

        test_input <=  to_signed(1024,16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output=255 report "The test case 1024 fail" severity failure;


        -- if there is no error message, that means all test case are passed.
        report "======Simulation Success======" severity Note;
        report "Please check the output message." severity Note;

        -- wait forever
        wait;

    end process; -- test_process

end arch ; -- arch
