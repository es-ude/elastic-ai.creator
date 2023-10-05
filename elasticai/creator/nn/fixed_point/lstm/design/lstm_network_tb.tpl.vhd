----------------------------------------------------------------------------------
-- Company:
-- Engineer:
--
-- Create Date:
-- Design Name:
-- Module Name:
-- Project Name:
-- Target Devices:
-- Tool Versions:
-- Description:
--
-- Dependencies:
--
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
--
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

library work;

entity $name is
    port ( clk: out std_logic);
end;

architecture Behavioral of $name is


    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------
    signal clk_period : time := 2 ps;
    signal clock : std_logic;
    signal enable : std_logic;

    signal reset: std_logic:='0';

    signal busy :  std_logic:='0';
    signal rd :  std_logic:='0';
    signal wr :  std_logic:='0';

    signal data_in :  std_logic_vector(8-1 downto 0);
    signal address_in :  std_logic_vector(4-1 downto 0);

    signal debug :  std_logic_vector(7 downto 0);
    signal lstm_network_enable : std_logic;
    signal result : std_logic_vector(7 downto 0);

    -- ToDo: pack this procedure to a package, such as Simulation pack.
    procedure send_data (
        addr_in : in std_logic_vector(4-1 downto 0);
        data_in : in std_logic_vector(8-1 downto 0);
        signal clock   : in std_logic;
        signal wr       : out std_logic;
        signal addr_out : out std_logic_vector(4-1 downto 0);
        signal data_out : out std_logic_vector(8-1 downto 0)) is
        begin
            addr_out <= addr_in;
            data_out <= data_in;
            wait for clk_period;
            wr <= '0';
            wait for clk_period*2;
            wr <= '1';
            wait for 1*clk_period;
    end send_data;

    type Data_ARRAY is array (0 to 31) of signed(8-1 downto 0);
   signal test_data: Data_ARRAY := (x"00",x"00",x"00",
                                    x"00",x"00",x"10",
                                    x"00",x"10",x"00",
                                    x"00",x"10",x"10",
                                    x"10",x"00",x"00",
                                    x"10",x"00",x"10",
                                    x"10",x"10",x"00",
                                    x"10",x"10",x"10", others=>X"00");
begin
    clock_process : process
    begin
        clock <= '0';
        wait for clk_period/2;
        clock <= '1';
        wait for clk_period/2;
    end process; -- clock_process

    clk <= clock;

    uut: entity work.${uut_name}(rtl)
    port map (
        clock => clock,
        enable => ${uut_name}_enable,
        x => data_in,
        addr_in => address_in,
        x_we => wr,
        done => busy,
        d_out => result
    );

    test_process: process

        begin
            Report "======Tests start======" severity Note;
            -- setting input x buffer --

            for round_idx in 0 to 7 loop
                reset <= '1';
                ${uut_name}_enable <= '0';

                wr<='0';
                wait for 2*clk_period;
                reset <= '0';

                for ii in 0 to 3 loop
                    send_data(std_logic_vector(to_signed(ii, 4)), std_logic_vector(test_data(ii+round_idx*3)), clock, wr, address_in, data_in);
                    wait for 10 ns;
                end loop;

                ${uut_name}_enable <= '1';
                wait for 2*clk_period;
                wait until busy = '1';

                wait for 20*clk_period;
                if signed(result)>=0 then
                    report "output is 1.0" severity Note;
                else
                    report "output is 0.0" severity Note;
                end if;


                wait for 2*clk_period;

            end loop;

            -- if there is no error message, that means all test case are passed.
            report "======Tests finished======" severity Note;
            report "Please check the output message." severity Note;

            -- wait forever
            wait;

    end process; -- test_process

end Behavioral;
