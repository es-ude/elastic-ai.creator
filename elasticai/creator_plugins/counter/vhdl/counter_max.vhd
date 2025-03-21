library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.counter_pkg.all;


library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.counter_pkg.all;


entity counter_max is
    generic (
        MAX_VALUE : natural 
    );
    port (
        clk : in std_logic;
        rst : in std_logic; 
        enable : in std_logic := '0'; 
        d_out : out std_logic_vector(clog2(MAX_VALUE+1) - 1 downto 0) := (others => '0');
        max_value_f : out std_logic:= '0'
    );
end entity;

architecture rtl of counter_max is
    signal d_out_int : std_logic_vector(d_out'length-1 downto 0);
begin
    d_out <= d_out_int;

    process (d_out) is
    begin
        if d_out_int=std_logic_vector(to_unsigned(MAX_VALUE, d_out'length)) then
            max_value_f <= '1';
        else
            max_value_f <= '0';
        end if;
    end process;


    counter_i : entity work.counter
      generic map (
        MAX_VALUE => MAX_VALUE
      )
      port map (
        rst => rst,
        enable => enable,
        d_out => d_out_int,
        clk => clk
    );


end architecture;
