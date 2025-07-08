library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

package counter_pkg is
  function clog2(n : natural) return natural;
 end package;
 
 package body counter_pkg is
 
    function clog2(n : natural) return natural is
    begin
        return natural(ceil(log2(real(n))));
    end function;
    
 end package body;
    

library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.counter_pkg.all;


entity counter is
    generic (
        MAX_VALUE : natural 
    );
    port (
        clk : in std_logic;
        rst : in std_logic; 
        enable : in std_logic := '0'; 
        d_out : out std_logic_vector(clog2(MAX_VALUE+1) - 1 downto 0) := (others => '0')
    );
end entity;

architecture rtl of counter is

    signal count_r : unsigned(d_out'range) := to_unsigned(0, d_out'length);

begin
    d_out <= std_logic_vector(count_r);


    process (clk, rst) is
    begin
        if (rst = '1') then
            count_r <= to_unsigned(0, count_r'length);
        elsif (rising_edge(clk)) then
            if (enable = '1') then
                if (count_r = MAX_VALUE) then
                    count_r <= to_unsigned(0, count_r'length);
                else
                    count_r <= count_r + 1;
                end if;
            end if;
        end if;
    
    end process;    


end architecture;
