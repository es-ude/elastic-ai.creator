library IEEE;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity $name is
    port (
        reset : in std_logic;
        next_sample : in std_logic;
        x1 : in std_logic_vector($total_width-1 downto 0);
        x2 : in std_logic_vector($total_width-1 downto 0);
        sum : out std_logic := '0';
        done : out std_logic := '0'
    );
end;

architecture rtl of $name is
    function popcount(x: std_logic_vector($total_width-1 downto 0)) return integer is
        variable num_ones : natural := 0;
        variable result : integer range -${total_width} to ${total_width} := 0;
    begin
        for i in x'range loop
          if x(i) = '1' then
              num_ones := num_ones + 1;
          end if;
        end loop;
        result := 2*num_ones - $total_width;
        return result;
    end function;

    function is_positive(x: integer) return std_logic is
        variable result : std_logic := '1';
    begin
        if x < 0 then
            result := '0';
        end if;
        return result;
    end function;
begin
    sum <= is_positive(popcount(x1 xnor x2));
end rtl;
