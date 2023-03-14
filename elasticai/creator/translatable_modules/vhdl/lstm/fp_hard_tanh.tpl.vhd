-- A LUT version of tanh
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity tanh is
    port (
        x : in signed({data_width}-1 downto 0);
        y : out signed({data_width}-1 downto 0)
    );

end tanh;

architecture rtl of tanh is
begin

    tanh_process:process(x)
    begin
    if x<=-16 then
        y <= to_signed({minus_one}, y'length);
    elsif x<16 then
        y <= x;
    else
        y <= to_signed({one}, y'length);
    end if;
    end process;
end rtl;
