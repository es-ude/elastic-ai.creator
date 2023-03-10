library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity $name is
    port (
        x : in signed($data_width downto 0);
        y : out signed($data_width downto 0)
    );

end sigmoid;

architecture rtl of $name is
begin
    $name_process : process(x)
    begin
    $process_content
    end process;
end rtl;
