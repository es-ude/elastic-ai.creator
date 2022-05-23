library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity $entity_name is
    port (
        input: in std_logic_vector($input_vector_start_bit downto 0);
        output: out std_logic_vector($output_vector_start_bit downto 0)
    );
end entity $entity_name;
architecture $architecture_name of $entity_name is begin
    process (input) is begin
        case input is
            $cases
        end case;
    end process;
end rtl;
