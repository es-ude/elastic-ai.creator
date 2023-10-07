def test_lut():
    expected = """library ieee;
    use ieee.std_logic_1164.all;

    entity my_lut is
        port (
            enable : in std_logic;
            clock : in std_logic;
            x : in std_logic_vector(3-1 downto 0);
            y : out std_logic_vector(1-1 downto 0);
        );
    end;

    architecture rtl of my_lut is
    begin
        process (x)
        begin
            case x is
                when '000' => y <= '0';
                when '001' => y <= '0';
                when '010' => y <= '0';
                when '011' => y <= '0';
                when '100' => y <= '1';
                when '101' => y <= '1';
                when '110' => y <= '1';
                when '111' => y <= '1';
            end case;
        end process;
    end rtl;"""
