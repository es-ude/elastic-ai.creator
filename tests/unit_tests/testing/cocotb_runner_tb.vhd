library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity COCOTB_TEST is
    generic (
        BITWIDTH : integer := 12;
        DEFINE_TEST : boolean := false;
        OFFSET : integer := 4;
        SCALE : integer := 1
    );
    port (
        A : in  std_logic_vector(BITWIDTH - 1 downto 0);
        Q : out std_logic_vector(BITWIDTH - 1 downto 0)
    );
end entity;

architecture Behavioral of COCOTB_TEST is
begin
    process(A)
        variable A_int : integer;
        variable Q_int : integer;
    begin
        A_int := to_integer(unsigned(A));

        if DEFINE_TEST then
            Q_int := SCALE * A_int + OFFSET;
        else
            Q_int := SCALE * A_int;
        end if;

        Q <= std_logic_vector(to_unsigned(Q_int, BITWIDTH));
    end process;
end architecture;
