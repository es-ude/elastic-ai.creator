library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity $name is
    port (
        reset : in std_logic;
        next_sample  : in std_logic;
        x1   : in signed(${total_width}-1 downto 0);
        x2 : in signed(${total_width}-1 downto 0);
        sum : out signed(${total_width}-1 downto 0) := (others => '0');
        done   : out std_logic := '0'
    );
end $name;

architecture rtl of $name is
    constant TOTAL_WIDTH : natural := ${total_width};
    constant FRAC_WIDTH : natural := ${frac_width};
    constant VECTOR_WIDTH : natural := ${vector_width};
begin
    ${name}_fxp_MAC : entity work.fxp_MAC_RoundToZero
        generic map(
            VECTOR_WIDTH => VECTOR_WIDTH,
            TOTAL_WIDTH=>TOTAL_WIDTH,
            FRAC_WIDTH => FRAC_WIDTH
        )
        port map (
            reset => reset,
            next_sample => next_sample,
            x1 => x1,
            x2 => x2,
            sum => sum,
            done => done
        );

end architecture;
