library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity ${name} is
generic (
    X_DATA_WIDTH : integer := ${x_data_width};
    Y_DATA_WIDTH : integer := ${y_data_width};
    THRESHOLD : integer := ${threshold};
    CLOCK_OPTION : boolean := ${clock_option}
);
port (
    enable : in std_logic;
	clock  : in std_logic;
	x  : in std_logic_vector(X_DATA_WIDTH - 1 downto 0);
	y : out std_logic_vector(Y_DATA_WIDTH - 1 downto 0)
);
end entity ${name};
architecture rtl of ${name} is
    signal signed_x : signed(X_DATA_WIDTH - 1 downto 0) := (others=>'0');
    signal signed_y : signed(Y_DATA_WIDTH - 1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    clocked: if CLOCK_OPTION generate
        main_process : process (enable, clock)
        begin
            if (enable = '0') then
                signed_y <= to_signed(THRESHOLD, Y_DATA_WIDTH);
            elsif (rising_edge(clock)) then
                if signed_x < THRESHOLD then
                    signed_y <= to_signed(THRESHOLD, Y_DATA_WIDTH);
                else
                    signed_y <= signed_x;
                end if;
            end if;
        end process;
    end generate;
    async: if (not CLOCK_OPTION) generate
        process (enable, signed_x)
        begin
            if enable = '0' then
                signed_y <= to_signed(THRESHOLD, Y_DATA_WIDTH);
            else
                if signed_x < THRESHOLD then
                    signed_y <= to_signed(THRESHOLD, Y_DATA_WIDTH);
                else
                    signed_y <= signed_x;
                end if;
            end if;
        end process;
    end generate;
end architecture;
