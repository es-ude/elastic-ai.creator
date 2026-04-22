library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        ONE_THRESHOLD : integer := ${one_threshold};
        MINUS_ONE_THRESHOLD : integer := ${minus_one_threshold}
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y      : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end entity ${name};
architecture rtl of ${name} is
    signal fxp_input : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal fxp_output : signed(DATA_WIDTH-1 downto 0) := (others=>'0');

begin
    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    main_process : process (enable, clock)
    begin
        if (enable = '0') then
            fxp_output <= to_signed(ONE_THRESHOLD, DATA_WIDTH);
        elsif (rising_edge(clock)) then
            if fxp_input <= to_signed(MINUS_ONE_THRESHOLD, DATA_WIDTH) then
                fxp_output <= to_signed(MINUS_ONE_THRESHOLD, DATA_WIDTH);
            elsif fxp_input >= to_signed(ONE_THRESHOLD, DATA_WIDTH) then
                fxp_output <= to_signed(ONE_THRESHOLD, DATA_WIDTH);
            else
                fxp_output <= fxp_input;
            end if;
        end if;
    end process;
end architecture rtl;
