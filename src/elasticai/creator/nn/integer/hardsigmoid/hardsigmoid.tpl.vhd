library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        THREE_THRESHOLD : integer := ${three_threshold};
        MINUS_THREE_THRESHOLD : integer := ${minus_three_threshold};
        ZERO_OUTPUT : integer := ${zero_output};
        ONE_OUTPUT : integer := ${one_output};
        TMP_THRESHOLD : integer := ${tmp_threshold}
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

    constant fxp_temp : signed(2*DATA_WIDTH-1 downto 0) := to_signed(TMP_THRESHOLD, 2*DATA_WIDTH);

    -----------------------------------------------------------
    -- functions
    -----------------------------------------------------------
    function sum_div_8_op(a : in signed(DATA_WIDTH-1 downto 0);
                    b : in signed(2*DATA_WIDTH-1 downto 0)
            ) return signed is
        variable TEMP : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMPB : signed(DATA_WIDTH*2-1 downto 0) := (others=>'0');
        variable TEMP1 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP2 : signed(DATA_WIDTH*2-1 downto 0) := (others=>'0');
        variable is_negative : boolean;
    begin

        is_negative := (a(a'left) = '1');

        if is_negative then
            TEMP := -a;
            TEMPB := -b;
        else
            TEMP := a;
            TEMPB := b;
        end if;

        TEMP1 := shift_right(TEMP, 3); -- divide by 8

        TEMP2 := TEMP1 + TEMPB;
        if is_negative then
            return -resize(TEMP2, DATA_WIDTH);
        else
            return resize(TEMP2, DATA_WIDTH);
        end if;
    end function;

begin
    fxp_input <= signed(x);
    y <= std_logic_vector(fxp_output);

    main_process : process (enable, clock)
    begin
        if (enable = '0') then
            fxp_output <= to_signed(ZERO_OUTPUT, DATA_WIDTH);
        elsif (rising_edge(clock)) then
            if fxp_input <= to_signed(MINUS_THREE_THRESHOLD, DATA_WIDTH) then
                fxp_output <= to_signed(ZERO_OUTPUT, DATA_WIDTH);
            elsif fxp_input >= to_signed(THREE_THRESHOLD, DATA_WIDTH) then
                fxp_output <= to_signed(ONE_OUTPUT, DATA_WIDTH);
            else
                fxp_output <= sum_div_8_op(fxp_input, fxp_temp);
            end if;
        end if;
    end process;
end architecture rtl;
