LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package lstm_common is
    function multiply_16_8(X1 : in signed(15 downto 0); X2 : in signed(15 downto 0)) return signed;
    function multiply_12_4(X1 : in signed(11 downto 0); X2 : in signed(11 downto 0)) return signed;
    function multiply_8_4(X1 : in signed(7 downto 0); X2 : in signed(7 downto 0)) return signed;
end package lstm_common;


package body lstm_common is

    function multiply_16_8(X1 : in signed(15 downto 0);
                      X2 : in signed(15 downto 0)) return signed is
        variable TEMP : signed(31 downto 0);
        variable TEMP2 : signed(15 downto 0) := (others=>'0');
        variable TEMP3 : signed(7 downto 0);

    begin
        TEMP := X1 * X2;

        TEMP2 := TEMP(23 downto 8);
        TEMP3 := TEMP(7 downto 0);
        if TEMP2(15) = '1' and TEMP3 /= 0 then
			TEMP2 := TEMP2 + 1;
		end if;

        if TEMP>0 and TEMP2<0 then
            TEMP2 := ('0', others => '1');
        elsif TEMP<0 and TEMP2>0 then
            TEMP2 := ('1', others => '0');
        end if;
        return TEMP2;
    end function;

    function multiply_8_4(X1 : in signed(7 downto 0);
                           X2 : in signed(7 downto 0)) return signed is
            variable TEMP : signed(15 downto 0);
            variable TEMP2 : signed(7 downto 0) := (others=>'0');
            variable TEMP3 : signed(3 downto 0);

        begin
            TEMP := X1 * X2;

            TEMP2 := TEMP(11 downto 4);
            TEMP3 := TEMP(3 downto 0);
            if TEMP2(7) = '1' and TEMP3 /= 0 then
                TEMP2 := TEMP2 + 1;
            end if;
            if TEMP>0 and TEMP2<0 then
                TEMP2 := ('0', others => '1');
            elsif TEMP<0 and TEMP2>0 then
                TEMP2 := ('1', others => '0');
            end if;
            return TEMP2;
        end function;

    function multiply_12_4(X1 : in signed(11 downto 0);
                                   X2 : in signed(11 downto 0)) return signed is
                    variable TEMP : signed(23 downto 0);
                    variable TEMP2 : signed(11 downto 0) := (others=>'0');
                    variable TEMP3 : signed(3 downto 0);

                begin
                    TEMP := X1 * X2;

                    TEMP2 := TEMP(15 downto 4);
                    TEMP3 := TEMP(3 downto 0);
                    if TEMP2(11) = '1' and TEMP3 /= 0 then
                        TEMP2 := TEMP2 + 1;
                    end if;
                    if TEMP>0 and TEMP2<0 then
                        TEMP2 := ('0', others => '1');
                    elsif TEMP<0 and TEMP2>0 then
                        TEMP2 := ('1', others => '0');
                    end if;
                    return TEMP2;
                end function;
end lstm_common;
