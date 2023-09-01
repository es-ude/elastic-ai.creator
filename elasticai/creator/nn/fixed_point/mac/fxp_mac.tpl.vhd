library IEEE;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fxp_MAC_RoundToEven is
    generic (
        VECTOR_WIDTH : integer;
        TOTAL_WIDTH   : integer;
        FRAC_WIDTH   : integer
    );
    port (
        reset : in std_logic;
        next_sample  : in std_logic;
        x1   : in signed(TOTAL_WIDTH-1 downto 0);
        x2 : in signed(TOTAL_WIDTH-1 downto 0);
        sum : out signed(TOTAL_WIDTH-1 downto 0);
        done   : out std_logic
    );
end fxp_MAC_RoundToEven;

architecture rtl of fxp_MAC_RoundToEven is

    function cut_down(x: in signed(2*TOTAL_WIDTH-1 downto 0))return signed is
        variable result : signed(TOTAL_WIDTH-1 downto 0) := (others=>'0');
        variable underflow_part : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin
        -- get result-range of x and underflow part
        result := x(TOTAL_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        underflow_part := x(FRAC_WIDTH-1 downto 0);

        -- round half to even
        if result(0) = '1' and underflow_part(FRAC_WIDTH-1) = '1' then
            if result(TOTAL_WIDTH-1) = '0' then
                result := result + 1;
            else
                result := result - 1;
            end if;
        end if;

        -- check if overflow occured
        if x>0 and result<0 then
            result := ('0', others => '1');
        elsif x<0 and result>0 then
            result := ('1', others => '0');
        end if;
        return result;
    end function;

begin
    mac : process (reset, next_sample)
        variable accumulator : signed(2*TOTAL_WIDTH-1 downto 0) := (others=>'0');
        variable vector_idx : integer range 0 to VECTOR_WIDTH := 0;
        type t_state is (s_compute, s_finished);
        variable state : t_state;
    begin
        if (reset='1') then
            accumulator := (others=>'0');
            done <= '0';
            state := s_compute;
            vector_idx := 0;
        elsif rising_edge(next_sample) then
            if state=s_compute then
                accumulator := x1*x2 + accumulator;
                vector_idx := vector_idx + 1;
                if vector_idx = VECTOR_WIDTH then
                   sum <= cut_down(accumulator);
                   done <= '1';
                   state := s_finished;
                end if;
            end if;
        end if;
    end process mac;
end rtl;
