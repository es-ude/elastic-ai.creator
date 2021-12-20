-- The sync version of the gate logic in lstm
-- input W[idx], X[idx], and b
-- output y = W[0]*X[0] + W[1]*X[1] + W[2]*X[2] + ... + W[vector_len-1]*X[vector_len-1] + b
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions


-- behavior: y=W*X + b
entity lstm_common_gate is
    generic (
            DATA_WIDTH : integer := 16;
            FRAC_WIDTH : integer := 8;
            VECTOR_LEN_WIDTH : integer := 8 -- supports at most 255 elements of vectors
        );
    port (
        reset : in std_logic;
        clk   : in std_logic;
        x     : in signed(DATA_WIDTH-1 downto 0);
        w     : in signed(DATA_WIDTH-1 downto 0);
        b     : in signed(DATA_WIDTH-1 downto 0);
        vector_len : in unsigned(VECTOR_LEN_WIDTH-1 downto 0);
        idx   : out unsigned(VECTOR_LEN_WIDTH-1 downto 0);
        ready : out std_logic;
        y     : out signed(DATA_WIDTH-1 downto 0)
    );
end lstm_common_gate;

-- The RTL implementations
architecture lstm_common_gate_rtl of lstm_common_gate is
    signal sum_s : signed(DATA_WIDTH*2-1 downto 0);
begin
    process(reset, clk)
    variable val_idx : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0');
    variable sum : signed(DATA_WIDTH*2-1 downto 0):=(others=>'0');
    variable val_ready : std_logic;
    variable val_y : signed(DATA_WIDTH-1 downto 0);
    begin
        if reset = '1' then
            val_idx := to_unsigned(0, VECTOR_LEN_WIDTH);
            val_ready := '0';
            sum := to_signed(0, DATA_WIDTH*2);
        elsif rising_edge(clk) then
            sum := sum + w * x;
            if val_ready = '0' then
                val_idx := val_idx + 1;
                if val_idx = vector_len then
                    val_y := b + shift_right(sum, FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
                    val_ready := '1';
                    val_idx := val_idx-1;
                end if;
            end if;
        end if;
        idx <= val_idx;
        ready <= val_ready;
        y <= val_y;
        sum_s <= sum;
    end process;
end lstm_common_gate_rtl;