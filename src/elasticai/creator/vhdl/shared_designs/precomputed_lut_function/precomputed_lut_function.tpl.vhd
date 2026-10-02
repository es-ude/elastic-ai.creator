library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ${name} is
    generic (
        BITWIDTH_INPUT : integer := ${input_data_width};
        BITWIDTH_OUTPUT : integer := ${output_data_width};
        PIPELINE_ENABLE : boolean := false
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(BITWIDTH_INPUT-1 downto 0);
        y      : out std_logic_vector(BITWIDTH_OUTPUT-1 downto 0)
    );
end ${name};

architecture rtl of ${name} is
    signal signed_x : signed(BITWIDTH_INPUT-1 downto 0) := (others=>'0');
    signal signed_y : signed(BITWIDTH_OUTPUT-1 downto 0) := (others=>'0');

    function compute(sx : signed; en : std_logic) return signed is
    begin
        if en = '0' then
            return to_signed(0, BITWIDTH_OUTPUT);
        ${process_content}
        end if;
    end function;
begin
    signed_x <= signed(x);
    y        <= std_logic_vector(signed_y);

    comb_gen : if not PIPELINE_ENABLE generate
        precomputed_process : process(signed_x, enable)
        begin
            signed_y <= compute(signed_x, enable);
        end process;
    end generate;

    pipe_gen : if PIPELINE_ENABLE generate
        precomputed_process : process(clock)
        begin
            if rising_edge(clock) then
                signed_y <= compute(signed_x, enable);
            end if;
        end process;
    end generate;
end rtl;
