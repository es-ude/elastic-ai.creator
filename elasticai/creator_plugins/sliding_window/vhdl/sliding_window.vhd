library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sliding_window is
    generic (
        INPUT_WIDTH : positive;
        OUTPUT_WIDTH : positive;
        STRIDE : positive := 1
);
    port (
        signal clk : in std_logic;
        signal d_in : in std_logic_vector(INPUT_WIDTH - 1 downto 0);
        signal d_out : out std_logic_vector(OUTPUT_WIDTH - 1 downto 0);
        signal valid_in : in std_logic;
        signal valid_out : out std_logic;
        signal rst : in std_logic
);
end entity;

architecture rtl of sliding_window is
  signal counter : integer := 0;
  signal intern_valid : std_logic := '1';
begin
  d_out <= d_in(INPUT_WIDTH - STRIDE * counter - 1 downto INPUT_WIDTH - STRIDE * counter - OUTPUT_WIDTH);
  valid_out <= valid_in and intern_valid;
  process (clk) is
      begin
          if rising_edge(clk) then
            if rst = '1' then
              counter <= 0;
              intern_valid <= '1';
            end if;
            if valid_in = '1' then
              if INPUT_WIDTH - STRIDE * (counter) - OUTPUT_WIDTH > 0 then
                counter <= counter + 1;
                intern_valid <= '1';
            else
                intern_valid <= '0';
            end if;
          end if;
      end if;
  end process;
end architecture;
