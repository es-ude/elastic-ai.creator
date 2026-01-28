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
        signal CLK : in std_logic;
        signal D_IN : in std_logic_vector(INPUT_WIDTH - 1 downto 0);
        signal D_OUT : out std_logic_vector(OUTPUT_WIDTH - 1 downto 0);
        signal SRC_VALID : in std_logic;
        signal VALID : out std_logic;
        signal RST : in std_logic;
        signal EN : in std_logic;
        signal READY: out std_logic := '1';
        signal DST_READY : in std_logic := '1'
);
end entity;

architecture rtl of sliding_window is
  signal counter : integer := 0;
  signal valid_reg : std_logic := '0';
begin
  D_OUT <= D_IN(INPUT_WIDTH - STRIDE * counter - 1 downto INPUT_WIDTH - STRIDE * counter - OUTPUT_WIDTH);
  VALID <= valid_reg;
  READY <= DST_READY;

  process (clk) is
      begin
          if rising_edge(clk) then
            if rst = '1' then
              counter <= 0;
              valid_reg <= '0';
            else 
              if SRC_VALID = '1' and DST_READY = '1' then
                -- first iteration, don't advance counter yet
                if valid_reg = '0' and counter = 0 then  
                  valid_reg <= '1';
                elsif INPUT_WIDTH - STRIDE * (counter) - OUTPUT_WIDTH > 0 then
                  counter <= counter + 1;
                  valid_reg <= '1';
                else
                  valid_reg <= '0';
              end if;
            end if;
          end if;
      end if;
  end process;
end architecture;
