library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity shift_register is
      generic (
      DATA_WIDTH: positive;
      NUM_POINTS: positive;
      SKIP: positive := 1  -- only write d on every SKIPth rising edge,
                             -- ie., if
                             -- number_of_observed_rising_edges % SKIP = 0
                             -- we use this to implement a layer with stride >
                             -- 1
    );
    port (
      RST : in std_logic;
      CLK : in std_logic;
      D_IN : in std_logic_vector(DATA_WIDTH - 1 downto 0);
      D_OUT : out std_logic_vector(DATA_WIDTH * NUM_POINTS - 1 downto 0);
      SRC_VALID : in std_logic;
      VALID : out std_logic;
      DST_READY : in std_logic;
      READY : out std_logic := '1';
      EN : in std_logic := '0'
    );
end entity;

architecture rtl of shift_register is
    signal intern_valid_in : std_logic := '1';
    signal counter : integer range 0 to SKIP - 1 := 0;
  begin

    pick_stride_version: 
    if SKIP > 1 generate

    process(CLK)
    begin
      if rising_edge(CLK) then
        if rst = '1' then
          counter <= 0;
        elsif SRC_VALID = '1' then
          if counter = SKIP - 1 then
            counter <= 0;
          else
            counter <= counter + 1;
          end if;
        end if;
      end if;
    end process;

    intern_valid_in <= '1' when counter = 0 and SRC_VALID = '1'
                    else '0';
            


    reg_i : entity work.base_shift_register
      generic map (
        DATA_WIDTH => DATA_WIDTH,
        NUM_POINTS => NUM_POINTS
      )
      port map (
        CLK => clk,
        D_IN => d_in,
        RST => rst,
        D_OUT => d_out,
        SRC_VALID => intern_valid_in,
        VALID => VALID,
        READY => READY,
        DST_READY => DST_READY,
        EN => EN
      );

  end generate;

  pick_non_stride_version:
  if SKIP = 1 generate
            reg_i : entity work.base_shift_register
                generic map (
        DATA_WIDTH => DATA_WIDTH,
        NUM_POINTS => NUM_POINTS
      )
              
              port map (
                CLK => CLK,
                D_IN => D_IN,
                D_OUT => D_OUT,
                RST => RST,
                EN => EN,
                SRC_VALID => SRC_VALID,
                VALID => VALID,
                READY => READY,
                DST_READY => DST_READY
                
              );
            end generate;
end architecture;
