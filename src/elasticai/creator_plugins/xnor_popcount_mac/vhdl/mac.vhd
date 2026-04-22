library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mac is
    generic (
      KERNEL_SIZE: positive;
      NUM_CHANNELS: positive
    );
    port (
      RST : in std_logic;
      CLK : in std_logic;
      D_IN : in std_logic_vector(KERNEL_SIZE - 1 downto 0);
      D_OUT : out std_logic_vector(NUM_CHANNELS -1 downto 0);
      SRC_VALID : in std_logic;
      VALID : out std_logic := '0';
      DST_READY : in std_logic;
      WEIGHT: in std_logic_vector(NUM_CHANNELS*KERNEL_SIZE - 1 downto 0);
      READY : out std_logic := '1';
      EN : in std_logic := '0'
    );
end entity;

architecture rtl of mac is
   type weight_t is array(0 to NUM_CHANNELS - 1) of std_logic_vector(KERNEL_SIZE -1 downto 0);
   type counters_t is array(0 to NUM_CHANNELS - 1) of integer range 0 to KERNEL_SIZE + 1;
   signal xnored_vector : weight_t;
   signal valid_reg : std_logic := '0';
    signal xnor_valid_reg : std_logic := '0';
begin

    -- Handshake contract: input is accepted on rising edge when EN, SRC_VALID and READY are high.
    -- READY is a passthrough of DST_READY, and VALID is asserted one cycle later
    -- (xnor is registered first, then popcount/sign is computed from that register).
    READY <= DST_READY;

    VALID <= valid_reg;

    process (CLK) is 
        variable counters : counters_t;
        variable counter : integer range 0 to KERNEL_SIZE;
    begin
        if rising_edge(CLK) then
            valid_reg <= '0';
            if RST = '1' then
                valid_reg <= '0';
                xnor_valid_reg <= '0';
            else
                -- Stage 1: emit output from previously captured xnor result.
                if xnor_valid_reg = '1' and DST_READY = '1' then
                    valid_reg <= '1';
                    xnor_valid_reg <= '0';
                    for j in counters'range loop
                        counter := 0;
                        for i in D_IN'range loop
                            if xnored_vector(j)(i) = '1' then
                                counter := counter + 1;
                            end if;
                        end loop;
                        counters(j) := counter;
                        if 2*counters(j) - D_IN'length >= 0 then
                            D_OUT(j) <= '1';
                        else
                            D_OUT(j) <= '0';
                        end if;
                    end loop;
                end if;

                -- Stage 0: accept new input and capture xnor for next cycle.
                -- This is a single-entry pipeline stage (no queueing).
                if EN = '1' and SRC_VALID = '1' and DST_READY = '1' then
                    xnor_valid_reg <= '1';
                    for i in xnored_vector'range loop
                        xnored_vector(i) <= WEIGHT((i+1)*KERNEL_SIZE - 1 downto i*KERNEL_SIZE) xnor D_IN;
                    end loop;
                end if;
            end if;
        end if;
    end process;
end architecture;
