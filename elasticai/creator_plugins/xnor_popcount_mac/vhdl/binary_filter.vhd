library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity binary_filter is
    generic (
        KERNEL_SIZE : positive;
        NUM_OUT_CHANNELS : positive;
        PARALLEL_INSTANCES : positive := 1
    );
    port (
        CLK : in std_logic;
        RST : in std_logic;
        D_IN : in std_logic_vector(KERNEL_SIZE * PARALLEL_INSTANCES - 1 downto 0);
        D_OUT : out std_logic_vector(NUM_OUT_CHANNELS * PARALLEL_INSTANCES - 1 downto 0);
        SRC_VALID : in std_logic;
        VALID : out std_logic;
        DST_READY : in std_logic;
        READY : out std_logic;
        EN : in std_logic
    );

end entity;

architecture rtl of binary_filter is
    constant weight : std_logic_vector(KERNEL_SIZE * NUM_OUT_CHANNELS - 1 downto 0) := (others => '0');
    signal valids : std_logic_vector(PARALLEL_INSTANCES - 1 downto 0);
    signal readys : std_logic_vector(PARALLEL_INSTANCES - 1 downto 0);

    function and_over_logic_vector(input : std_logic_vector) return std_logic is
        variable tmp : std_logic := '1';
        begin
            for i in input'range loop
                tmp := tmp and input(i);
            end loop;
        return tmp;
    end function;
begin

    VALID <= and_over_logic_vector(valids);
    READY <= and_over_logic_vector(readys);

    parallel_filters:
    for i in 0 to PARALLEL_INSTANCES - 1 generate
        filter : entity work.mac(rtl) 
            generic map (
                KERNEL_SIZE => KERNEL_SIZE,
                NUM_CHANNELS => NUM_OUT_CHANNELS
            )
            port map (
                CLK => CLK,
                RST => RST,
                SRC_VALID => SRC_VALID,
                DST_READY => DST_READY,
                EN => EN,
                WEIGHT => weight,
                D_IN => D_IN((i+1)*KERNEL_SIZE- 1 downto i*KERNEL_SIZE),
                D_OUT => D_OUT((i+1)*NUM_OUT_CHANNELS - 1 downto i*NUM_OUT_CHANNELS),
                VALID => valids(i),
                READY => readys(i)
            );
    end generate;

end architecture;

