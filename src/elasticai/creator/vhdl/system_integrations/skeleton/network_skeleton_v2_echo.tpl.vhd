library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity $name is
    port (
        -- control interface
        clock                : in std_logic;
        clk_hadamard                : in std_logic;
        reset                : in std_logic; -- controls functionality (sleep)
        busy                : out std_logic; -- done with entire calculation
        wake_up             : out std_logic;
        -- indicate new data or request
        rd                    : in std_logic;    -- request a variable
        wr                 : in std_logic;     -- request changing a variable

        -- data interface
        data_in            : in std_logic_vector(7 downto 0);
        address_in        : in std_logic_vector(15 downto 0);
        data_out            : out std_logic_vector(7 downto 0);

        debug                : out std_logic_vector(7 downto 0);

        led_ctrl             : out std_logic_vector(3 DOWNTO 0)
    );
end;

architecture rtl of $name is
    constant DATA_WIDTH_IN : integer := ${data_width};
    constant NUM_VALUES : integer := ${num_values};
    signal network_enable :  std_logic;

    signal wake_up_write: std_logic := '0';
    signal wake_up_read : std_logic := '0';

    type buf_data_in_t is array (0 to NUM_VALUES) of std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal data_buf_in : buf_data_in_t;
    type skeleton_id_data_t is array (0 to 15) of std_logic_vector(7 downto 0);
    signal skeleton_id_str : skeleton_id_data_t := (${id});
begin

    busy <= '0';
    wake_up <= wake_up_write;
    wake_up <= wake_up_read;

    receive_data_from_middleware: process (clock)
    variable int_addr : integer range 0 to 18 + NUM_VALUES;
    begin
        if rising_edge(clock) then
            if reset = '1' then
                network_enable <= '0';
            else
                if wr = '1' then
                    int_addr := to_integer(unsigned(address_in));
                    if int_addr = 16 then
                        network_enable <= data_in(0);
                        wake_up_write <= '1';
                    elsif int_addr >= 18 and int_addr < 18 + NUM_VALUES then
                        data_buf_in(int_addr-18) <= data_in(DATA_WIDTH_IN-1 downto 0);
                    end if;
                end if;
            end if;
        end if;
    end process;

    sendback_data_to_middleware: process  (clock)
    variable int_addr : integer range 0 to 18 + NUM_VALUES;
    begin
        if rising_edge(clock) then
            if rd = '1' then
                int_addr := to_integer(unsigned(address_in));
                if int_addr <= 15 then
                    data_out(7 downto 0) <= skeleton_id_str(int_addr);
                elsif int_addr >= 18 and int_addr < 18 + NUM_VALUES then
                    data_out(7 downto 0) <= std_logic_vector(signed(data_buf_in(int_addr-18))+1);
                    wake_up_read <= '0';
                end if;
            end if;
        end if;
    end process;
end rtl;
