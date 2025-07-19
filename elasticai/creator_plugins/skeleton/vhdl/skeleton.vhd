library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity skeleton is
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

architecture rtl of skeleton is
    constant ENABLE_ADDRESS : natural := 16;
    constant RESERVED_ADDRESS : natural := 17;
    constant DATA_IN_SEGMENT_START : natural := 18;
    constant DATA_OUT_SEGMENT_START : natural := DATA_IN_SEGMENT_START;
    constant skeleton_id_str : skeleton_id_t := SKELETON_ID;
    signal network_address : std_logic_vector(15 downto 0);
    signal network_address_i : integer range 0 to 2**16 - 1;
    signal network_enable : std_logic := '0';
    signal network_d_out : STD_LOGIC_VECTOR(8 - 1 downto 0);
    signal done :  std_logic;
    signal address_in_i : integer range 0 to 2**16 - 1 := 0;
begin

    address_in_i <= TO_INTEGER(UNSIGNED(address_in));
    network_address <= std_logic_vector(to_unsigned(network_address_i, network_address'length));
    


    update_network_address:
    process (address_in_i, rd, wr) is
    begin
        if rd = '1' and wr = '0' then
            if address_in_i >= DATA_OUT_SEGMENT_START then
                network_address_i <= address_in_i - DATA_OUT_SEGMENT_START;
            else
                network_address_i <= 0;
            end if;
        elsif wr = '1' and rd = '0' then
            if address_in_i >= DATA_IN_SEGMENT_START then
                network_address_i <= address_in_i - DATA_IN_SEGMENT_START;
            else
                network_address_i <= 0;
            end if;
        else
            network_address_i <= 0;
        end if;

    end process;

    update_enable:
    process (clock) is
    begin
        if rising_edge(clock) then
            if address_in_i = ENABLE_ADDRESS then
                network_enable <= data_in(0);
            end if;
        end if;
    end process;
    
    update_data_out:
    process (clock) is
    begin
        if rising_edge(clock) then
            if address_in_i >= DATA_OUT_SEGMENT_START then
                data_out <= network_d_out;
            elsif address_in_i < ENABLE_ADDRESS then
                data_out <= skeleton_id_str(address_in_i);
            else
                data_out <= (others => 'X');
            end if;
        end if;
    end process;

  


    network : entity work.buffered_network_wrapper(rtl)
        port map(
            d_in => data_in,
            d_out => network_d_out,
            address => network_address,
            done => done,
            wr => wr,
            enable => network_enable,
            rst => reset,
            clk => clock
        );

    busy <= not done;
end rtl;
