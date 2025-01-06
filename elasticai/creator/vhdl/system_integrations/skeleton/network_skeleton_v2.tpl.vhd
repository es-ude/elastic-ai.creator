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
    constant DATA_WIDTH_IN : integer := ${data_width_in};
    constant DATA_WIDTH_OUT : integer := ${data_width_out};
    constant X_ADDR_WIDTH : integer := ${x_addr_width};
    constant X_NUM_VALUES : integer := ${x_num_values};
    constant Y_ADDR_WIDTH : integer:= ${y_addr_width};
    constant Y_NUM_VALUES : integer := ${y_num_values};

    signal network_enable :  std_logic;

    signal c_config_en :  std_logic;
    signal done :  std_logic;

    signal x :  std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal y : std_logic_vector(DATA_WIDTH_OUT-1 downto 0);
    signal x_address :  std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal y_address :  std_logic_vector(Y_ADDR_WIDTH-1 downto 0);

    type buf_data_in_t is array (0 to X_NUM_VALUES) of std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal data_buf_in : buf_data_in_t;
    type skeleton_id_data_t is array (0 to 15) of std_logic_vector(7 downto 0);
    signal skeleton_id_str : skeleton_id_data_t := (${id});

    function pad_output_to_middleware(network_out : std_logic_vector(DATA_WIDTH_OUT-1 downto 0)) return std_logic_vector is
    variable k : std_logic_vector(7 downto 0);
    begin
        if DATA_WIDTH_OUT /= 8 then
            k(7 downto DATA_WIDTH_OUT) := (others => '0');
        end if;
        k(DATA_WIDTH_OUT-1 downto 0) := network_out;
        return k;
    end function;

begin

    i_network: entity work.${network_name}(rtl)
    port map (
        clock => clock,
        enable => network_enable,
        x => x,
        x_address => x_address,
        y => y,
        y_address => y_address,
        done => done
    );

    busy <= not done;
    wake_up <= done;

    receive_data_from_middleware: process (clock)
    variable int_addr : integer range 0 to 18 + X_NUM_VALUES;
    begin
        if rising_edge(clock) then
            if reset = '1' then
                network_enable <= '0';
            else
                if wr = '1' then
                    int_addr := to_integer(unsigned(address_in));
                    if int_addr = 16 then
                        network_enable <= data_in(0);
                    elsif int_addr >= 18 and int_addr < 18 + X_NUM_VALUES then
                        data_buf_in(int_addr-18) <= data_in(DATA_WIDTH_IN-1 downto 0);
                    end if;
                end if;
            end if;
        end if;
    end process;

    sendback_data_to_middleware: process  (clock)
    variable int_addr : integer range 0 to 18 + Y_NUM_VALUES;
    begin
        if rising_edge(clock) then
            if rd = '1' then
                int_addr := to_integer(unsigned(address_in));
                if int_addr <= 15 then
                    data_out(7 downto 0) <= skeleton_id_str(int_addr);
                elsif int_addr >= 18 and int_addr < 18 + Y_NUM_VALUES then
                    y_address <= std_logic_vector(unsigned(address_in(y_address'length-1 downto 0))-18);
                    data_out(7 downto 0) <= pad_output_to_middleware(y);
                end if;
            end if;
        end if;
    end process;

    send_buf_to_network: process (clock)
    begin
        if falling_edge(clock) then
            x <= data_buf_in(to_integer(unsigned(x_address)));
        end if;
    end process;

end rtl;
