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
    constant DATA_WIDTH : integer := 8;
    constant FRAC_WIDTH : integer := 4;

    constant X_ADDR_WIDTH : integer := 4;


    signal network_enable :  std_logic;

    signal c_config_en :  std_logic;
    signal done :  std_logic;

    signal x_config_en :  std_logic;
    signal x_config_data :  std_logic_vector(DATA_WIDTH-1 downto 0);
    signal x_config_addr :  std_logic_vector(X_ADDR_WIDTH-1 downto 0);

    signal network_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);

begin

--    led_ctrl(0) <= reset;
    led_ctrl(2) <= network_enable;
    led_ctrl(3) <= done;
--    led_ctrl(3) <= '0';

    i_${network_name}: entity work.${network_name}(rtl)
    generic map(
        DATA_WIDTH  => DATA_WIDTH,
        FRAC_WIDTH  => FRAC_WIDTH,
        IN_ADDR_WIDTH => X_ADDR_WIDTH
    )
    port map (
        clock => clock,
        enable => network_enable,
        x => x_config_data,
        addr_in => x_config_addr,
        x_we => x_config_en,

        done => done,

        d_out => network_out_data
    );

    -- orignial implementation
    busy <= not done;
    wake_up <= done;


    -- process data receive
    process (clock, rd, wr, reset)
        variable int_addr : integer range 0 to 20000;
    begin

        if reset = '1' then

            network_enable <= '0';
            led_ctrl(1) <='0';
        else
        -- beginning/end
            if rising_edge(clock) then
                -- process address of written value

                -- calculate <= '0'; -- set to not calculate (can be overwritten below)

                if wr = '1' or rd = '1' then
                    -- variable being set
                    -- reverse from big to little endian
                    int_addr := to_integer(unsigned(address_in));
                    if wr = '1' then

                        if int_addr<6 then
                            x_config_data <= data_in(7 downto 0);
                            x_config_addr <= address_in(x_config_addr'length-1 downto 0);
                            x_config_en <= '1';
                            led_ctrl(1) <= '1';
                        elsif int_addr=100 then
                            network_enable <= data_in(0);
                        end if;
                    elsif rd = '1' then
                        if int_addr=0 then
                            data_out(7 downto 0) <= x"aa";
                        elsif int_addr=1 then
                            data_out(7 downto 0) <= network_out_data;
                        elsif int_addr=2 then
                            data_out(7 downto 0) <= x"bb";
                        elsif int_addr=2000 then
                            data_out(7 downto 0) <= x"14";

                        else
                            data_out(7 downto 0) <= address_in(7 downto 0);
                        end if;
                    end if;
                else
                    x_config_en <= '0';
                end if;

            end if;
        end if;

    end process;

end rtl;
