library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width};
        IN_CHANNELS : integer := ${in_channels};
        OUT_CHANNELS : integer := ${out_channels};
        KERNEL_SIZE : integer := ${kernel_size};
        STRIDE : integer := ${stride};
        PADDING : integer := ${padding};
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH - 1 downto 0);
        x   : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end ${name};

architecture rtl of ${name} is
    function multiply_accumulate(
                    w : in signed(DATA_WIDTH downto 0);
                    x_in : in signed(DATA_WIDTH downto 0);
                    y_0 : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0)
            ) return signed is
        variable TMP : signed(2 * (DATA_WIDTH + 1) - 1 downto 0) := (others => '0');
    begin
        TMP := w * x_in;
        return TMP + y_0;
    end function;

    function scaling(
        x_in : in signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
        scaler_m : in signed(M_Q_DATA_WIDTH - 1 downto 0);
        scaler_m_shift : in integer
    ) return signed is
        variable TMP : signed(2 * (DATA_WIDTH + 1) + M_Q_DATA_WIDTH - 1 downto 0) := (others => '0');
    begin
        TMP := x_in * scaler_m;
        TMP := shift_right(TMP, scaler_m_shift);
        return resize(TMP, DATA_WIDTH + 1);
    end function;

    signal weights_rom : array(0 to OUT_CHANNELS - 1, 0 to IN_CHANNELS - 1, 0 to KERNEL_SIZE - 1) of signed(DATA_WIDTH downto 0);
    signal biases_rom : array(0 to OUT_CHANNELS - 1) of signed(DATA_WIDTH downto 0);
    signal mac_result : signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
    signal scaled_output : signed(DATA_WIDTH downto 0);

    signal x_idx : integer range 0 to IN_CHANNELS - 1 := 0;
    signal y_idx : integer range 0 to OUT_CHANNELS - 1 := 0;
    signal kernel_idx : integer range 0 to KERNEL_SIZE - 1 := 0;
    signal state : integer := 0;
begin
    process (clock)
    begin
        if rising_edge(clock) then
            if enable = '1' then
                case state is
                    when 0 =>
                        mac_result <= (others => '0');
                        state <= 1;

                    when 1 =>
                        if kernel_idx < KERNEL_SIZE then
                            mac_result <= multiply_accumulate(
                                weights_rom(y_idx, x_idx, kernel_idx),
                                signed(x),
                                mac_result
                            );
                            kernel_idx <= kernel_idx + 1;
                        else
                            state <= 2;
                        end if;

                    when 2 =>
                        scaled_output <= scaling(mac_result, to_signed(M_Q, M_Q_DATA_WIDTH), M_Q_SHIFT);
                        y <= std_logic_vector(resize(scaled_output + to_signed(Z_Y, scaled_output'length), DATA_WIDTH));
                        done <= '1';
                        state <= 0;

                    when others =>
                        state <= 0;
                end case;
            else
                done <= '0';
            end if;
        end if;
    end process;
end rtl;
