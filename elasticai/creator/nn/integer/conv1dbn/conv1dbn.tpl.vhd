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
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        M_Q_DATA_WIDTH : integer := ${m_q_data_width};
        RESOURCE_OPTION : string := "${resource_option}";
        WEIGHTS_ROM_NAME : string := "${weights_rom_name}";
        BIAS_ROM_NAME : string := "${bias_rom_name}"
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
    -- Signal Declarations
    signal M_Q_SIGNED : signed(M_Q_DATA_WIDTH - 1 downto 0) := to_signed(M_Q, M_Q_DATA_WIDTH);
    signal x_buffer : array(0 to KERNEL_SIZE - 1) of signed(DATA_WIDTH - 1 downto 0);
    signal mac_result : signed(2 * (DATA_WIDTH + 1) - 1 downto 0);
    signal scaled_output : signed(DATA_WIDTH downto 0);
    signal x_idx : integer range 0 to KERNEL_SIZE - 1 := 0;
    signal y_idx : integer range 0 to OUT_CHANNELS - 1 := 0;

    -- FSM States
    type t_state is (s_idle, s_load_x, s_compute, s_scale, s_output, s_done);
    signal state : t_state := s_idle;

    -- Address and Data Signals for ROMs
    signal weights_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal weights_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal bias_addr : std_logic_vector(Y_ADDR_WIDTH - 1 downto 0);
    signal bias_data : std_logic_vector(2 * (DATA_WIDTH + 1) - 1 downto 0);

begin
    process(clock)
    begin
        if rising_edge(clock) then
            if enable = '0' then
                state <= s_idle;
                done <= '0';
            else
                case state is
                    when s_idle =>
                        x_idx <= 0;
                        y_idx <= 0;
                        mac_result <= (others => '0');
                        scaled_output <= (others => '0');
                        state <= s_load_x;

                    when s_load_x =>
                        if x_idx < KERNEL_SIZE then
                            x_buffer(x_idx) <= signed(x);
                            x_idx <= x_idx + 1;
                        else
                            x_idx <= 0;
                            state <= s_compute;
                        end if;

                    when s_compute =>
                        mac_result <= mac_result + signed(weights_data) * x_buffer(x_idx);
                        if x_idx < KERNEL_SIZE - 1 then
                            x_idx <= x_idx + 1;
                        else
                            state <= s_scale;
                        end if;

                    when s_scale =>
                        scaled_output <= scaling(mac_result + signed(bias_data), M_Q_SIGNED, M_Q_SHIFT);
                        state <= s_output;

                    when s_output =>
                        y <= std_logic_vector(scaled_output + to_signed(Z_Y, scaled_output'length));
                        if y_idx < OUT_CHANNELS - 1 then
                            y_idx <= y_idx + 1;
                            state <= s_load_x;
                        else
                            state <= s_done;
                        end if;

                    when s_done =>
                        done <= '1';
                        state <= s_idle;

                    when others =>
                        state <= s_idle;
                end case;
            end if;
        end if;
    end process;

    -- Instantiate ROM for weights
    rom_w : entity ${work_library_name}.${weights_rom_name}(rtl)
    port map (
        clk => clock,
        en  => '1',
        addr => weights_addr,
        data => weights_data
    );

    -- Instantiate ROM for biases
    rom_b : entity ${work_library_name}.${bias_rom_name}(rtl)
    port map (
        clk => clock,
        en  => '1',
        addr => bias_addr,
        data => bias_data
    );

end architecture;
