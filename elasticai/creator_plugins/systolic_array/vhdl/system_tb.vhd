library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity system_tb is
end system_tb;

architecture rtl of system_tb is

    signal finished : std_logic := '0';
    constant half_period : time := 5 ns;
    constant clk_period : time := 10 ns;
    signal i : integer;

    constant DATA_WIDTH : integer := 8;
    constant X_ADDR_WIDTH : integer :=5;
    constant Y_ADDR_WIDTH : integer := 2;    
    constant IN_FEATURE_NUM : integer := 32;
    constant OUT_FEATURE_NUM : integer := 3;
    constant BAUD_RATE : integer := 115200;
    constant IN_BYTE_NUM : integer := 32;
    constant OUT_BYTE_NUM : integer := 3;
    
    signal clk : std_logic := '0';
    signal rx_tm : std_logic;
    signal reset : std_logic;
    signal tx_tm : std_logic;
    
    signal baud_pulse   : std_logic;
    signal os_pulse     : std_logic;
    
    signal tx_en        : std_logic;
    signal tx_data      : std_logic_vector(7 downto 0);
    signal tx_busy      : std_logic;
    
    signal rx_done      : std_logic;
    signal rx_error     : std_logic;
    signal rx_data      : std_logic_vector(7 downto 0);

    -- Stimulus data
    type x_buf_type is array(0 to 32-1) of std_logic_vector(8-1 downto 0);
    signal x_buf_0 : x_buf_type := ("00000011","00000011","00000011","00000100","00000111","00001100","00010011","00011001","00011001","00010101","00001110","00000111","00000000","11111010","11110010","11101011","11100110","11100010","11100000","11100001","11100011","11100110","11101001","11101011","11101101","11101111","11101111","11101111","11101111","11101111","11110000","11110000");
    signal x_buf_1 : x_buf_type := ("00000010","00000010","00000010","00000010","00000001","00000000","00000011","00010101","00100000","00010000","11111000","11101101","11101110","11110001","11110100","11110111","11111001","11111100","11111101","11111111","11111111","00000000","11111111","11111111","11111111","11111111","11111111","11111111","00000000","00000000","00000000","00000000");
    signal x_buf_2 : x_buf_type := ("00000000","00000000","00000000","00000000","00000000","11111110","00000011","00010110","00100000","00001101","11110110","11101101","11101111","11110001","11110011","11110110","11111000","11111010","11111011","11111100","11111101","11111101","11111110","11111110","11111110","11111110","11111111","00000000","00000000","00000000","00000000","00000000");
    
    type y_buf_type is array(0 to 3-1) of std_logic_vector(8-1 downto 0);
    signal y_buf_0 : y_buf_type;
    signal y_buf_1 : y_buf_type;
    signal y_buf_2 : y_buf_type;

begin

    -- UART Stub module
    tx_stub : entity work.uart_tx(Behavioral)
    generic map(
        d_width => 8,
        stop_bits => 1,
        use_parity => 0,
        parity_eo => '0'
    )
    port map(
        clk     => clk,
        clk_en  => baud_pulse,
        reset_n => '0',
        tx_en   => tx_en,
        tx_data => tx_data,
        tx_busy => tx_busy,
        tx      => rx_tm
    );
    
    rx_stub : entity work.uart_rx(Behavioral)
    generic map(
        d_width => 8,
        stop_bits => 1,
        use_parity => 0,
        parity_eo => '0',
        os_rate =>  16
    )
    port map(
        clk => clk,
        clk_en => os_pulse,
        reset_n => '0',
        rx => tx_tm,
        rx_done => rx_done,
        rx_error => rx_error,
        rx_data => rx_data
    );
    
    baud_module : entity work.baudrate_gen
    generic map(
        clk_freq    => 100_000_000,
        baud_rate   => 115_200,
        os_rate     => 16)
    port map(
        clk         => clk,
        reset_n     => '0',
        baud_pulse  => baud_pulse,
        os_pulse    => os_pulse);
        
    dut : entity work.top_modul(rtl)
    generic map(
        DATA_WIDTH => DATA_WIDTH,
        X_ADDR_WIDTH => X_ADDR_WIDTH,
        Y_ADDR_WIDTH => Y_ADDR_WIDTH,
        IN_FEATURE_NUM =>  IN_FEATURE_NUM,
        OUT_FEATURE_NUM => OUT_FEATURE_NUM,
        IN_BYTE_NUM => IN_BYTE_NUM,
        OUT_BYTE_NUM => OUT_BYTE_NUM
    )
    port map(
        clk => clk,
        rx => rx_tm,
        nRST => reset,
        tx => tx_tm
    );
    
    clk <= not clk after half_period when finished /= '1' else '0';
    
    stimulus : process
    begin
    
        wait for clk_period;
        
        -- First inference
        
        for idx in 0 to 32-1 loop
        
            i <= idx;
            tx_data <= x_buf_0(idx);
            tx_en <= '1';
            wait until tx_busy = '0';
            tx_en <= '0';
        
        end loop;
        
        for idx in 0 to 3-1 loop
            
            i <= idx;
            wait until rx_done = '1';
            y_buf_0(idx) <= rx_data;
        
        end loop;
        
        wait for clk_period;
        
        -- Second inference
        
        for idx in 0 to 32-1 loop
        
            i <= idx;
            tx_data <= x_buf_1(idx);
            tx_en <= '1';
            wait until tx_busy = '0';
            tx_en <= '0';
        
        end loop;
        
        for idx in 0 to 3-1 loop
            
            i <= idx;
            wait until rx_done = '1';
            y_buf_1(idx) <= rx_data;
        
        end loop;
        
        -- Thirs inference
        
        for idx in 0 to 32-1 loop
        
            i <= idx;
            tx_data <= x_buf_2(idx);
            tx_en <= '1';
            wait until tx_busy = '0';
            tx_en <= '0';
        
        end loop;
        
        for idx in 0 to 3-1 loop
            
            i <= idx;
            wait until rx_done = '1';
            y_buf_2(idx) <= rx_data;
        
        end loop;
        
        finished <= '1';
    
    end process stimulus;

end rtl;
