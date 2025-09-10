-- Silas Brandenburg

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.numeric_std.ALL;

entity top_modul is
    generic (
        DATA_WIDTH : integer := 8;
        X_ADDR_WIDTH : integer := 5;
        Y_ADDR_WIDTH : integer := 2;
        IN_FEATURE_NUM : integer := 32;
        OUT_FEATURE_NUM : integer := 3;
        BAUD_RATE : integer := 115200;
        IN_BYTE_NUM : integer := 32;
        OUT_BYTE_NUM : integer := 3
    );
    Port ( clk : in STD_LOGIC;
           rx : in STD_LOGIC;
           nRST : in STD_LOGIC;    -- Reset button is '0'  when pressed
           tx : out STD_LOGIC
    );
end top_modul;

architecture rtl of top_modul is
    
    -- UART registers and pins
    signal reset_n      : std_logic;                    -- Reset Pin for UART Module; Reset on '1'
    signal tx_en        : std_logic;                    -- Enables the transmission when set on '1'
    signal tx_data      : std_logic_vector(7 downto 0); -- Data to be transmitted
    signal tx_busy      : std_logic;                    -- Is '1' as long as the transmission component is transmitting
    signal rx_done      : std_logic;                    -- Flag: Is '1' for one cycle when data reception is done
    signal rx_data      : std_logic_vector(7 downto 0); -- Received Data
    signal rx_error     : std_logic;                     
    
    -- Input buffer regs and pins
    signal input_buffer_data      : std_logic_vector(IN_FEATURE_NUM*8-1 downto 0); -- Data Output of receive unit
    signal input_buffer_done      : std_logic;                        -- Flag: Is set to '1' for one cycle when all data is received and buffered
    signal input_buffer_release   : std_logic;                        -- Flag: Set to one to 'release' the receive units outputs. The RU holds the buffer stable until release is '1'.
    
    
    -- Network wrapper regs and pins
    signal reset_nw     : std_logic;
    signal enable_network : std_logic;
    signal output_valid     : std_logic;
    signal input_data   : std_logic_vector(IN_FEATURE_NUM*DATA_WIDTH-1 downto 0);
    
    -- Output buffer regs and pins
    signal output_buffer_data      : std_logic_vector(OUT_FEATURE_NUM*8-1 downto 0); -- Data Input to send unit, this has to be filled with the Data you wannt to send over UART Example fo
    signal output_buffer_enable    : std_logic;                        -- Flag: Set on '1' for one cycle to start send unit
    signal output_buffer_done      : std_logic;                        -- Flag: Is set to '1' for one cycle when transmission of all data is done 
    
    -- State Machine
    type es_machine is (idle, received, inference, send, wait_tx);     
    signal es_state : es_machine := idle;
    
begin

    uart : entity work.uart_rxtx
        generic map(
            clk_freq => 100_000_000,
            baud_rate => BAUD_RATE,
            os_rate => 16,
            d_width => 8,
            stop_bits => 1,
            use_parity => 0,
            parity_eo => '0'
        )
        port map(
            clk => clk,
            reset_n => reset_n,
            tx_en => tx_en,
            tx_data => tx_data,
            rx => rx,
            tx => tx,
            tx_busy => tx_busy,
            rx_done => rx_done,
            rx_error => rx_error,
            rx_data => rx_data
        );
        
    network_wrapper : entity work.network_wrapper(rtl)
        generic map(
            DATA_WIDTH => DATA_WIDTH,
            X_ADDR_WIDTH => X_ADDR_WIDTH,
            Y_ADDR_WIDTH => Y_ADDR_WIDTH,
            IN_FEATURE_NUM => IN_FEATURE_NUM,
            OUT_FEATURE_NUM => OUT_FEATURE_NUM
        )
        port map(
            clk => clk,
            enable => enable_network,
            nRST => '1',
            input_data => input_data,
            output_data => output_buffer_data,
            output_valid => output_valid
        );
        
    output_buffer : entity work.out_buffer
        generic map(
            OUT_BYTE_NUM => OUT_BYTE_NUM
        )
        port map(
            clk => clk,
            data_in => output_buffer_data,
            enable  => output_buffer_enable,
            done    => output_buffer_done,
            tx_data => tx_data,
            tx_en   => tx_en,
            tx_busy => tx_busy
        );
        
    input_buffer : entity work.input_buffer
        generic map(
            IN_BYTE_NUM => IN_BYTE_NUM
        )
        port map(
            clk => clk,
            data_out => input_buffer_data,
            done => input_buffer_done,
            release => input_buffer_release,
            rx_data => rx_data,
            rx_done => rx_done
        );
    
    MAIN : process(clk)
        
    begin
    
    if input_buffer_done = '1' then -- Switch state when input buffer is full
    
        es_state <= received; 
    
    elsif rising_edge(clk) then
        
        case es_state is
        
        when idle => 
            output_buffer_enable <= '0';
            if input_buffer_done = '1' then
                es_state <= received;
            end if;
            
        when received =>   
            input_data <= input_buffer_data;       
            enable_network <= '1';
            es_state <= inference;
            
        when inference =>
            enable_network <= '0';
            if output_valid = '1' then
                input_buffer_release <= '1'; -- Input buffer can reset itself because buffer is already read
                es_state <= send;
            end if;
            
        when send => 
            input_buffer_release <= '0';      
            output_buffer_enable <= '1';       -- Enable output buffer
            es_state <= wait_tx;
            
        when wait_tx =>
            output_buffer_enable <= '0';       
            if output_buffer_done = '1' then   -- Wait until output buffer is done
                es_state <= idle;
            end if;
            
        end case;
        
    end if;
    
    end process MAIN;

end rtl;
