library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity out_buffer is
    generic (
        OUT_BYTE_NUM : integer := 1
    );
    port (
        clk : in std_logic;
        -- Interface Pins
        data_in : in std_logic_vector(OUT_BYTE_NUM*8-1 downto 0);   -- Data that has to be send
        enable  : in std_logic;                                     -- Flag; Has to be '1' to start the send process
        done    : out std_logic;                                    -- Flag: Is '1' for one cycle when SU is done
        -- Control Pins/Pins that have to be mapped onto the uart module
        tx_data : out std_logic_vector(7 downto 0);                 
        tx_en   : out std_logic;
        tx_busy : in std_logic
    );
end out_buffer;

architecture rtl of out_buffer is

    signal byte_count : integer := OUT_BYTE_NUM;
    type buf_machine is (idle, buf, send, disable_tx, wait_tx);
    signal buf_state : buf_machine := idle;

begin

    MAIN : process(clk)
    begin
    
        if enable = '1' then
            buf_state <= buf;    -- Switch to Buf State when output buffer gets enabled
        
        elsif rising_edge(clk) then
        
            case buf_state is
            
            when idle =>  
                tx_data <= (others => '0');
                tx_en <= '0';
                done <= '0';
                byte_count <= OUT_BYTE_NUM;
                buf_state <= idle;
            
            when buf => 
                tx_data <= data_in(byte_count*8-1 downto byte_count*8-8); -- Put current byte on TX Bus
                buf_state <= send;
            
            when send =>
                tx_en <= '1';                   -- Enable Transmission
                byte_count <= byte_count - 1;   -- Decrease Counter
                buf_state <= disable_tx;
                
            when disable_tx => 
                tx_en <= '0';   -- Disable Transmission
                buf_state <= wait_tx;
                
            when wait_tx =>
                if tx_busy = '0' then       -- Wait until TX Module is not busy anymore
                    if byte_count = 0 then  -- If all bytes have been send byte_count is zero
                        done <= '1';        -- Issue that output buffer is done/empty
                        buf_state <= idle;   -- Go back to idle state
                    else 
                        buf_state <= buf;    -- If not all bytes have been send go back to buf state
                    end if;
                end if;
        
            end case;
        
        end if;
    
    end process MAIN;

end rtl;
