library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity input_buffer is
    generic (
        IN_BYTE_NUM : integer -- Number of bytes my differ from number of features in the future
    );
    port( 
        clk         : in std_logic;
        -- Interface Pins
        data_out    : out std_logic_vector(IN_BYTE_NUM*8-1 downto 0);
        done        : out std_logic;
        release     : in std_logic;
        -- Control Pins
        rx_data     : in std_logic_vector(7 downto 0);
        rx_done     : in std_logic
     );
end input_buffer;

architecture rtl of input_buffer is

    signal rx_done_sync : std_logic := '0'; -- Synchronized version of rx_done
    signal byte_count : integer := IN_BYTE_NUM;
    
    type buffer_machine is (idle, buf, done_buf, hold_output, zero_output);
    signal buf_state : buffer_machine := idle;
    
    signal in_buf : std_logic_vector(IN_BYTE_NUM*8-1 downto 0) := (others => '0');

begin

    MAIN : process(clk)
    begin
    
        if rx_done = '1' then
        
            buf_state <= buf;
        
        elsif rising_edge(clk) then
        
            case buf_state is
            
                when idle =>
                    done <= '0';
                     if byte_count = 0 then
                        done <= '1';
                        buf_state <= hold_output;
                    else
                        buf_state <= idle;
                    end if;
                    
                when buf =>
                    in_buf(byte_count*8-1 downto byte_count*8-8) <= rx_data;
                    buf_state <= done_buf;
                    
                when done_buf =>
                    byte_count <= byte_count - 1;
                    buf_state <= idle;
                    
                when hold_output =>
                    done <= '0';
                    data_out <= in_buf;
                    if release = '1' then
                        buf_state <= zero_output;
                    end if;
                    
                when zero_output =>
                    in_buf <= (others => '0');
                    data_out <= (others => '0');
                    byte_count <= IN_BYTE_NUM;
                    buf_state <= idle;
                
            end case;
            
        end if;
    
    end process MAIN;

end rtl;
