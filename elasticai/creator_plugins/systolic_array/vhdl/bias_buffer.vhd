----------------------------------------------------------------------------------
-- Engineer: Silas Brandenburg
-- 
-- Create Date: 02/24/2025 10:17:39 AM
-- Design Name: 
-- Module Name: unified_buffer - rtl
-- Project Name: Data Processing Unit
-- Target Devices: Digilent Arty S7
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.bus_package.all;

entity bias_buffer is
    generic(
        MAX_FEATURE_NUM : integer := 32;
        X_ADDR_WIDTH : integer := 5
    );
    port (
        clk : in std_logic;
        nRST : in std_logic;
        enable : in std_logic; -- Needs to be driven high the whole time
        num_valid_vals : in std_logic_vector(X_ADDR_WIDTH-1 downto 0); -- Number of valid Values in Buffer (e.g. 10-1 for third layer) 
        r_addr : in std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        r_data : out std_logic_vector(8-1 downto 0);
        r_w : in std_logic; -- 1 for read, 0 for write
        w_addr : in std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        w_data : in bus_array_4_8
    );
end bias_buffer;

architecture rtl of bias_buffer is

    --------------------------------------------
    -- Types, Signals and Constants
    --------------------------------------------
    type t_buf is array(0 to MAX_FEATURE_NUM-1) of std_logic_vector(7 downto 0);
    signal uni_buf : t_buf := (others => (others => '0'));

    --------------------------------------------
    -- State Machines
    --------------------------------------------
    type t_write is (w_read, w_write);
    signal s_write : t_write;

    --------------------------------------------
    -- Procedures
    --------------------------------------------
    procedure write_and_zero_pad (
        variable addr_w : in integer;
        variable num_valid_vals_int : in integer;
        signal uni_buf : out t_buf;
        signal data_in : in bus_array_4_8
    ) is
    begin
        for ii in 0 to 3 loop
            if addr_w+ii <= num_valid_vals_int then
                uni_buf(addr_w+ii) <= data_in(ii);
            else    
                uni_buf(addr_w+ii) <= (others => '0');
            end if;
        end loop;
            
    end procedure;

begin

    read : process(clk)
    
        variable addr_r : integer;
    
    begin
    
        if rising_edge(clk) then
    
            if enable ='0' then
                
                addr_r := 0;
                
            elsif r_w='1' and enable='1' then
                
                addr_r := to_integer(unsigned(r_addr));
                r_data <= uni_buf(addr_r);
                
            end if;
        
        end if;
    
    end process;

    write : process(clk)
    
        variable addr_w : integer;
        variable num_valid_vals_int : integer;
        
    begin
        if rising_edge(clk) then
        
            if nRST = '0' then
            
                uni_buf <= (others => (others => '0'));
                s_write <= w_read;
                
            elsif enable = '0' then
            
                addr_w := 0;
                s_write <= w_read;
                
            elsif r_w = '0' and enable = '1' then
            
                case (s_write) is
                
                    when w_read =>
                        num_valid_vals_int := to_integer(unsigned(num_valid_vals));
                        addr_w := to_integer(unsigned(w_addr)); -- Direkte Nutzung
                        s_write <= w_write;
                    
                    when w_write =>
                        write_and_zero_pad(addr_w, num_valid_vals_int, uni_buf, w_data);
                        s_write <= w_read;
                        
                end case;
                
            end if;
            
        end if;
        
    end process;

end rtl;