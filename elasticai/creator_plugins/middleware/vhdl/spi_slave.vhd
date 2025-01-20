----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08/21/2022 09:45:55 PM
-- Design Name: 
-- Module Name: spi_slave - Behavioral
-- Project Name: 
-- Target Devices: 
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity spi_slave is
  port(
    reset_n      : in    std_logic;  --reset from master
    sclk         : in    std_logic;  --spi clk from master
    ss_n         : in    std_logic;  --active low slave select
    mosi         : in    std_logic;  --master out, slave in
    miso         : out    std_logic; --master in, slave out
    
    clk          : in std_logic;
    -- Parallel interface to control the skeleton
    addr         : out    std_logic_vector(15 downto 0) := (others => '0');  -- which address of the user logic to read/write
    
    we           : out    std_logic;    -- enable writing to user logic
    data_wr     : out    std_logic_vector(7 downto 0) := (others => '0');  -- data writing to user logic
     
    re        : out    std_logic;
    data_rd      : in    std_logic_vector(7 downto 0) := (others => '0')  --logic provide data to transmit register
    );
    
end spi_slave;

architecture rtl OF spi_slave is
    signal mode    : std_logic;  --groups modes by clock polarity relation to data
    signal bit_cnt_s : integer;  --'1' for active transaction bit

    signal rx_buf  : std_logic_vector(7 downto 0) := (others => '0');  --receiver buffer
    signal tx_buf  : std_logic_vector(7 downto 0) := (others => '0');  --transmit buffer
    signal command  : std_logic_vector(7 downto 0) := (others => '0');  --receiver buffer
    
    type state_t is (s_idle, s_cmd, s_addr_h, s_addr_l, s_data);
    signal state   : state_t;
    
    signal addr_read, addr_write : std_logic_vector(15 downto 0) := (others => '0');
    signal out_trigger : std_logic;
    
    signal re_s, we_s : std_logic;
    
BEGIN
   re <= re_s;
   we <= we_s;
   
   addr <= addr_read when re_s='1' else
           addr_write when we_s='1';

    --keep track of miso/mosi bit counts for data alignment
    process(ss_n, sclk, reset_n)
    variable rx_buf_var : std_logic_vector(7 downto 0) := (others => '0');
    variable bit_count_var : integer range 0 to 8;
    variable bytes_counter : integer range 0 to 1023;
    variable addr_to_write_var : std_logic_vector(15 downto 0) := (others => '0');
    variable addr_offset_var : std_logic_vector(15 downto 0) := (others => '0');
    begin
        if(ss_n = '1' or reset_n = '0') then                         --this slave is not selected or being reset
            state <= s_idle;
            out_trigger <= '0';
            bit_count_var := 0;
            re_s <= '0';
        else                                                         --this slave is selected

            IF(falling_edge(sclk)) then                                  --new bit on miso/mosi
            
                rx_buf_var(7-bit_count_var) := mosi;
                bit_count_var := bit_count_var+1;
                
                if bit_count_var=8 then
                    bit_count_var :=0 ;
                    if state = s_cmd then
                        command <= rx_buf_var;
                        state <= s_addr_h;
                        if rx_buf_var=x"40" then
                            re_s <= '1';
                        end if;
                    elsif state = s_addr_h then
                        addr_offset_var(15 downto 8) := rx_buf_var;
                        state <= s_addr_l;
                    elsif state = s_addr_l then
                        state <= s_data;
                        addr_offset_var(7 downto 0) := rx_buf_var;
                        
                        addr_read <= std_logic_vector(unsigned(addr_offset_var)); -- first addr
                        addr_to_write_var := std_logic_vector(unsigned(addr_offset_var));
                        addr_write <= addr_to_write_var;
                        bytes_counter := 1;
                    elsif state = s_data then
                        addr_write <= addr_to_write_var;
                        addr_to_write_var := std_logic_vector(unsigned(addr_offset_var)+to_unsigned(bytes_counter,addr'length)); -- increase the address
                        bytes_counter := bytes_counter+1;
                        data_wr <= rx_buf_var;
                        if command=x"80" then
                            out_trigger <= '1';
                        end if;
                     end if;
                elsif bit_count_var=2 then
                        if state = s_idle then
                            state <= s_cmd;
                        end if;
                        out_trigger <= '0';
                elsif bit_count_var=1 then
                    if command=x"40" then
                        addr_read <= std_logic_vector(unsigned(addr_offset_var)+to_unsigned(bytes_counter,addr'length));
                    end if;
                end if;
            end if;
            
        end if;
        bit_cnt_s <= bit_count_var;
    end process;
    
    process(sclk,ss_n,reset_n)
    variable temp_d : std_logic_vector(7 downto 0);
    begin
        if(ss_n = '1' or reset_n = '0') then                         --this slave is not selected or being reset
            miso <= '0';
        else
            if rising_edge(sclk) then
                    
                     if command=x"40" and state=s_data then  --write status register to master\
                        if bit_cnt_s =0 then
                            temp_d := data_rd;
                        end if;
                        miso <= temp_d(7-bit_cnt_s);                  --send transmit register data to master
                     else
                        miso <= '0';
                     end if;
                 end if;       
        end if;
    end process;
    
    
    process(reset_n, clk, out_trigger)
    variable timer_down:integer range 0 to 4;
    begin
        if reset_n='0' then
            we_s <= '0';
        elsif(rising_edge(clk)) then
        
            if out_trigger='1' then
                we_s <= '1';
                timer_down := 4;
             end if;
            timer_down := timer_down-1;
            if timer_down=0 then
                we_s <= '0';
            end if;
           
        end if;
    end process;

end rtl;
