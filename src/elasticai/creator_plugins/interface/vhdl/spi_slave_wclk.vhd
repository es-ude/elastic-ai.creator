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
-- SPI CONFIGURATION MODE: CPOL = 0, CPHA = 1 --> MODE = 1, MSB first

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity spi_slave is
  port(
    clk             : in    std_logic;
    reset_n         : in    std_logic;  --reset from master
    sclk            : in    std_logic;  --spi clk from master
    ss_n            : in    std_logic;  --active low slave select
    mosi            : in    std_logic;  --master out, slave in
    miso            : out   std_logic;  --master in, slave out
    -- Parallel interface to control the skeleton
    we              : out   std_logic;    -- enable writing to user logic
    re              : out   std_logic;
    addr            : out   std_logic_vector(15 downto 0) := (others => '0');  -- which address of the user logic to read/write
    data_wr         : out   std_logic_vector(7 downto 0) := (others => '0');  -- data writing to user logic
    data_rd         : in    std_logic_vector(7 downto 0) := (others => '0')  --logic provide data to transmit register
);
    
end spi_slave;

architecture rtl OF spi_slave is
    signal bit_cnt_s : integer;  --'1' for active transaction bit
    signal rx_buf  : std_logic_vector(7 downto 0) := (others => '0');  --receiver buffer
    signal tx_buf  : std_logic_vector(7 downto 0) := (others => '0');  --transmit buffer
    signal command  : std_logic_vector(7 downto 0) := (others => '0');  --receiver buffer
    type state_t is (s_idle, s_cmd, s_addr_h, s_addr_l, s_data);
    signal state   : state_t;
    signal addr_read, addr_write : std_logic_vector(15 downto 0) := (others => '0');
    signal out_trigger : std_logic;
    signal re_s, we_s : std_logic;
    signal sclk_dly : std_logic;
    signal sclk_falling_edge : std_logic;
    signal sclk_rising_edge : std_logic;
    signal sclk_buf : std_logic;
    signal mosi_buf : std_logic;
    
BEGIN
   re <= re_s;
   we <= we_s;   
   addr <= addr_read when re_s='1' else addr_write when we_s='1' else (others => '0');
   sclk_falling_edge <= not sclk_buf and sclk_dly;
   sclk_rising_edge <= sclk_buf and not sclk_dly; 

    --keep track of miso/mosi bit counts for data alignment
    process(clk)
    variable rx_buf_var : std_logic_vector(7 downto 0) := (others => '0');
    variable bit_count_var : integer range 0 to 8;
    variable bytes_counter : integer range 0 to 1023;
    variable addr_to_write_var : std_logic_vector(15 downto 0) := (others => '0');
    variable addr_offset_var : std_logic_vector(15 downto 0) := (others => '0');
    variable temp_d : std_logic_vector(7 downto 0);
    begin
        if(rising_edge(clk)) then
            if(ss_n = '1' or reset_n = '0') then                         --this slave is not selected or being reset
                state <= s_idle;
                out_trigger <= '0';
                bit_count_var := 0;
                re_s <= '0';
                miso <= '0';
                sclk_buf <= '0';
                sclk_dly <= '0';
                mosi_buf <= '0';
            else     
                -- delay of the SCLK pin for edge detection
                sclk_buf <= sclk;  
                sclk_dly <= sclk_buf;
                mosi_buf <= mosi;                                                   
            
                -- controlling the miso pin
                if sclk_rising_edge = '1' then
                    if command=x"40" and state = s_data then  --write status register to master\
                        if bit_cnt_s =0 then
                            temp_d := data_rd;
                        end if;
                        miso <= temp_d(7-bit_cnt_s);                  --send transmit register data to master
                    else
                        miso <= '0';
                    end if;
                end if;
        
                -- sensing new bit input on mosi
                if sclk_falling_edge = '1' then         
                    rx_buf_var(7-bit_count_var) := mosi_buf;
                    bit_count_var := bit_count_var+1;
                    if bit_count_var = 8 then
                        bit_count_var := 0;
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
                            addr_to_write_var := std_logic_vector(unsigned(addr_offset_var)+to_unsigned(bytes_counter, addr'length)); -- increase the address
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
                            addr_read <= std_logic_vector(unsigned(addr_offset_var)+to_unsigned(bytes_counter, addr'length));
                        end if;
                    end if;
                end if;
            end if;
            bit_cnt_s <= bit_count_var;
        end if;
    end process;   
    
    process(clk)
    variable timer_down:integer range 0 to 4;
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                we_s <= '0';
            else
                if out_trigger='1' then
                    we_s <= '1';
                    timer_down := 4;
                 end if;
                timer_down := timer_down-1;
                if timer_down=0 then
                    we_s <= '0';
                end if;
            end if;
        end if;
    end process;
end rtl;
