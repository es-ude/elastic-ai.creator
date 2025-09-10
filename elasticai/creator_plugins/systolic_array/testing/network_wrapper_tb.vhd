library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity network_wrapper_tb is
end network_wrapper_tb;

architecture rtl of network_wrapper_tb is

    signal finished : std_logic := '0';
    constant half_period : time := 5 ns;
    constant clk_period : time := 10 ns;

    constant DATA_WIDTH : integer := 8;
    constant X_ADDR_WIDTH : integer := 5;
    constant Y_ADDR_WIDTH : integer := 2;
    constant IN_FEATURE_NUM : integer := 32;
    constant OUT_FEATURE_NUM : integer := 3;

    signal clk : std_logic := '0';
    signal enable : std_logic;
    signal input_data : std_logic_vector(IN_FEATURE_NUM*DATA_WIDTH-1 downto 0) 
        := "0000000000000000111111011111110011111101000000010000011100001101000011110001000000001110000011000000101000001001000001110000010100000100000000110000000100000000000000001111111111111110111111101111111011111110111111101111111011111110111111101111111011111110";
    signal output_data : std_logic_vector(OUT_FEATURE_NUM*DATA_WIDTH-1 downto 0);
    signal output_valid : std_logic;

begin

    dut : entity work.network_wrapper(rtl)
    generic map(
        DATA_WIDTH => DATA_WIDTH,
        X_ADDR_WIDTH => X_ADDR_WIDTH,
        Y_ADDR_WIDTH => Y_ADDR_WIDTH,
        IN_FEATURE_NUM => IN_FEATURE_NUM,
        OUT_FEATURE_NUM => OUT_FEATURE_NUM      
    )
    port map(
        clk => clk,
        enable => enable,
        input_data => input_data,
        output_data => output_data,
        output_valid => output_valid
    );

    clk <= not clk after half_period when finished /= '1' else '0';
    
    stimulus : process
    begin
    
        wait for clk_period;
        enable <= '1';
        wait for clk_period;
        enable <= '0';
        
        wait until output_valid = '1';
        
        wait for clk_period;
        enable <= '1';
        wait for clk_period;
        enable <= '0';
        
        wait until output_valid = '1';
        
        wait for clk_period;
        enable <= '1';
        wait for clk_period;
        enable <= '0';
        
        wait until output_valid = '1';
        
        finished <= '1';
    
    end process stimulus;

end rtl;
