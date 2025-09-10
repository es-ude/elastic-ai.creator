library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity nn_tb is
end nn_tb;

architecture Behavioral of nn_tb is

    signal enable       : std_logic;
    signal clock        : std_logic := '0';
    signal x_address    : std_logic_vector(5-1 downto 0);
    signal y_address    : std_logic_vector(2-1 downto 0);
    signal x            : std_logic_vector(8-1 downto 0);
    signal y            : std_logic_vector(8-1 downto 0);
    signal done         : std_logic;
    
    signal finished     : std_logic := '0';
    constant half_period : time := 5 ns;
    constant clk_period : time := 10 ns;
    
    type x_buf_type is array(0 to 32-1) of std_logic_vector(8-1 downto 0);
    signal x_buf : x_buf_type := ("00000011","00000010","00000010","00000001","00000000","11111111","11111110","00000101","00010000","00001011","11111110","11110101","11110110","11111000","11111010","11111011","11111100","11111101","11111101","11111101","11111101","11111110","11111111","11111111","00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000");
    
    type y_buf_type is array(0 to 3-1) of std_logic_vector(8-1 downto 0);
    signal y_buf : y_buf_type;
    
    signal exp_output : y_buf_type := ("11011111","11101011","11100000");
begin

    DUT : entity work.ae_v1_wo_bn_encoder(rtl)
    port map(
        enable => enable,
        clock => clock,
        x_address => x_address,
        y_address => y_address,
        x => x,
        y => y,
        done => done
    );

    clock <= not clock after half_period when finished /= '1' else '0';
    
    stimulus : process
    begin
        
        wait for clk_period;
        enable <= '1';
        wait for clk_period;
        
        wait until done = '1';
        
        for j in 0 to 3-1 loop
            y_address <= std_logic_vector(to_unsigned(j,2));
            wait for clk_period;
            y_buf(j) <= y;
        end loop;
        
    end process;
    
    write_x : process(clock)
    begin
    
        if rising_edge(clock) then
            x <= x_buf(to_integer(unsigned(x_address)));
        end if;
    
    end process;

end Behavioral;
