library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.bus_package.all;
use ieee.std_logic_unsigned.all;

entity network_wrapper is
    generic (
        DATA_WIDTH : integer := 8;
        FRAC_WIDTH : integer := 5;
        X_ADDR_WIDTH : integer := 5;
        W_ADDR_WIDTH : integer := 10;
        Y_ADDR_WIDTH : integer := 2;
        IN_FEATURE_NUM : integer := 32;
        OUT_FEATURE_NUM : integer := 3
    );
    Port ( 
        clk : in STD_LOGIC;
        nRST : in STD_LOGIC;
        enable : in STD_LOGIC; -- Should be driven high when input data is valid
        input_data : in std_logic_vector(IN_FEATURE_NUM*DATA_WIDTH-1 downto 0);
        output_data : out std_logic_vector(OUT_FEATURE_NUM*DATA_WIDTH-1 downto 0);
        output_valid : out std_logic
        );
end network_wrapper;

architecture rtl of network_wrapper is

    -- Network Regs and Pins
    signal network_enable   : std_logic := '0';
    signal x_addr           : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal x_bus            : bus_array_4_8;
    signal w_addr           : std_logic_vector(W_ADDR_WIDTH-1 downto 0);
    signal w_bus            : bus_array_4_8;
    signal o_addr           : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal o_bus            : bus_array_4_8;
    signal network_done     : std_logic;

    type input_reg_t is array(0 to IN_FEATURE_NUM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal input_reg : input_reg_t := (others => (others => '0'));

    type output_reg_t is array(0 to OUT_FEATURE_NUM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal output_reg : output_reg_t := (others => (others => '0'));
    signal output_cnt : integer := 0;
    
    type nnw_machine is (idle, inference, read, done);
    signal nnw_state : nnw_machine := idle;
    
begin

    network : entity work.dpu(rtl)
    generic map(
        X_ADDR_WIDTH => X_ADDR_WIDTH,
        IN_FEATURE_NUM => IN_FEATURE_NUM,
        W_ADDR_WIDTH => W_ADDR_WIDTH,
        NUM_LAYER => 3,
        NETWORK_DIMENSIONS => (32,20,14,3),
        WEIGHT_NUM => 999,
        DATA_WIDTH => DATA_WIDTH,
        FRAC_WIDTH => FRAC_WIDTH
    )
    port map(
        clk => clk,
        nRST => '1',
        enable => network_enable,
        x_addr => x_addr,
        x_bus => x_bus,
        w_addr => w_addr,
        w_bus => w_bus,
        o_addr => o_addr,
        o_bus => o_bus,
        done => network_done
    );
    
    wram : entity work.weight_bram(rtl)
    generic map(
        WEIGHT_NUM => 999,
        BITWIDTH => DATA_WIDTH    
    )
    port map(
        clk => clk,
        en => network_enable,
        addr => w_addr,
        data => w_bus
    );
    
    input_mapping : for i in 0 to IN_FEATURE_NUM-1 generate
        input_reg(i) <= input_data((IN_FEATURE_NUM-i)*DATA_WIDTH-1 downto (IN_FEATURE_NUM-i-1)*DATA_WIDTH);
    end generate input_mapping;
    
    output_mapping : for i in 0 to OUT_FEATURE_NUM-1 generate
        output_data((OUT_FEATURE_NUM-i)*DATA_WIDTH-1 downto (OUT_FEATURE_NUM-i-1)*DATA_WIDTH) <= output_reg(i);
    end generate output_mapping;
    
    main : process(clk)
    begin
        
        if rising_edge(clk) then
        
            case nnw_state is
            
                when idle =>
                    output_valid <= '0';
                    if enable = '1' then
                        nnw_state <= inference;
                        network_enable <= '1';
                    end if;
                    
                when inference =>
                    if network_done = '1' then
                        nnw_state <= read;
                    end if;
                    
                when read =>
                    o_addr <= (others => '0');
                    nnw_state <= done;
                
                when done =>
                    for ii in 0 to 2 loop
                        output_reg(ii) <= o_bus(ii); 
                    end loop;
                    output_valid <= '1';
                    network_enable <= '0';
                    nnw_state <= idle;
            
            end case;
        
        end if;
        
    end process main;
    
    send_buf_to_network: process (clk)
    begin
        if rising_edge(clk) then
            for ii in 0 to 3 loop
                x_bus(ii) <= input_reg(conv_integer(x_addr+ii));
            end loop;
        end if;
    end process;

end rtl;
