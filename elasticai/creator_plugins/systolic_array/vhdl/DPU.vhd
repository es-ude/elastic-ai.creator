----------------------------------------------------------------------------------
-- Engineer: Silas Brandenburg
-- 
-- Create Date: 02/22/2025 12:34:10 PM
-- Design Name: 
-- Module Name: DPU - rtl
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


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.bus_package.all;

entity DPU is
    generic (
        X_ADDR_WIDTH : integer := 5;
        IN_FEATURE_NUM : integer := 32;
        W_ADDR_WIDTH : integer := 10;
        NUM_LAYER : integer := 3;
        NETWORK_DIMENSIONS : dim_array := (32, 20, 14, 3);
        WEIGHT_NUM : integer := 999;
        DATA_WIDTH : integer := 8;
        FRAC_WIDTH : integer := 5
    );
    port ( 
        clk : in std_logic;
        nRST : in std_logic;
        enable : in std_logic;
        -- Activation Buffer Interface
        x_addr : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        x_bus : in bus_array_4_8;
        -- Weight Block Interface
        w_addr : out std_logic_vector(W_ADDR_WIDTH-1 downto 0);
        w_bus : in bus_array_4_8;
        -- Output Buffer Interface
        o_addr : in std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        o_bus : out bus_array_4_8;
        -- Done
        done : out std_logic
    );
end DPU;

architecture rtl of DPU is
    ---------------------------------------------
    -- Signals for Instantiations
    ---------------------------------------------
    -- Activation Buffer
    signal enable_act_buf           : std_logic;
    signal num_valid_vals_act_buf   : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal r_addr_act_buf           : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal r_data_act_buf           : bus_array_4_8;
    signal r_w_act_buf              : std_logic;
    signal w_addr_act_buf           : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal w_data_act_buf           : bus_array_4_8;
    -- Weight Buffer
    signal enable_wt_buf            : std_logic;
    signal num_valid_vals_wt_buf    : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal r_addr_wt_buf            : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal r_data_wt_buf            : bus_array_4_8;
    signal r_w_wt_buf               : std_logic;
    signal w_addr_wt_buf            : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal w_data_wt_buf            : bus_array_4_8;
    -- Bias Buffer
    signal enable_b_buf             : std_logic;
    signal num_valid_vals_b_buf     : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal r_addr_b_buf             : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal r_data_b_buf             : std_logic_vector(8-1 downto 0);
    signal r_w_b_buf                : std_logic;
    signal w_addr_b_buf             : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal w_data_b_buf             : bus_array_4_8;
    -- Matrix processing unit
    signal  enable_mpu              : std_logic;
    signal  dim_mpu                 : integer;
    signal  enable_inp_buf_inf      : std_logic;
    signal  r_addr_inp_buf_inf      : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal  r_data_inp_buf_inf      : bus_array_4_8;
    signal  r_w_inp_buf_inf         : std_logic;
    signal  enable_wt_buf_inf       : std_logic;
    signal  r_addr_wt_buf_inf       : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal  r_data_wt_buf_inf       : bus_array_4_8;
    signal  r_w_wt_buf_inf          : std_logic;
    signal  bias_mpu                : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal  output_mpu              : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal  done_mpu                : std_logic;
    -- Output Buffer
    signal  enable_o_buf            : std_logic;
    signal  r_addr_o_buf            : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal  r_data_o_buf            : bus_array_4_8;
    signal  r_w_o_buf               : std_logic;
    signal  w_addr_o_buf            : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal  w_data_o_buf            : std_logic_vector(DATA_WIDTH-1 downto 0);

    ---------------------------------------------
    -- State Machines/Process signals
    ---------------------------------------------
    -- Main Process
    type    t_main is (m_idle, m_buf_act, m_buf_b, m_buf_wt, m_inference, m_step, m_buf_out, m_done);
    signal  s_main : t_main := m_idle; 
    signal  current_layer : integer;
    signal  current_column : integer;
    signal  current_address_w_ram   : integer;
    signal  bias_addr_inf           : integer;
    signal  output_addr_inf         : integer;
    signal  enable_o_buf_main       : std_logic;
    signal  enable_o_buf_main_done  : std_logic;
    signal  r_w_o_buf_main          : std_logic;
    signal  r_w_o_buf_main_done     : std_logic;
    -- Buffer Activations Process
    signal  start_act_buf           : std_logic;
    signal  enable_act_buf_ba       : std_logic;
    signal  current_address_act_buf : integer;
    signal  r_w_act_buf_ba          : std_logic;
    signal  enable_o_buf_ba         : std_logic;
    signal  r_w_o_buf_ba            : std_logic;
    signal  r_addr_o_buf_ba         : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal  done_act_buf            : std_logic;
    type    t_buf_act is (ba_idle, ba_read, ba_write, ba_done);
    signal  s_buf_act               : t_buf_act := ba_idle;
    -- Buffer Bias Process
    signal  start_b_buf             : std_logic;
    signal  enable_b_buf_bb         : std_logic;
    signal  current_address_b_block : integer;
    signal  current_address_b_buf   : integer;
    signal  done_b_buf              : std_logic;
    type    t_buf_b is (bb_idle, bb_read, bb_write, bb_done);
    signal  s_buf_b                 : t_buf_b := bb_idle;
    -- Buffer Weights Process
    signal  start_wt_buf            : std_logic;
    signal  enable_wt_buf_bw        : std_logic;
    signal  current_address_w_block : integer;
    signal  current_address_w_buf   : integer;
    signal  r_w_wt_buf_bw           : std_logic;
    signal  done_wt_buf             : std_logic;
    type    t_buf_wt is (bw_idle, bw_read, bw_write, bw_done);
    signal  s_buf_wt                : t_buf_wt := bw_idle;
    -- Inference Process
    signal  start_inference         : std_logic;
    signal  enable_b_buf_inf        : std_logic;
    signal  enable_o_buf_inf        : std_logic;
    signal  r_w_o_buf_inf           : std_logic;
    signal  done_inference          : std_logic;
    type    t_inf is (inf_idle, inf_calc, inf_done);
    signal  s_inf                   : t_inf := inf_idle;
    ---------------------------------------------
    -- Signals
    ---------------------------------------------
    signal  w_addr_b : std_logic_vector(W_ADDR_WIDTH-1 downto 0);
    signal  w_addr_w : std_logic_vector(W_ADDR_WIDTH-1 downto 0); 

begin

    ---------------------------------------------
    -- Instantiations
    ---------------------------------------------
    -- Unified Buffer for Inputs
    act_buf : entity work.unified_buffer(rtl)
    generic map(
        MAX_FEATURE_NUM => NETWORK_DIMENSIONS(0),
        X_ADDR_WIDTH => X_ADDR_WIDTH
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable_act_buf,
        num_valid_vals => num_valid_vals_act_buf,
        r_addr => r_addr_act_buf,
        r_data => r_data_act_buf,
        r_w => r_w_act_buf,
        w_addr => w_addr_act_buf,
        w_data => w_data_act_buf
    );
    -- Unified Buffer for Weights 
    wt_buf : entity work.unified_buffer(rtl)
    generic map(
        MAX_FEATURE_NUM => NETWORK_DIMENSIONS(0),
        X_ADDR_WIDTH => X_ADDR_WIDTH 
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable_wt_buf,
        num_valid_vals => num_valid_vals_wt_buf,
        r_addr => r_addr_wt_buf,
        r_data => r_data_wt_buf,
        r_w => r_w_wt_buf,
        w_addr => w_addr_wt_buf,
        w_data => w_data_wt_buf
    );
    -- Unified Buffer for Bias
    b_buf : entity work.bias_buffer(rtl)
    generic map(
        MAX_FEATURE_NUM => NETWORK_DIMENSIONS(1),
        X_ADDR_WIDTH => X_ADDR_WIDTH
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable_b_buf,
        num_valid_vals => num_valid_vals_b_buf,
        r_addr => r_addr_b_buf,
        r_data => r_data_b_buf,
        r_w => r_w_b_buf,
        w_addr => w_addr_b_buf,
        w_data => w_data_b_buf
    );
    -- Matrix processing unit
    mpu : entity work.mpu(rtl)
    generic map(
        X_ADDR_WIDTH => X_ADDR_WIDTH,
        MAX_DIM => NETWORK_DIMENSIONS(1),
        DATA_WIDTH => DATA_WIDTH,
        FRAC_WIDTH => FRAC_WIDTH
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable_mpu,
        dim => dim_mpu,
        enable_inp_buf => enable_inp_buf_inf,
        r_addr_inp_buf => r_addr_act_buf,
        r_data_inp_buf => r_data_act_buf,
        r_w_inp_buf => r_w_inp_buf_inf,
        enable_wt_buf => enable_wt_buf_inf,
        r_addr_wt_buf => r_addr_wt_buf,
        r_data_wt_buf => r_data_wt_buf,
        r_w_wt_buf => r_w_wt_buf_inf,
        bias => bias_mpu,
        current_layer => current_layer,
        output => output_mpu,
        done => done_mpu
    );
    -- Output buffer
    o_buf : entity work.output_buffer(rtl)
    generic map(
        MAX_FEATURE_NUM => NETWORK_DIMENSIONS(1),
        X_ADDR_WIDTH => X_ADDR_WIDTH
    )
    port map(
        clk => clk,
        nRST => nRST,
        enable => enable_o_buf,
        r_addr => r_addr_o_buf,
        r_data => r_data_o_buf,
        r_w => r_w_o_buf,
        w_addr => w_addr_o_buf,
        w_data => w_data_o_buf
    );

    main : process(clk)
    begin
        
        if rising_edge(clk) then 
    
            case (s_main) is
                
                when m_idle =>
                    current_layer <= 0;
                    current_address_w_ram <= 0;
                    bias_addr_inf <= 0;
                    output_addr_inf <= 0;
                    r_w_b_buf <= '0';
                    done <= '0';
                    enable_o_buf_main_done <= '0';
                    r_w_o_buf_main_done <= '0';
                    if enable = '1' then
                        s_main <= m_buf_act;
                    end if;
                
                when m_buf_act =>
                    start_act_buf <= '1';
                    current_column <= 0;
                    if done_act_buf = '1' then
                        start_act_buf <= '0';
                        s_main <= m_buf_b;
                    end if;
                    
                when m_buf_b =>
                    start_b_buf <= '1';
                    r_w_b_buf <= '0';
                    if done_b_buf='1' then
                        start_b_buf <= '0';
                        current_address_w_ram <= current_address_w_ram + NETWORK_DIMENSIONS(current_layer+1); --Bummst bei letztem Layer
                        s_main <= m_buf_wt;
                    end if;
                
                when m_buf_wt =>
                    start_wt_buf <= '1';
                    if done_wt_buf = '1' then
                        start_wt_buf <= '0';
                        current_address_w_ram <= current_address_w_ram + NETWORK_DIMENSIONS(current_layer);
                        s_main <= m_inference;
                    end if;
                    
                
                when m_inference =>
                    start_inference <= '1';
                    r_w_b_buf <= '1';
                    r_addr_b_buf <= std_logic_vector(to_unsigned(bias_addr_inf, X_ADDR_WIDTH));
                    w_addr_o_buf <= std_logic_vector(to_unsigned(output_addr_inf, X_ADDR_WIDTH));
                    if done_inference = '1' then
                        start_inference <= '0';
                        bias_addr_inf <= bias_addr_inf + 1;
                        output_addr_inf <= output_addr_inf + 1;
                        s_main <= m_step;
                    end if;
                
                when m_step =>
                    if current_column >= NETWORK_DIMENSIONS(current_layer+1)-1 then
                        s_main <= m_buf_out;
                    else
                        current_column <= current_column + 1;
                        s_main <= m_buf_wt;
                    end if;
                    
                when m_buf_out =>
                    if current_layer >= NUM_LAYER-1 then
                        s_main <= m_done;
                    else 
                        current_layer <= current_layer + 1;
                        bias_addr_inf <= 0;
                        output_addr_inf <= 0;
                        s_main <= m_buf_act;
                    end if;
                
                when m_done =>
                    done <= '1';
                    enable_o_buf_main_done <= '1';
                    r_w_o_buf_main_done <= '1';
                    if enable = '0' then
                        enable_o_buf_main_done <= '0';
                        done <= '0';
                        s_main <= m_idle;
                    end if;
            
            end case;
        
        end if;
    
    end process;
    
    buffer_activations : process(clk) 
    begin
    
        if rising_edge(clk) then
        
            case(s_buf_act) is
            
                when ba_idle =>
                    current_address_act_buf <= 0;
                    enable_act_buf_ba <= '0';
                    enable_o_buf_ba <= '0';
                    done_act_buf <= '0';
                    if start_act_buf = '1' then
                        num_valid_vals_act_buf <= std_logic_vector(to_unsigned(NETWORK_DIMENSIONS(current_layer)-1, X_ADDR_WIDTH));
                        r_w_act_buf_ba <= '0';
                        r_w_o_buf_ba <= '1';
                        s_buf_act <= ba_read;
                    end if;
                
                when ba_read =>
                    if current_layer=0 then
                        x_addr <= std_logic_vector(to_unsigned(current_address_act_buf, X_ADDR_WIDTH));
                        w_addr_act_buf <= std_logic_vector(to_unsigned(current_address_act_buf, X_ADDR_WIDTH));
                    else    
                        r_addr_o_buf_ba <= std_logic_vector(to_unsigned(current_address_act_buf, X_ADDR_WIDTH));
                        w_addr_act_buf <= std_logic_vector(to_unsigned(current_address_act_buf, X_ADDR_WIDTH));
                        enable_o_buf_ba <= '1';
                    end if;
                    s_buf_act <= ba_write;
                
                when ba_write =>
                    enable_act_buf_ba <= '1'; 
                    if current_address_act_buf+4 >= NETWORK_DIMENSIONS(current_layer) then
                        s_buf_act <= ba_done;
                    else
                        current_address_act_buf <= current_address_act_buf + 4;
                        s_buf_act <= ba_read;
                    end if;
                    
                when ba_done =>
                    done_act_buf <= '1';
                    if start_act_buf = '0' then
                        enable_act_buf_ba <= '0';    
                        s_buf_act <= ba_idle;
                    end if;
                    
            end case;
            
            if current_layer = 0 then
                w_data_act_buf <= x_bus;
            else
                w_data_act_buf <= r_data_o_buf;
            end if;
        
        end if;
    
    end process;
    
    buffer_bias : process(clk)
    begin
    
        if rising_edge(clk) then
        
            case(s_buf_b) is
            
                when bb_idle =>
                    current_address_b_block <= current_address_w_ram;
                    current_address_b_buf <= 0;
                    enable_b_buf_bb <= '0';
                    done_b_buf <= '0';
                    if start_b_buf = '1' then
                        num_valid_vals_b_buf <= std_logic_vector(to_unsigned(NETWORK_DIMENSIONS(current_layer+1)-1, X_ADDR_WIDTH));
                        s_buf_b <= bb_read;
                    end if;
                
                when bb_read =>
                    w_addr_b <= std_logic_vector(to_unsigned(current_address_b_block, W_ADDR_WIDTH)); 
                    w_addr_b_buf <= std_logic_vector(to_unsigned(current_address_b_buf, X_ADDR_WIDTH));
                    s_buf_b <= bb_write;
                
                when bb_write =>
                    enable_b_buf_bb <= '1';
                    if current_address_b_buf >= NETWORK_DIMENSIONS(current_layer+1) then
                        s_buf_b <= bb_done;
                    else
                        current_address_b_buf <= current_address_b_buf + 4;
                        current_address_b_block <= current_address_b_block + 4;     
                        s_buf_b <= bb_read;
                    end if;
                
                when bb_done =>
                    done_b_buf <= '1';
                    enable_b_buf_bb <= '0';
                    if start_b_buf='0' then
                        s_buf_b <= bb_idle;
                    end if;
            
            end case;
            
            w_data_b_buf <= w_bus;
        
        end if;
    
    end process;
    
    buffer_weight : process(clk) 
    begin
    
        if rising_edge(clk) then
        
            case(s_buf_wt) is
            
                when bw_idle =>
                    done_wt_buf <= '0';
                    current_address_w_block <= current_address_w_ram;
                    current_address_w_buf <= 0;
                    enable_wt_buf_bw <= '0';
                    if start_wt_buf = '1' then
                        num_valid_vals_wt_buf <= std_logic_vector(to_unsigned(NETWORK_DIMENSIONS(current_layer)-1, X_ADDR_WIDTH));
                        r_w_wt_buf_bw <= '0';
                        s_buf_wt <= bw_read;
                    end if;
                
                when bw_read =>
                    w_addr_w <= std_logic_vector(to_unsigned(current_address_w_block, W_ADDR_WIDTH));
                    w_addr_wt_buf <= std_logic_vector(to_unsigned(current_address_w_buf, X_ADDR_WIDTH));
                    s_buf_wt <= bw_write;
                
                when bw_write =>
                    enable_wt_buf_bw <= '1';
                    if current_address_w_buf >= NETWORK_DIMENSIONS(current_layer) then 
                        done_wt_buf <= '1';
                        s_buf_wt <= bw_done;
                    else
                        current_address_w_block <= current_address_w_block + 4;
                        current_address_w_buf <= current_address_w_buf + 4;
                        s_buf_wt <= bw_read;
                    end if;
                
                when bw_done =>
                    enable_wt_buf_bw <= '0';
                    if start_b_buf='0' then
                        s_buf_wt <= bw_idle;
                    end if;
            
            end case;
        
            w_data_wt_buf <= w_bus;
        
        end if;
    
    end process;
    
    inference : process(clk)
    begin
    
        if rising_edge(clk) then
        
            case (s_inf) is 
            
                when inf_idle =>
                    enable_b_buf_inf <= '0';
                    enable_o_buf_inf <= '0';
                    done_inference <= '0';
                    if start_inference = '1' then
                        enable_b_buf_inf <= '1';
                        r_w_o_buf_inf <= '0';
                        dim_mpu <= NETWORK_DIMENSIONS(current_layer);
                        s_inf <= inf_calc;
                    end if;
                
                when inf_calc =>
                    bias_mpu <= r_data_b_buf;
                    enable_mpu <= '1';
                    if done_mpu = '1' then 
                        enable_o_buf_inf <= '1';
                        enable_mpu <= '0'; 
                        s_inf <= inf_done;
                    end if;
                
                when inf_done =>
                    done_inference <= '1';
                    w_data_o_buf <= output_mpu;
                    if start_inference = '0' then
                        s_inf <= inf_idle;
                    end if;
            
            end case;
        
        end if;
    
    end process;
    
    -- signal abstraction so that two processes can write
    w_addr <= w_addr_b when start_b_buf /= '0' else w_addr_w;
    -- Input buffer  
    enable_act_buf <= enable_act_buf_ba when start_act_buf /= '0' else enable_inp_buf_inf;
    r_w_act_buf <= r_w_act_buf_ba when start_act_buf /= '0' else r_w_inp_buf_inf;
    -- Weight buffer
    enable_wt_buf <= enable_wt_buf_bw when start_wt_buf /= '0' else enable_wt_buf_inf;
    r_w_wt_buf <= r_w_wt_buf_bw when start_wt_buf /= '0' else r_w_wt_buf_inf;
    -- Bias buffer
    enable_b_buf <= enable_b_buf_bb when start_b_buf /= '0' else enable_b_buf_inf;
    -- Output buffer
    enable_o_buf <= enable_o_buf_ba when start_act_buf /= '0' else enable_o_buf_main;
    r_w_o_buf <= r_w_o_buf_ba when start_act_buf /= '0' else r_w_o_buf_main;
    r_addr_o_buf <= r_addr_o_buf_ba when start_act_buf /= '0' else o_addr;
    
    enable_o_buf_main <= enable_o_buf_inf when start_inference /= '0' else enable_o_buf_main_done;
    r_w_o_buf_main <= r_w_o_buf_inf when start_inference /= '0' else r_w_o_buf_main_done; 
    
    o_bus <= r_data_o_buf; 

end rtl;
