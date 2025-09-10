----------------------------------------------------------------------------------
-- Engineer: Silas Brandenburg
-- 
-- Create Date: 03/03/2025 03:50:39 PM
-- Design Name: Matrix Processing Unit
-- Module Name: MPU - rtl
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

entity MPU is
    generic (
        X_ADDR_WIDTH : in integer;
        MAX_DIM : in integer;
        DATA_WIDTH : in integer;
        FRAC_WIDTH : in integer
    );
    port (
        clk : in std_logic;
        nRST : in std_logic;
        enable : in std_logic;
        dim : in integer;
        -- Input buffer interface
        enable_inp_buf  : out std_logic; 
        r_addr_inp_buf  : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        r_data_inp_buf  : in  bus_array_4_8;
        r_w_inp_buf     : out std_logic;
        -- Weight buffer interface
        enable_wt_buf   : out std_logic; 
        r_addr_wt_buf   : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        r_data_wt_buf   : in bus_array_4_8;
        r_w_wt_buf      : out std_logic;
        -- Bias
        bias            : in std_logic_vector(DATA_WIDTH-1 downto 0);
        -- Activation 
        current_layer   : in integer;
        -- Output
        output          : out std_logic_vector(DATA_WIDTH-1 downto 0);    
        done            : out std_logic
    );
end MPU;

architecture rtl of MPU is

    -- State Machine
    type t_inf_m is (s_idle, s_read, s_mac, s_step_mac, s_bias, s_act, s_done);
    signal s_inf : t_inf_m := s_idle;
    
    -- Output buffer
    type t_o_buf is array(0 to MAX_DIM-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal o_buf : t_o_buf;
    
    -- Intermediate buffer
    type t_i_buf is array(0 to 3) of signed(2*DATA_WIDTH-1 downto 0);
    signal i_buf : t_i_buf; 
    attribute use_dsp : string;
    attribute use_dsp of i_buf : signal is "yes";
    signal acc_buf : signed(2*DATA_WIDTH-1 downto 0);
    attribute use_dsp of acc_buf : signal is "yes";
    
    -- Control signals
    signal step_mac : integer;
    signal rounded_dim : integer;

    -- Activation Functions
    signal act_x_0 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal act_y_0 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal act_x_1 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal act_y_1 : std_logic_vector(DATA_WIDTH-1 downto 0);
    
    -- Multiplier
    signal A1 : signed(DATA_WIDTH-1 downto 0);
    signal B1 : signed(DATA_WIDTH-1 downto 0);
    signal Q1 : signed(2*DATA_WIDTH-1 downto 0);
    signal A2 : signed(DATA_WIDTH-1 downto 0);
    signal B2 : signed(DATA_WIDTH-1 downto 0);
    signal Q2 : signed(2*DATA_WIDTH-1 downto 0);
    signal A3 : signed(DATA_WIDTH-1 downto 0);
    signal B3 : signed(DATA_WIDTH-1 downto 0);
    signal Q3 : signed(2*DATA_WIDTH-1 downto 0);
    signal A4 : signed(DATA_WIDTH-1 downto 0);
    signal B4 : signed(DATA_WIDTH-1 downto 0);
    signal Q4 : signed(2*DATA_WIDTH-1 downto 0);
    
    -- Procedures
    procedure mac(
        signal r_data_inp_buf : in bus_array_4_8;
        signal r_data_wt_buf : in bus_array_4_8;
        signal i_buf : out t_i_buf 
    ) is
    begin
        for ii in 0 to 3 loop
            i_buf(ii) <= (signed(r_data_inp_buf(ii)) * signed(r_data_wt_buf(ii)));
        end loop;
    end procedure;
    
    -- Functions
    function cut_down(x: in signed(2*DATA_WIDTH-1 downto 0))return signed is
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin

        TEMP2 := x(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP3 := x(FRAC_WIDTH-1 downto 0);
        if TEMP2(DATA_WIDTH-1) = '1' and TEMP3 /= 0 then
            TEMP2 := TEMP2 + 1;
        end if;

        if x>0 and TEMP2<0 then
            TEMP2 := ('0', others => '1');
        elsif x<0 and TEMP2>0 then
            TEMP2 := ('1', others => '0');
        end if;
        return TEMP2;
    end function;
    
    component mutArrayS
        generic (
            DATA_WIDTH : integer := 6
        );
        port (
            A : in signed(DATA_WIDTH-1 downto 0);
            B : in signed(DATA_WIDTH-1 downto 0);
            Q : out signed(2*DATA_WIDTH-1 downto 0)
        );
    end component;
    
begin

    tanh_0 : entity work.tanh_0(rtl)
    port map (
        clock => clk,
        enable => '1',
        x => act_x_0,
        y => act_y_0
    );
    
    tanh_1 : entity work.tanh_1(rtl)
    port map (
        clock => clk,
        enable => '1',
        x => act_x_1,
        y => act_y_1
    );
    
    mult_1 : mutArrayS
    generic map (
        DATA_WIDTH => DATA_WIDTH
    )
    port map(
        A => A1,
        B => B1,
        Q => Q1
    );
    
    mult_2 : mutArrayS
    generic map (
        DATA_WIDTH => DATA_WIDTH
    )
    port map(
        A => A2,
        B => B2,
        Q => Q2
    );
    
    mult_3 : mutArrayS
    generic map (
        DATA_WIDTH => DATA_WIDTH
    )
    port map(
        A => A3,
        B => B3,
        Q => Q3
    );
    
    mult_4 : mutArrayS
    generic map (
        DATA_WIDTH => DATA_WIDTH
    )
    port map(
        A => A4,
        B => B4,
        Q => Q4
    );

    main : process(clk)
    begin
        
        if rising_edge(clk) then
        
            case (s_inf) is
            
                when s_idle =>
                    r_addr_inp_buf <= (others => '0');
                    r_addr_wt_buf <= (others => '0');
                    acc_buf <= (others => '0');
                    step_mac <= 0;
                    done <= '0';
                    if enable = '1' then
                        s_inf <= s_read;
                        rounded_dim <= to_integer(unsigned(std_logic_vector(TO_UNSIGNED(dim + 3, 8)) and not std_logic_vector(TO_UNSIGNED(3,8))));
                        enable_inp_buf <= '1';
                        r_w_inp_buf <= '1';
                        enable_wt_buf <= '1';
                        r_w_wt_buf <= '1';
                    end if;
                    
                when s_read =>
                    s_inf <= s_mac;
                
                when s_mac =>
                    --mac(r_data_inp_buf, r_data_wt_buf, i_buf);
                    A1 <= signed(r_data_inp_buf(0)); 
                    B1 <= signed(r_data_wt_buf(0));
                    A2 <= signed(r_data_inp_buf(1)); 
                    B2 <= signed(r_data_wt_buf(1));
                    A3 <= signed(r_data_inp_buf(2)); 
                    B3 <= signed(r_data_wt_buf(2));
                    A4 <= signed(r_data_inp_buf(3)); 
                    B4 <= signed(r_data_wt_buf(3));
                    step_mac <= step_mac + 4;
                    s_inf <= s_step_mac;
                    
                when s_step_mac =>
                    if step_mac > rounded_dim then
                        s_inf <= s_bias;
                    else 
                        s_inf <= s_read;
                        acc_buf <= acc_buf + Q1 + Q2 + Q3 + Q4; --i_buf(0) + i_buf(1) + i_buf(2) + i_buf(3);
                        r_addr_inp_buf <= std_logic_vector(TO_UNSIGNED(step_mac, X_ADDR_WIDTH));
                        r_addr_wt_buf <= std_logic_vector(TO_UNSIGNED(step_mac, X_ADDR_WIDTH));
                    end if;
                    
                when s_bias =>
                    acc_buf <= acc_buf + (resize(signed(bias), 2*DATA_WIDTH) sll FRAC_WIDTH);
                    s_inf <= s_act;
                
                when s_act =>
                    if current_layer = 0 then
                        act_x_0 <= std_logic_vector(cut_down(acc_buf));
                    elsif current_layer = 1 then
                        act_x_1 <= std_logic_vector(cut_down(acc_buf));
                    end if;
                    s_inf <= s_done;   
                
                when s_done =>
                    if current_layer = 0 then
                        output <= act_y_0; 
                    elsif current_layer = 1 then
                        output <= act_y_1;
                    else
                        output <= std_logic_vector(cut_down(acc_buf));
                    end if;
                    done <= '1';
                    if enable = '0' then
                        s_inf <= s_idle;
                    end if;
            
            end case;
        
        end if;
            
    end process;

end rtl;
