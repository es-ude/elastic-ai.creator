LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ${name} is
    generic (
        DATA_WIDTH  : integer := ${data_width};    -- that fixed point data has 16bits
        FRAC_WIDTH  : integer := ${frac_width};     -- and 8bits is for the factional part
        IN_ADDR_WIDTH  : integer := ${in_addr_width};     -- supports up to 16 inputs

        LSTM_INPUTS  : integer := ${input_size};    -- supports up to 16 inputs
        LSTM_CELL_HIDDEN_SIZE : integer := ${hidden_size};
        LSTM_CELL_X_H_ADDR_WIDTH : integer := ${x_h_addr_width};
        LSTM_CELL_HIDDEN_ADDR_WIDTH : integer := ${hidden_addr_width};
        LSTM_CELL_W_ADDR_WIDTH : integer := ${w_addr_width};

        LINEAR_IN_FEATURE : integer := ${linear_in_features};
        LINEAR_OUT_FEATURE : integer := ${linear_out_features}
    );
    port (
        clock     : in std_logic;
        enable    : in std_logic;    -- start computing when it is '1'
        x     : in std_logic_vector(DATA_WIDTH-1 downto 0);
        addr_in   : in std_logic_vector(IN_ADDR_WIDTH-1 downto 0);
        x_we     : in std_logic;
        done : out std_logic;
        d_out : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end;

architecture rtl of ${name} is

    type INPUT_ARRAY is array (0 to 2**IN_ADDR_WIDTH-1) of signed(DATA_WIDTH-1 downto 0);
    shared variable input_buffer : INPUT_ARRAY := (x"10", x"10", x"00", others=>to_signed(0,DATA_WIDTH));
    type TYPE_STATE is (s_reset, s_lstm, s_linear, s_done);
    signal network_state : TYPE_STATE := s_reset;

    signal lstm_input_addr : unsigned(IN_ADDR_WIDTH-1 downto 0);
    signal i_lstm_cell_done : std_logic;
    signal i_lstm_cell_reset: std_logic;
    signal i_lstm_cell_enable: std_logic;
    signal i_lstm_cell_zero_state : std_logic;
    signal i_lstm_cell_x_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_lstm_cell_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_lstm_cell_out_addr : std_logic_vector(LSTM_CELL_HIDDEN_ADDR_WIDTH-1 downto 0);
    signal i_lstm_cell_out_en : std_logic;

    signal i_linear_enable : std_logic;
    signal i_linear_read_addr : std_logic_vector(LSTM_CELL_HIDDEN_ADDR_WIDTH-1 downto 0);
    signal i_linear_x_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_linear_w_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_linear_b_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_linear_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal i_linear_done : std_logic;

    type t_y_array is array (0 to 31) of integer;
    shared variable test_x_ram : t_y_array:=(-7,-4,-8,4,5,5,-10,0,8,8,12,-8,1,-6,-13,-5,0,4,9,5,0,0,0,0,0,0,0,0,0,0,0,0); --27
begin

    INPUT_CONFIG: process(clock, x_we)
    begin
        if rising_edge(clock) then
            if x_we = '1' then
                input_buffer(to_integer(unsigned(addr_in))) := signed(x);
            end if;
        end if;
    end process; -- INPUT_CONFIG

    INPUT_READ: process(clock, enable)
    begin
        if falling_edge(clock) then
            if enable ='1' then
                i_lstm_cell_x_data <= std_logic_vector(input_buffer(to_integer(lstm_input_addr)));
            end if;
        end if;
    end process; -- INPUT_READ

    NETWORK_CTRL: process(clock)
    variable i_lstm_cell_itter : integer := 0;
    begin
        if rising_edge(clock) then
            if enable = '0' then
                i_lstm_cell_out_en<='0';
                network_state <= s_reset;
                i_lstm_cell_reset <= '1';
                i_lstm_cell_enable <= '0';
                done <='0';
                i_linear_enable <='0';
            else
                if network_state = s_reset then
                    i_lstm_cell_itter := 0;
                    network_state <= s_lstm;
                    i_lstm_cell_zero_state <= '1';
                else
                    if network_state = s_lstm then
                        i_lstm_cell_enable <= '1';
                        if i_lstm_cell_reset='1' then

                            i_lstm_cell_reset <= '0';
                        else
                            if i_lstm_cell_done ='1' then
                                i_lstm_cell_zero_state <= '0';
                                if i_lstm_cell_itter = LSTM_INPUTS-1 then
                                    network_state <= s_linear;
                                    i_lstm_cell_out_en<='1';
                                else
                                    i_lstm_cell_itter := i_lstm_cell_itter + 1;
                                    i_lstm_cell_reset <= '1';
                                end if;
                            end if;
                        end if;

                    else
                        if network_state = s_linear then
                            if i_linear_enable='0' then
                                i_linear_enable <='1';
                            else
                                if i_linear_done='1' then
                                    network_state <= s_done;
                                    done <='1';
                                    i_lstm_cell_out_en<='0';
                                end if;
                            end if;

                        end if;
                    end if;
                end if;
            end if;
            lstm_input_addr <= to_unsigned(i_lstm_cell_itter, IN_ADDR_WIDTH);
        end if;
    end process; -- NETWORK_CTRL


    i_lstm_cell: entity work.${lstm_cell_name}(rtl)

    port map (
        clock => clock,
        reset => i_lstm_cell_reset,
        enable => i_lstm_cell_enable,
        zero_state => i_lstm_cell_zero_state,
        x_data => i_lstm_cell_x_data,
        done => i_lstm_cell_done,
        h_out_en => i_lstm_cell_out_en,
        h_out_data => i_lstm_cell_out_data,
        h_out_addr => i_lstm_cell_out_addr
    );

    i_lstm_cell_out_addr <= i_linear_read_addr;

    i_linear_x_data <= i_lstm_cell_out_data;

    i_linear : entity work.${linear_name}(rtl)
    port map (
        enable => i_linear_enable,
        clock => clock,
        x_address => i_linear_read_addr,
        x => i_linear_x_data,
        y_address => (others=>'0'),
        y => i_linear_out_data,

        done => i_linear_done
    );

    d_out <= i_linear_out_data;



end architecture rtl;
