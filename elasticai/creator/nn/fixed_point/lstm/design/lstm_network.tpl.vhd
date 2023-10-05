LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library xil_defaultlib;

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
    signal ${lstm_cell_instance_name}_done : std_logic;
    signal ${lstm_cell_instance_name}_reset: std_logic;
    signal ${lstm_cell_instance_name}_enable: std_logic;
    signal ${lstm_cell_instance_name}_zero_state : std_logic;
    signal ${lstm_cell_instance_name}_x_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ${lstm_cell_instance_name}_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ${lstm_cell_instance_name}_out_addr : std_logic_vector(LSTM_CELL_HIDDEN_ADDR_WIDTH-1 downto 0);
    signal ${lstm_cell_instance_name}_out_en : std_logic;

    signal ${linear_instance_name}_enable : std_logic;
    signal ${linear_instance_name}_read_addr : std_logic_vector(LSTM_CELL_HIDDEN_ADDR_WIDTH-1 downto 0);
    signal ${linear_instance_name}_x_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ${linear_instance_name}_w_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ${linear_instance_name}_b_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ${linear_instance_name}_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal ${linear_instance_name}_done : std_logic;

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
                ${lstm_cell_instance_name}_x_data <= std_logic_vector(input_buffer(to_integer(lstm_input_addr)));
            end if;
        end if;
    end process; -- INPUT_READ

    NETWORK_CTRL: process(clock)
    variable ${lstm_cell_instance_name}_itter : integer := 0;
    begin
        if rising_edge(clock) then
            if enable = '0' then
                ${lstm_cell_instance_name}_out_en<='0';
                network_state <= s_reset;
                ${lstm_cell_instance_name}_reset <= '1';
                ${lstm_cell_instance_name}_enable <= '0';
                done <='0';
                ${linear_instance_name}_enable <='0';
            else
                if network_state = s_reset then
                    ${lstm_cell_instance_name}_itter := 0;
                    network_state <= s_lstm;
                    ${lstm_cell_instance_name}_zero_state <= '1';
                else
                    if network_state = s_lstm then
                        ${lstm_cell_instance_name}_enable <= '1';
                        if ${lstm_cell_instance_name}_reset='1' then

                            ${lstm_cell_instance_name}_reset <= '0';
                        else
                            if ${lstm_cell_instance_name}_done ='1' then
                                ${lstm_cell_instance_name}_zero_state <= '0';
                                if ${lstm_cell_instance_name}_itter = LSTM_INPUTS-1 then
                                    network_state <= s_linear;
                                    ${lstm_cell_instance_name}_out_en<='1';
                                else
                                    ${lstm_cell_instance_name}_itter := ${lstm_cell_instance_name}_itter + 1;
                                    ${lstm_cell_instance_name}_reset <= '1';
                                end if;
                            end if;
                        end if;

                    else
                        if network_state = s_linear then
                            if ${linear_instance_name}_enable='0' then
                                ${linear_instance_name}_enable <='1';
                            else
                                if ${linear_instance_name}_done='1' then
                                    network_state <= s_done;
                                    done <='1';
                                    ${lstm_cell_instance_name}_out_en<='0';
                                end if;
                            end if;

                        end if;
                    end if;
                end if;
            end if;
            lstm_input_addr <= to_unsigned(${lstm_cell_instance_name}_itter, IN_ADDR_WIDTH);
        end if;
    end process; -- NETWORK_CTRL


    i_${lstm_cell_instance_name}: entity work.${lstm_cell_name}(rtl)

    port map (
        clock => clock,
        reset => ${lstm_cell_instance_name}_reset,
        enable => ${lstm_cell_instance_name}_enable,
        zero_state => ${lstm_cell_instance_name}_zero_state,
        x_data => ${lstm_cell_instance_name}_x_data,
        done => ${lstm_cell_instance_name}_done,
        h_out_en => ${lstm_cell_instance_name}_out_en,
        h_out_data => ${lstm_cell_instance_name}_out_data,
        h_out_addr => ${lstm_cell_instance_name}_out_addr
    );

    ${lstm_cell_instance_name}_out_addr <= ${linear_instance_name}_read_addr;

    ${linear_instance_name}_x_data <= ${lstm_cell_instance_name}_out_data;

--    ${linear_instance_name}_x_data <= std_logic_vector(to_signed(test_x_ram(to_integer(unsigned(${linear_instance_name}_read_addr))),d_out'length));

    i_${linear_instance_name} : entity work.${linear_name}(rtl)
    port map (
        enable => ${linear_instance_name}_enable,
        clock => clock,
        x_address => ${linear_instance_name}_read_addr,
        x => ${linear_instance_name}_x_data,
        y_address => (others=>'0'),
        y => ${linear_instance_name}_out_data,

        done => ${linear_instance_name}_done
    );

    d_out <= ${linear_instance_name}_out_data;

--    i_output_sigmoid : entity work.fp_hard_sigmoid_2(rtl)
--    port map (
--        enable =>${linear_instance_name}_done,
--        clock => clock,
--        x => ${linear_instance_name}_out_data,
--        y => d_out
--    );


end architecture rtl;
