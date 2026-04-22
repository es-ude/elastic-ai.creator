library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.padding_pkg.all;

entity buffered_network_wrapper is
  generic (
    DATA_IN_WIDTH : integer;
    DATA_IN_DEPTH : integer;
    DATA_OUT_WIDTH : integer;
    DATA_OUT_DEPTH : integer;
    KERNEL_SIZE : integer;
    STRIDE : integer
    );

  port (
    signal clk : in std_logic;
    signal valid_in : in std_logic;
    signal rst : in std_logic;
    signal d_in : in std_logic_vector(DATA_IN_DEPTH * size_in_bytes(DATA_IN_WIDTH) * 8 - 1 downto 0);
    signal d_out : out std_logic_vector(DATA_OUT_DEPTH * size_in_bytes(DATA_OUT_WIDTH) * 8 - 1 downto 0);
    signal valid_out : out std_logic
    );
end entity;


architecture rtl of buffered_network_wrapper is
    signal ai_input_v : std_logic_vector(DATA_IN_WIDTH * DATA_IN_DEPTH - 1 downto 0);
    signal ai_input_window : std_logic_vector(KERNEL_SIZE - 1 downto 0);
    signal ai_output_v : std_logic_vector(DATA_OUT_WIDTH * DATA_OUT_DEPTH - 1 downto 0);
    signal d_out_network : std_logic_vector(DATA_OUT_WIDTH - 1 downto 0);


    signal valid_in_network : std_logic;
    signal valid_out_network : std_logic;
    signal valid_in_sr : std_logic;
    signal valid_out_sr : std_logic;
    signal valid_out_sw : std_logic;
    signal rst_network : std_logic;
    signal valid_in_internal : std_logic;

    signal delayed_enable: std_logic;

    signal has_been_enabled: std_logic := '0';

  begin


    check_if_has_been_enabled:
    process (clk, rst) is begin
        -- forget that we saw enabled when we see a reset
        has_been_enabled <= (not rst and has_been_enabled);
        if rising_edge(clk) then
            if valid_in = '1' then
                has_been_enabled <= '1';
            end if;
        end if;
    end process;

    -- Without this, clients would always have to add padding
    -- data until they observe valid_out.
    valid_in_internal <= valid_in or (has_been_enabled and not valid_out_sr);

    -- feed the shift reg until full
    valid_in_sr <= valid_out_network and not valid_out_sr;

    valid_out <= valid_out_sr;
    valid_in_network <= valid_out_sw;

    unpadder : entity work.padding_remover(rtl)
        generic map (
            DATA_WIDTH => DATA_IN_WIDTH,
            DATA_DEPTH => DATA_IN_DEPTH
            )
        port map (
            d_in => d_in,
            d_out => ai_input_v
        );


  sliding_window_i : entity work.sliding_window(rtl)
        generic map (
           INPUT_WIDTH => DATA_IN_WIDTH * DATA_IN_DEPTH,
           OUTPUT_WIDTH => KERNEL_SIZE,
           STRIDE => STRIDE
        )
        port map (
            clk => clk,
            d_in => ai_input_v,
            d_out => ai_input_window,
            valid_in => valid_in,
            valid_out => valid_out_sw,
            rst => rst
        );


    network_i: entity work.network(rtl)
    port map (
        clk => clk,
        valid_in => valid_in_network,
        d_in => ai_input_window,
        d_out => d_out_network,
        rst => rst,
        valid_out => valid_out_network
    );

    shift_reg_i : entity work.shift_register(rtl)
        generic map (
            DATA_WIDTH => DATA_OUT_WIDTH,
            NUM_POINTS => DATA_OUT_DEPTH
        )
        port map (
            clk => clk,
            valid_in => valid_in_sr,
            valid_out => valid_out_sr,
            d_in => d_out_network,
            d_out => ai_output_v,
            rst => rst
        );

      padder : entity work.padder(rtl)
        generic map (
            DATA_WIDTH => DATA_OUT_WIDTH,
            DATA_DEPTH => DATA_OUT_DEPTH
            )
        port map (
            d_in => ai_output_v,
            d_out => d_out
        );



end architecture;
