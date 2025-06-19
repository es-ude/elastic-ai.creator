library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.padding_pkg.all;

entity buffered_network_wrapper is
  generic (
    DATA_IN_WIDTH : integer;
    DATA_IN_DEPTH : integer;
    DATA_OUT_WIDTH : integer;
    DATA_OUT_DEPTH : integer
    );

  port (
    signal clk : in std_logic;
    signal valid_in : in std_logic;
    signal rst : in std_logic;
    signal d_in : in std_logic_vector(8 - 1 downto 0);
    signal d_out : out std_logic_vector(8 - 1 downto 0);
    signal ready_out : out std_logic;
    signal ready_in : in std_logic;
    signal valid_out : out std_logic
    );
end entity;


architecture rtl of buffered_network_wrapper is
    signal ai_input_v : std_logic_vector(DATA_IN_WIDTH * DATA_IN_DEPTH - 1 downto 0);
    signal ai_output_v : std_logic_vector(DATA_OUT_WIDTH * DATA_OUT_DEPTH - 1 downto 0);


    signal valid_out_sr : std_logic;
    signal valid_in_internal : std_logic;


    signal has_been_enabled: std_logic := '0';

    signal padder_valid_in : std_logic := '0';
    signal padder_valid_out : std_logic := '0';

    signal remover_valid_out : std_logic := '0';

  begin


    check_if_has_been_enabled:
    process (clk, rst) is begin
        -- forget that we saw enabled when we see a reset
        has_been_enabled <= (not rst and has_been_enabled and remover_valid_out);
        if rising_edge(clk) then
            if ready_in = '1' then
                has_been_enabled <= remover_valid_out;
            end if;
        end if;
    end process;

    -- Without this, clients would always have to add padding
    -- data until they observe valid_out.
    valid_in_internal <= valid_in and not valid_out_sr;


    valid_out <= valid_out_sr;

    unpadder : entity work.dynamic_padding_remover(rtl)
        generic map (
            DATA_WIDTH => DATA_IN_WIDTH
            )
        port map (
            d_in => d_in,
            d_out => ai_input_v,
            valid_in => valid_in,
            valid_out => remover_valid_out,
            ready_in => '1',
            ready_out => ready_out,
            clk => clk,
            rst => rst
        );


    network_i: entity work.network(rtl)
    port map (
        clk => clk,
        valid_in => has_been_enabled,
        d_in => ai_input_v ,
        d_out => ai_output_v,
        rst => rst,
        valid_out => valid_out
    );

    padder : entity work.dynamic_padder(rtl)
        generic map (
            DATA_WIDTH => DATA_OUT_WIDTH
            )
        port map (
            d_in => ai_output_v,
            d_out => d_out,
            rst => rst,
            clk => clk,
            valid_in => padder_valid_in,
            valid_out => padder_valid_out,
            ready_in => ready_in,
            ready_out => ready_out
        );



end architecture;
