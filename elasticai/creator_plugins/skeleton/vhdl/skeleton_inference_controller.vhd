library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity skeleton_inference_controller is
    generic (
        DATA_IN_WIDTH : integer;
        DATA_IN_DEPTH : integer;
        DATA_OUT_WIDTH : integer;
        DATA_OUT_DEPTH : integer
    );

    port (
        signal clk : in std_logic;
        signal network_enable : in std_logic;
        signal rst : in std_logic;
        signal input_wr_enable : in std_logic;
        signal input_wr_address : in std_logic_vector(15 downto 0);
        signal input_wr_data : in std_logic_vector(7 downto 0);
        signal output_rd_enable : in std_logic;
        signal output_rd_address : in std_logic_vector(15 downto 0);
        signal output_rd_data : out std_logic_vector(7 downto 0);
        signal done : out std_logic
    );
end entity;

architecture rtl of skeleton_inference_controller is
    constant OUT_WORD_WIDTH : natural := get_width_in_bytes(DATA_OUT_WIDTH) * 8;
    constant OUT_NUMS : natural := DATA_OUT_DEPTH * get_width_in_bytes(DATA_OUT_WIDTH);
    constant OUT_WORD_ADDR_WIDTH : natural := log2(fmax(DATA_OUT_DEPTH, 2));
    constant OUT_ADDR_WIDTH : natural := log2(fmax(OUT_NUMS, 2));

    type state_t is (IDLE, RUNNING);

    signal state : state_t := IDLE;

    signal output_bram_write_enable : std_logic := '0';
    signal output_bram_write_address : std_logic_vector(OUT_WORD_ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal output_bram_write_data : std_logic_vector(OUT_WORD_WIDTH - 1 downto 0) := (others => '0');
    signal output_bram_read_address_i : std_logic_vector(OUT_ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal output_bram_read_data : std_logic_vector(7 downto 0);

    signal frame_start : std_logic := '0';
    signal frame_stream_valid : std_logic;
    signal frame_stream_data : std_logic_vector(DATA_IN_WIDTH - 1 downto 0);
    signal frame_stream_done : std_logic;

    signal runner_output_word_valid : std_logic;
    signal runner_output_word_address : std_logic_vector(OUT_WORD_ADDR_WIDTH - 1 downto 0);
    signal runner_output_word_data : std_logic_vector(OUT_WORD_WIDTH - 1 downto 0);
    signal runner_done : std_logic;
    signal frame_done_seen : std_logic := '0';
    signal runner_done_seen : std_logic := '0';

    signal done_i : std_logic := '0';
begin
    done <= done_i;

    output_bram_read_address_i <=
        std_logic_vector(resize(unsigned(output_rd_address), OUT_ADDR_WIDTH));
    output_rd_data <= output_bram_read_data;

    frame_ingress_i : entity work.skeleton_input_adapter(rtl)
        generic map (
            DATA_IN_WIDTH => DATA_IN_WIDTH,
            DATA_IN_DEPTH => DATA_IN_DEPTH
        )
        port map (
            clk => clk,
            rst => rst,
            start => frame_start,
            input_wr_enable => input_wr_enable,
            input_wr_address => input_wr_address,
            input_wr_data => input_wr_data,
            stream_valid => frame_stream_valid,
            stream_data => frame_stream_data,
            stream_done => frame_stream_done
        );

    network_runner_i : entity work.skeleton_network_runner(rtl)
        generic map (
            DATA_IN_WIDTH => DATA_IN_WIDTH,
            DATA_OUT_WIDTH => DATA_OUT_WIDTH,
            DATA_OUT_DEPTH => DATA_OUT_DEPTH
        )
        port map (
            clk => clk,
            rst => rst,
            start => frame_start,
            stream_valid => frame_stream_valid,
            stream_data => frame_stream_data,
            output_word_valid => runner_output_word_valid,
            output_word_address => runner_output_word_address,
            output_word_data => runner_output_word_data,
            done => runner_done
        );

    output_bram_i : entity work.asymmetric_dual_port_bram(rtl)
        generic map (
            WRITE_DATA_WIDTH => OUT_WORD_WIDTH,
            WRITE_ADDRESS_WIDTH => OUT_WORD_ADDR_WIDTH,
            WRITE_SIZE => DATA_OUT_DEPTH,
            READ_DATA_WIDTH => 8,
            READ_ADDRESS_WIDTH => OUT_ADDR_WIDTH,
            READ_SIZE => OUT_NUMS
        )
        port map (
            read_clk => clk,
            read_address => output_bram_read_address_i,
            read_enable => output_rd_enable,
            d_out => output_bram_read_data,
            d_out_valid => open,
            write_clk => clk,
            write_address => output_bram_write_address,
            write_enable => output_bram_write_enable,
            d_in => output_bram_write_data
        );

    process(clk)
    begin
        if rising_edge(clk) then
            output_bram_write_enable <= '0';
            frame_start <= '0';

            if rst = '1' then
                state <= IDLE;
                done_i <= '0';
                frame_done_seen <= '0';
                runner_done_seen <= '0';
            else
                case state is
                    when IDLE =>
                        if network_enable = '1' then
                            done_i <= '0';
                            frame_done_seen <= '0';
                            runner_done_seen <= '0';
                            frame_start <= '1';
                            state <= RUNNING;
                        end if;

                    when RUNNING =>
                        if frame_stream_done = '1' then
                            frame_done_seen <= '1';
                        end if;

                        if runner_done = '1' then
                            runner_done_seen <= '1';
                        end if;

                        if (frame_done_seen = '1' or frame_stream_done = '1') and
                            (runner_done_seen = '1' or runner_done = '1') then
                            done_i <= '1';
                            state <= IDLE;
                        end if;
                end case;

                if runner_output_word_valid = '1' then
                    output_bram_write_enable <= '1';
                    output_bram_write_address <= runner_output_word_address;
                    output_bram_write_data <= runner_output_word_data;
                end if;
            end if;
        end if;
    end process;
end architecture;
