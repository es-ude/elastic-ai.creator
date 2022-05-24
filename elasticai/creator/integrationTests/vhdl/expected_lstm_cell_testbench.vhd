library ieee, work;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.lstm_common.all;

entity lstm_cell_tb is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8;

        INPUT_SIZE : integer := 5;
        HIDDEN_SIZE : integer := 20;

        X_H_ADDR_WIDTH : integer := 5;
        HIDDEN_ADDR_WIDTH : integer := 5;
        W_ADDR_WIDTH : integer := 9
    );
    port (
        clk : out std_logic
    );
end entity lstm_cell_tb;

architecture rtl of lstm_cell_tb is
    signal clk_period : time := 10 ns;
    signal clock : std_logic;
    signal enable : std_logic;

    signal reset: std_logic:='0';

    signal x_config_en: std_logic:='0';
    signal x_config_data:std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0');
    signal x_config_addr:std_logic_vector(X_H_ADDR_WIDTH-1 downto 0) :=(others=>'0');

    signal h_config_en: std_logic:='0';
    signal h_config_data:std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0');
    signal h_config_addr:std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0) :=(others=>'0');

    signal c_config_en: std_logic:='0';
    signal c_config_data:std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0');
    signal c_config_addr:std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0) :=(others=>'0');
    signal done :  std_logic:='0';

    signal h_out_en : std_logic;
    signal h_out_addr : std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0) :=(others=>'0');
    signal h_out_data : std_logic_vector(DATA_WIDTH-1 downto 0):=(others=>'0');

    type X_H_ARRAY is array (0 to 511) of signed(16-1 downto 0);
    type C_ARRAY is array (0 to 511) of signed(16-1 downto 0);

    signal test_x_h_data : X_H_ARRAY := (x"018a",x"ffb5",x"fdd3",x"0091",x"feeb",x"0099",x"fe72",x"ffa9",x"01da",x"ffc9",x"ff42",x"0090",x"0042",x"ffd4",x"ff53",x"00f0",x"007d",x"0134",x"0015",x"fecd",x"ffff",x"ff7c",x"ffb2",x"fe6c",x"01b4",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",others=>(others=>'0'));

    signal test_c_data : C_ARRAY := (x"0034",x"ff8d",x"ff6e",x"ff72",x"fee0",x"ffaf",x"fee9",x"ffeb",x"ffe9",x"00af",x"ff2a",x"0000",x"ff40",x"002f",x"009f",x"00a3",x"ffc2",x"024d",x"fe1f",x"fff4",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",x"0000",others=>(others=>'0'));

    procedure send_x_h_data (
        addr_in : in std_logic_vector(X_H_ADDR_WIDTH-1 downto 0);
        data_in : in std_logic_vector(DATA_WIDTH-1 downto 0);
        signal clock : in std_logic;
        signal wr : out std_logic;
        signal addr_out : out std_logic_vector(X_H_ADDR_WIDTH-1 downto 0);
        signal data_out : out std_logic_vector(DATA_WIDTH-1 downto 0)) is
        begin
            addr_out <= addr_in;
            data_out <= data_in;
            wait until clock='0';
            wr <= '1';
            wait until clock='1';
            wait until clock='0';
            wr <= '0';
            wait until clock='1';
    end send_x_h_data;

    procedure send_c_data (
        addr_in : in std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0);
        data_in : in std_logic_vector(DATA_WIDTH-1 downto 0);
        signal clock : in std_logic;
        signal wr : out std_logic;
        signal addr_out : out std_logic_vector(HIDDEN_ADDR_WIDTH-1 downto 0);
        signal data_out : out std_logic_vector(DATA_WIDTH-1 downto 0)) is
        begin
            addr_out <= addr_in;
            data_out <= data_in;
            wait until clock='0';
            wr <= '1';
            wait until clock='1';
            wait until clock='0';
            wr <= '0';
            wait until clock='1';
    end send_c_data;

begin
    clock_process: process
    begin
        clock <= '0';
        wait for clk_period/2;
        clock <= '1';
        wait for clk_period/2;
    end process clock_process;

    uut: entity work.lstm_cell(rtl)
    generic map (
        DATA_WIDTH => DATA_WIDTH,
        FRAC_WIDTH => FRAC_WIDTH,
        INPUT_SIZE => INPUT_SIZE,
        HIDDEN_SIZE => HIDDEN_SIZE,

        X_H_ADDR_WIDTH => X_H_ADDR_WIDTH,
        HIDDEN_ADDR_WIDTH => HIDDEN_ADDR_WIDTH,
        W_ADDR_WIDTH => W_ADDR_WIDTH
    )
    port map (
        clock => clock,
        reset => reset,
        enable => enable,
        x_h_we => x_config_en,
        x_h_data => x_config_data,
        x_h_addr => x_config_addr,

        c_we => c_config_en,
        c_data_in => c_config_data,
        c_addr_in => c_config_addr,

        done => done,

        h_out_en => h_out_en,
        h_out_data => h_out_data,
        h_out_addr => h_out_addr

    );

    clk <= clock;

    test_process: process
        begin
            report "======Simulation Start======" severity Note;

            reset <= '1';
            h_out_en <= '0';
            wait for 2*clk_period;
            reset <= '0';

            for ii in 0 to 24 loop send_x_h_data(std_logic_vector(to_unsigned(ii, X_H_ADDR_WIDTH)), std_logic_vector(test_x_h_data(ii)), clock, x_config_en, x_config_addr, x_config_data);
                wait for 10 ns;
            end loop;

            for ii in 0 to 19 loop send_c_data(std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH)), std_logic_vector(test_c_data(ii)), clock, c_config_en, c_config_addr, c_config_data);
                wait for 10 ns;
            end loop;

            enable <= '1';
            wait until done = '1';
            wait for 1*clk_period;
            enable <= '0';

            -- reference h_out: [34, -80, -32, -28, -88, 11, -60, 6, -16, 18, -32, 46, -77, 15, 70, 27, 13, 112, -156, 3];
            for ii in 0 to 19 loop h_out_addr <= std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH));
                h_out_en <= '1';
                wait for 2*clk_period;
                report "The value of h_out(" & integer'image(ii)& ") is " & integer'image(to_integer(signed(h_out_data)));
            end loop;

            wait for 10*clk_period;
            report "======Simulation Success======" severity Note;
            report "Please check the output message." severity Note;

            wait;

    end process test_process;

end architecture rtl;
