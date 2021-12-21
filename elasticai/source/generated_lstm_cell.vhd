library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
LIBRARY work;
use work.all;

entity lstm_cell is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8
    );
    port (
        x : in signed(DATA_WIDTH-1 downto 0);
        c_in : in signed(DATA_WIDTH-1 downto 0);
        h_in : in signed(DATA_WIDTH-1 downto 0);
        c_out : out signed(DATA_WIDTH-1 downto 0);
        h_out : out signed(DATA_WIDTH-1 downto 0)
    );
end entity lstm_cell;

architecture lstm_cell_rtl of lstm_cell is

    signal wii : signed(DATA_WIDTH-1 downto 0) := X"ffff"; -- W_ii;
    signal wif : signed(DATA_WIDTH-1 downto 0) := X"0089"; -- W_if;
    signal wig : signed(DATA_WIDTH-1 downto 0) := X"ff2e"; -- W_ig;
    signal wio : signed(DATA_WIDTH-1 downto 0) := X"ff44"; -- W_io;
    signal whi : signed(DATA_WIDTH-1 downto 0) := X"ff9e"; -- W_hi;
    signal whf : signed(DATA_WIDTH-1 downto 0) := X"0044"; -- W_hf;
    signal whg : signed(DATA_WIDTH-1 downto 0) := X"fffb"; -- W_hg;
    signal who : signed(DATA_WIDTH-1 downto 0) := X"00ca"; -- W_ho;
    signal bi : signed(DATA_WIDTH-1 downto 0) := X"fef5"; -- b_ii + b_hi;
    signal bf : signed(DATA_WIDTH-1 downto 0) := X"ff9b"; -- b_if + b_hf;
    signal bg : signed(DATA_WIDTH-1 downto 0) := X"ff4a"; -- b_ig + b_hg;
    signal bo : signed(DATA_WIDTH-1 downto 0) := X"ffd8"; -- b_io + b_ho;

-- Intermediate results
-- Input gate without/with activation
-- i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi})
signal i_wo_activation : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
        signal i : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- Forget gate without/with activation
-- f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf})
signal f_wo_activation : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal f : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- Cell gate without/with activation
-- g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg})
signal g_wo_activation : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal g : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- Output gate without/with activation
-- o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho})
signal o_wo_activation : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
        signal o : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- new_cell_state without/with activation
-- c' = f * c + i * g 
signal c_new : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal c_new_wo_activation : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- h' = o * \tanh(c')
signal h_new : signed(DATA_WIDTH-1 downto 0):=(others=>'0');


component mac_async is
    generic (
        DATA_WIDTH : integer := DATA_WIDTH;
        FRAC_WIDTH : integer := FRAC_WIDTH
    );
    port (
        x1 : in signed(DATA_WIDTH-1 downto 0);
        x2 : in signed(DATA_WIDTH-1 downto 0);
        w1 : in signed(DATA_WIDTH-1 downto 0);
        w2 : in signed(DATA_WIDTH-1 downto 0);
        b : in signed(DATA_WIDTH-1 downto 0);
        y : out signed(DATA_WIDTH-1 downto 0)
    );
end component mac_async;

component sigmoid is
    generic (
        DATA_WIDTH : integer := DATA_WIDTH;
        FRAC_WIDTH : integer := FRAC_WIDTH
    );
    port (
        x : in signed(DATA_WIDTH-1 downto 0);
        y : out signed(DATA_WIDTH-1 downto 0)
    );
end component sigmoid;

component tanh is
    generic (
        DATA_WIDTH : integer := DATA_WIDTH;
        FRAC_WIDTH : integer := FRAC_WIDTH
    );
    port (
        x : in signed(DATA_WIDTH-1 downto 0);
        y : out signed(DATA_WIDTH-1 downto 0)
    );
end component tanh;

begin

    c_out <= c_new_wo_activation;
    h_out <= h_new;

    FORGET_GATE_MAC: mac_async
    port map (
        x1 => x,
        x2 => h_in,
        w1 => wif,
        w2 => whf,
        b => bf,
        y => f_wo_activation
    );

    FORGET_GATE_SIGMOID: sigmoid
    port map (
        f_wo_activation,
        f
    );

    INPUT_GATE_MAC: mac_async
    port map (
        x1 => x,
        x2 => h_in,
        w1 => wii,
        w2 => whi,
        b => bi,
        y => i_wo_activation
    );

    INPUT_GATE_SIGMOID: sigmoid
    port map (
        i_wo_activation,
        i
    );

    CELL_GATE_MAC: mac_async
    port map (
        x1 => x,
        x2 => h_in,
        w1 => wig,
        w2 => whg,
        b => bg,
        y => g_wo_activation
    );

    CELL_GATE_TANH: tanh
    port map (
        g_wo_activation,
        g
    );

    NEW_CELL_STATE_MAC: mac_async
    port map (
        x1 => f,
        x2 => i,
        w1 => c_in,
        w2 => g,
        b => (others=>'0'),
        y => c_new_wo_activation
    );

    NEW_CELL_STATE_TANH: tanh
    port map (
        c_new_wo_activation,
        c_new
    );

    MAC_ASYNC_4: mac_async
    port map (
        x1 => x,
        x2 => h_in,
        w1 => wio,
        w2 => who,
        b => bo,
        y => o_wo_activation
    );

    SIGMOID_1: sigmoid
    port map (
        x => o_wo_activation,
        y => o
    );

    H_OUT_PROCESS: process(o,c_new)
    begin
        h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    end process;

end architecture lstm_cell_rtl ; -- lstm_cell_rtl

