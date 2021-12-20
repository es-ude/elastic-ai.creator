library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
use ieee.math_real.all;                 -- for the ceiling and log constant calculation function

entity lstm_common_gate_tb is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8;
        VECTOR_LEN_WIDTH : integer := 4
        );
    port ( clk: out std_logic);
end entity ; -- lstm_common_gate_tb

architecture arch of lstm_common_gate_tb is

    ------------------------------------------------------------
    -- Testbench Data Type
    ------------------------------------------------------------
    type RAM_ARRAY is array (0 to 9 ) of signed(DATA_WIDTH-1 downto 0);

    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------
    signal clk_period : time := 2 ps;
    signal clock : std_logic;
    signal reset, ready : std_logic:='0';
    signal X_MEM : RAM_ARRAY :=(others=>(others=>'0'));
    signal W_MEM : RAM_ARRAY:=(others=>(others=>'0'));
    signal x, w, y, b : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
    signal vector_len : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0');
    signal idx : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0');

    ------------------------------------------------------------
    -- Declare Components for testing
    ------------------------------------------------------------
    component lstm_common_gate is
        generic (
                DATA_WIDTH : integer := 16;
                FRAC_WIDTH : integer := 8;
                VECTOR_LEN_WIDTH : integer := 4
            );
        port (
            reset : in std_logic;
            clk : in std_logic;
            x : in signed(DATA_WIDTH-1 downto 0);
            w : in signed(DATA_WIDTH-1 downto 0);
            b : in signed(DATA_WIDTH-1 downto 0);
            vector_len : in unsigned(VECTOR_LEN_WIDTH-1 downto 0);
            idx : out unsigned(VECTOR_LEN_WIDTH-1 downto 0);
            ready : out std_logic;
            y : out signed(DATA_WIDTH-1 downto 0)
        );
    end component lstm_common_gate;

begin

    clock_process : process
    begin
        clock <= '0';
        wait for clk_period/2;
        clock <= '1';
        wait for clk_period/2;
    end process; -- clock_process

    uut: lstm_common_gate
    port map (
        reset => reset,
        clk => clock,
        x => x,
        w => w,
        b => b,
        vector_len => vector_len,
        idx => idx,
        ready => ready,
        y => y
    );

    x <= X_MEM(to_integer(idx));
    w <= W_MEM(to_integer(idx));

    test_process: process is
    begin
        Report "======Simulation start======" severity Note;

        vector_len <= to_unsigned(10, VECTOR_LEN_WIDTH);

        X_MEM <= (x"0013",x"0000",x"0010",x"0013",x"000c",x"0005",x"0005",x"0013",x"0004",x"0002");
        W_MEM <= (x"0011",x"0018",x"0000",x"000d",x"0014",x"000f",x"0012",x"0007",x"0017",x"0012");
        b <= x"008a";

        reset <= '1';
        wait for 2*clk_period;
        wait until clock = '0';
        reset <= '0';
        wait until ready = '1';
        
        report "expected output is 142, value of 'y' is " & integer'image(to_integer(signed(y)));
        assert y = 142 report "The 0. test case fail" severity error;
        reset <= '1';
        wait for 1*clk_period;

        X_MEM <= (x"0014",x"000d",x"0017",x"0008",x"0002",x"0007",x"0002",x"0015",x"0001",x"0010");
        W_MEM <= (x"000e",x"0014",x"0005",x"0015",x"0009",x"0013",x"0007",x"0016",x"0008",x"0004");
        b <= x"0064";

        reset <= '1';
        wait for 2*clk_period;
        wait until clock = '0';
        reset <= '0';
        wait until ready = '1';
        
        report "expected output is 105, value of 'y' is " & integer'image(to_integer(signed(y)));
        assert y = 105 report "The 1. test case fail" severity error;
        reset <= '1';
        wait for 1*clk_period;

        X_MEM <= (x"000f",x"0017",x"000d",x"000f",x"0001",x"0009",x"0002",x"0007",x"0008",x"0013");
        W_MEM <= (x"0001",x"000a",x"0008",x"0010",x"0008",x"0001",x"0016",x"0013",x"0016",x"000a");
        b <= x"009b";

        reset <= '1';
        wait for 2*clk_period;
        wait until clock = '0';
        reset <= '0';
        wait until ready = '1';
        
        report "expected output is 159, value of 'y' is " & integer'image(to_integer(signed(y)));
        assert y = 159 report "The 2. test case fail" severity error;
        reset <= '1';
        wait for 1*clk_period;

        X_MEM <= (x"000c",x"0007",x"0001",x"0019",x"0008",x"000c",x"0019",x"000b",x"0008",x"000d");
        W_MEM <= (x"000e",x"0015",x"0001",x"000b",x"0014",x"0012",x"000f",x"0000",x"0008",x"000e");
        b <= x"004c";

        reset <= '1';
        wait for 2*clk_period;
        wait until clock = '0';
        reset <= '0';
        wait until ready = '1';
        
        report "expected output is 82, value of 'y' is " & integer'image(to_integer(signed(y)));
        assert y = 82 report "The 3. test case fail" severity error;
        reset <= '1';
        wait for 1*clk_period;

        X_MEM <= (x"0005",x"0013",x"0002",x"0013",x"000c",x"000f",x"0003",x"0004",x"0010",x"0001");
        W_MEM <= (x"0006",x"000d",x"0005",x"0009",x"0017",x"0017",x"000e",x"000d",x"0000",x"0019");
        b <= x"0092";

        reset <= '1';
        wait for 2*clk_period;
        wait until clock = '0';
        reset <= '0';
        wait until ready = '1';
        
        report "expected output is 150, value of 'y' is " & integer'image(to_integer(signed(y)));
        assert y = 150 report "The 4. test case fail" severity error;
        reset <= '1';
        wait for 1*clk_period;


        -- if there is no error message, that means all test case are passed.
        report "======Simulation Success======" severity Note;
        report "Please check the output message." severity Note;

        -- wait forever
        wait;

    end process; -- test_process

end arch ; -- arch

