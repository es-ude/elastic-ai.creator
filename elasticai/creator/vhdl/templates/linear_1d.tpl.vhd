LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library {work_library_name};
use {work_library_name}.lstm_common.all;

entity linear_1d is
  generic (
    ADDR_WIDTH :integer := {addr_width};
    DATA_WIDTH :integer := {data_width};
    IN_FEATURE_COUNT :integer := {in_feature_count};
    OUT_FEATURE_COUNT :integer := {out_feature_count}
  );
  port (
    clock : in std_logic;
    addr  : out std_logic_vector(ADDR_WIDTH-1 downto 0);
    enable : in std_logic;
    x_in  : in std_logic_vector(DATA_WIDTH-1 downto 0);
    y_out : out std_logic_vector(DATA_WIDTH-1 downto 0);
    done : out std_logic
  ) ;
end linear_1d ;

architecture rtl of linear_1d is

    signal addr_s : std_logic_vector(ADDR_WIDTH-1 downto 0) ;
    signal test_mul : signed(2*DATA_WIDTH-1 downto 0) ;
    signal test_sum : signed(2*DATA_WIDTH-1 downto 0) ;
    signal w_in, b_in : signed(DATA_WIDTH-1 downto 0) ;
    signal std_w_in, std_b_in : std_logic_vector(DATA_WIDTH-1 downto 0) ;
    signal n_clock : std_logic ;
begin

    n_clock <= not clock ;

    process(clock)
        variable var_addr : integer range 0 to 2**ADDR_WIDTH-1:= 0;
        variable fsm : integer:=0;
        variable temp_mul : signed(2*DATA_WIDTH-1 downto 0);
        variable sum : signed(2*DATA_WIDTH-1 downto 0):=(others=>'0');
        variable temp_x : signed(DATA_WIDTH-1 downto 0);
        variable temp_w : signed(DATA_WIDTH-1 downto 0);
        variable prefetc_flag:std_logic;
    begin
        if rising_edge(clock) then
            if enable = '0' then
                var_addr := 0;
                fsm := 0;
                sum := (others=>'0');
                temp_mul := (others=>'0');
                done <= '0';
            else

            if fsm=0 then
                fsm := 1;
            elsif fsm =1 then
                if prefetc_flag='0' then
                    prefetc_flag := '1';
                    temp_x := signed(x_in);
                    temp_w := signed(w_in);
                    temp_mul := multiply_16_8_without_cut(temp_x,temp_w);
                else
                    sum := sum + temp_mul;
                    var_addr := var_addr + 1;
                    if var_addr=IN_FEATURE_COUNT then
                        fsm := 2;
                        var_addr := 0;
                    end if;
                    prefetc_flag := '0';
                end if;
            elsif fsm =2 then
                done <= '1';
                y_out <= std_logic_vector(cut_16_to_8(test_sum)+signed(b_in));
            end if;
            end if;
            addr_s <= std_logic_vector(to_unsigned(var_addr, ADDR_WIDTH));

            test_mul <= temp_mul;
            test_sum <= sum;
        end if;
    end process;

    addr <= addr_s;

    -- Weights
    rom_w : entity {work_library_name}.w_rom(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_s,
        data => std_w_in
    );
    w_in <= signed(std_w_in);

    -- Bias
    rom_b : entity {work_library_name}.b_rom(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => (others=>'0'), -- ToDo: at the moment only one bias is supported
        data => std_b_in
    );
    b_in <= signed(std_b_in);

end architecture ; -- rtl
