library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

library work;
use work.all;


entity ${network_name} is
  generic (
    X_ADDR_WIDTH : integer := ${x_addr_width};
    Y_ADDR_WIDTH : integer := ${y_addr_width};
    DATA_WIDTH : integer := ${data_width};
    L1_IN_FEATURE_NUM : integer := ${l1_in_feature_num};
    L1_OUT_FEATURE_NUM : integer := ${l1_out_feature_num};
    L2_OUT_FEATURE_NUM : integer := ${l2_out_feature_num}
  ) ;
  port (
    clock : in std_logic;
    enable : in std_logic;

    x_addr : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    x_in : in std_logic_vector(DATA_WIDTH-1 downto 0);

    y_addr : in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    y_out : out std_logic_vector(DATA_WIDTH-1 downto 0);
    done : out std_logic
  ) ;
end ${network_name};

architecture rtl of ${network_name} is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    -- Log2 funtion is for calculating the bitwidth of the address lines
    -- for bias and weights rom
    function log2(val : INTEGER) return natural is
        variable res : natural;
    begin
        for i in 1 to 31 loop
            if (val <= (2 ** i)) then
                res := i;
                exit;
            end if;
        end loop;
        return res;
    end function log2;

    -----------------------------------------------------------
    -- Signals
    -----------------------------------------------------------
    signal l1_enable, l2_enable, relu_enable : std_logic;
    signal l1_clock, l2_clock, relu_clock : std_logic;
    signal l1_x_addr : std_logic_vector(log2(L1_IN_FEATURE_NUM)-1 downto 0);
    signal l1_y_addr,l2_x_addr : std_logic_vector(log2(L1_OUT_FEATURE_NUM)-1 downto 0);
    signal l2_y_addr : std_logic_vector(log2(L2_OUT_FEATURE_NUM)-1 downto 0);
    signal l1_x_in, l2_x_in : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal l1_y_out, l2_y_out : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal relu_x_in, relu_y_out : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal l1_done, l2_done : std_logic;
begin
    -- enable signals
    l1_enable <= enable;
    l2_enable <= l1_done;
    relu_enable <= l1_done;

    -- clock signals
    l1_clock <= clock;
    l2_clock <= clock;
    relu_clock <= clock;

    -- x, y_addr signals
    x_addr <= l1_x_addr;
    l1_y_addr <= l2_x_addr;
    l2_y_addr <= y_addr;

    -- x, y_in signals
    l1_x_in <= x_in;
    relu_x_in <= l1_y_out;
    l2_x_in <= relu_y_out;
    y_out <= l2_y_out;

    done <= l2_done;

    linear1: entity work.${linear1_name}(rtl)
    port map (
        enable => l1_enable,
        clock  => l1_clock,
        x_addr  => l1_x_addr,
        y_addr  => l1_y_addr,

        x_in  => l1_x_in,
        y_out => l1_y_out,

        done  => l1_done
      );

    relu: entity work.${relu1_name}(rtl)
    port map (
        enable => relu_enable,
        clock  => relu_clock,

        x  => relu_x_in,
        y => relu_y_out
    );

    linear2: entity work.${linear2_name}(rtl)
    port map (
        enable => l2_enable,
        clock  => l2_clock,
        x_addr  => l2_x_addr,
        y_addr  => l2_y_addr,

        x_in  => l2_x_in,
        y_out => l2_y_out,

        done  => l2_done
    );

end rtl ; -- rtl
