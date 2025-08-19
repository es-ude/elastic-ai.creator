library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;

-- layer_name is for distinguish same type of layers (with various weights) in one module
-- MAC operator with one multiplier
entity ${layer_name} is
    generic (
        DATA_WIDTH   : integer := ${data_width};
        FRAC_WIDTH   : integer := ${frac_width};
        X_ADDR_WIDTH : integer := ${x_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        IN_FEATURE_NUM : integer := ${in_feature_num};
        OUT_FEATURE_NUM : integer := ${out_feature_num};
        RESOURCE_OPTION : string := ${resource_option}
        -- can be "distributed", "block", or  "auto"
    );
    port (
        enable      : in    std_logic;
        clock       : in    std_logic;
        x_address   : out   std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address   : in    std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        x           : in    std_logic_vector(DATA_WIDTH-1 downto 0);
        y           : out   std_logic_vector(DATA_WIDTH-1 downto 0);
        done        : out   std_logic
    );
end ${layer_name};

architecture rtl of ${layer_name} is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    -- FXP_ROUNDING with clamping if range violation is available
    function FXP_ROUNDING(
        x0: in signed(2*DATA_WIDTH-1 downto 0)
    ) return signed is
        variable TEMP0 : signed(DATA_WIDTH-1 downto 0) := (others => '0');
        variable TEMP1 : signed(FRAC_WIDTH-1 downto 0) := (others => '0');
    begin
        TEMP0 := x0(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP1 := x0(FRAC_WIDTH-1 downto 0);

        if (x0(2*DATA_WIDTH-1) = '1' and TEMP0(DATA_WIDTH-1) = '0') then
            TEMP0 := ('1', others => '0');
        elsif (x0(2*DATA_WIDTH-1) = '0' and TEMP0(DATA_WIDTH-1) = '1') then
            TEMP0 := ('0', others => '1');
        else
            if TEMP0(DATA_WIDTH-1) = '1' and TEMP1 /= 0 then
                TEMP0 := TEMP0 + 1;
            end if;
        end if;

        return TEMP0;
    end function;

    -- log2 function is for calculating the bitwidth of the address lines
    function log2(
        val : INTEGER
    ) return natural is
        variable res : natural;
    begin
        for i in 1 to ${log2_max_value} loop
            if (val <= (2 ** i)) then
                res := i;
                exit;
            end if;
        end loop;
        return res;
    end function log2;

    -----------------------------------------------------------
    -- Process
    -----------------------------------------------------------
    signal addr_w   : unsigned(log2(IN_FEATURE_NUM*OUT_FEATURE_NUM)-1 downto 0) := (others => '0');
    signal addr_b   : unsigned(Y_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal addr_x   : unsigned(X_ADDR_WIDTH-1 downto 0) := (others => '0');

    signal w_in, b_in           : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal fxp_x, fxp_w, fxp_b  : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal buf_x, buf_w, buf_b  : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal mac_y                : signed(2*DATA_WIDTH-1 downto 0) := (others=>'0');
    signal enable_mac, reset_mac, done_int : std_logic;

    -- simple solution for the output buffer
    type t_y_array is array (0 to OUT_FEATURE_NUM) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_ram                    : t_y_array;
    attribute rom_style             : string;
    attribute rom_style of y_ram    : signal is RESOURCE_OPTION;
begin
    -- connecting signals to ports
    fxp_w <= signed(w_in);
    fxp_x <= signed(x);
    fxp_b <= signed(b_in);

    done <= done_int;
    x_address <= std_logic_vector(addr_x);
    enable_mac <= enable and not done_int;

    -- Pipelined MAC operator and saving into buffer
    mac: process(clock)
    begin
        if rising_edge(clock) then
            if (enable_mac = '0') then
                buf_x <= (others => '0');
                buf_w <= (others => '0');
                buf_b <= (others => '0');
                mac_y <= (others => '0');
            else
                if (reset_mac = '1') then
                    buf_x <= (others => '0');
                    buf_w <= (others => '0');
                    buf_b <= (others => '0');
                    mac_y <= (others => '0');
                    y_ram(to_integer(unsigned(addr_b))) <= std_logic_vector(FXP_ROUNDING(mac_y + buf_w * buf_x + SHIFT_LEFT(RESIZE(buf_b, 2*DATA_WIDTH), FRAC_WIDTH)));
                else
                    buf_x <= fxp_x;
                    buf_w <= fxp_w;
                    buf_b <= fxp_b;
                    mac_y <= mac_y + (buf_w * buf_x);
                end if;
            end if;
        end if;
    end process mac;

    -- Counter Operator for controlling the linear layer
    control : process (clock)
    begin
        if rising_edge(clock) then
            if (enable = '0') then
                done_int <= '0';
                addr_x <= (others => '0');
                addr_w <= (others => '0');
                addr_b <= (others => '0');
                reset_mac <= '0';
            else
                if (done_int <= '0') then
                    if (addr_x = IN_FEATURE_NUM-1) then
                        if (reset_mac = '0') then
                            reset_mac <= '1';
                        else
                            reset_mac <= '0';

                            addr_x <= (others => '0');
                            if (addr_b = OUT_FEATURE_NUM-1) then
                                addr_b <= (others => '0');
                                addr_w <= (others => '0');
                                done_int <= '1';
                            else
                                addr_b <= addr_b + 1;
                                addr_w <= addr_w + 1;
                                done_int <= '0';
                            end if;
                        end if;
                    else
                        done_int <= '0';
                        addr_x <= addr_x + 1;
                        addr_b <= addr_b;
                        addr_w <= addr_w + 1;
                    end if;
                else
                    done_int <= '1';
                    addr_x <= (others => '0');
                    addr_w <= (others => '0');
                    addr_b <= (others => '0');
                end if;
            end if;
        end if;
    end process control;

    -- Reading operator
    y_reading : process (clock)
    begin
        if rising_edge(clock) then
            if (done_int = '1') then
                y <= y_ram(to_integer(unsigned(y_address)));
            end if;
        end if;
    end process y_reading;

    -- Weights
    rom_w : entity ${work_library_name}.${weights_rom_name}(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => std_logic_vector(addr_w),
        data => w_in
    );

    -- Bias
    rom_b : entity ${work_library_name}.${bias_rom_name}(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => std_logic_vector(addr_b),
        data => b_in
    );
end architecture rtl;
