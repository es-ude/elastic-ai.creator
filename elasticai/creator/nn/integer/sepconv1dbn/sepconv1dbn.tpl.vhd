library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library ${work_library_name};
use ${work_library_name}.all;
entity ${name} is
    generic (
        X_ADDR_WIDTH : integer := ${x_addr_width};
        DEPTH_Y_ADDR_WIDTH : integer := ${depth_y_addr_width};
        Y_ADDR_WIDTH : integer := ${y_addr_width};
        DATA_WIDTH : integer := ${data_width}
    );
    port (
        enable: in std_logic;
        clock: in std_logic;
        x_address: out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        x: in std_logic_vector(DATA_WIDTH-1 downto 0);
        y_address: in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        y: out std_logic_vector(DATA_WIDTH-1 downto 0);
        done: out std_logic
    );
end entity ${name};
architecture rtl of ${name} is
    signal depth_enable : std_logic;
    signal depth_clock : std_logic;
    signal depth_x_address : std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal depth_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal depth_y_address : std_logic_vector(DEPTH_Y_ADDR_WIDTH-1 downto 0);
    signal depth_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal depth_done : std_logic;
    signal depth_valid : std_logic;
    signal depth_ready : std_logic;
    signal point_enable : std_logic;
    signal point_clock : std_logic;
    signal point_x_address : std_logic_vector(DEPTH_Y_ADDR_WIDTH-1 downto 0);
    signal point_x : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal point_y_address : std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
    signal point_y : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal point_done : std_logic;
    signal point_valid : std_logic;
    signal point_ready : std_logic;
begin
    depth_enable <= enable;
    point_enable <= enable;
    done <= point_done;
    depth_clock <= clock;
    point_clock <= clock;
    x_address <= depth_x_address;
    depth_y_address <= point_x_address;
    point_y_address <= y_address;
    depth_x <= x;
    point_x <= depth_y;
    y <= point_y;
    inst_${name}_depthconv1d_0 : entity work.${name}_depthconv1d_0(rtl)
    port map (
        enable => depth_enable,
        clock => depth_clock,
        x_address => depth_x_address,
        x => depth_x,
        y_address => depth_y_address,
        y => depth_y,
        done => depth_done,
        valid => depth_valid,
        ready => depth_ready
    );
    inst_${name}_pointconv1dbn_0 : entity work.${name}_pointconv1dbn_0(rtl)
    port map (
        enable => point_enable,
        clock => point_clock,
        x_address =>  point_x_address,
        x => point_x,
        y_address => point_y_address,
        y => point_y,
        done => point_done,
        valid => depth_valid,
        ready => depth_ready
    );
end architecture rtl;
