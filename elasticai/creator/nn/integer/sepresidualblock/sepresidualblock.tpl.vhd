library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library ${work_library_name};
use ${work_library_name}.all;

entity ${name} is
    generic (
        DATA_WIDTH : integer := ${data_width};
        IN_CHANNELS : integer := ${in_channels};
        OUT_CHANNELS : integer := ${out_channels};
        KERNEL_SIZE : integer := ${kernel_size};
        M_Q : integer := ${m_q};
        M_Q_SHIFT : integer := ${m_q_shift};
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock : in std_logic;
        x_address : out std_logic_vector(IN_CHANNELS - 1 downto 0);
        x : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y_address : in std_logic_vector(OUT_CHANNELS - 1 downto 0);
        y : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done : out std_logic
    );
end ${name};

architecture rtl of ${name} is
    signal temp_y : std_logic_vector(DATA_WIDTH - 1 downto 0);
begin
    -- Depthwise Convolution
    -- Pointwise Convolution
    -- Shortcut Addition
    -- Final ReLU
end rtl;
