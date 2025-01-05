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
        Z_X : integer := ${z_x};
        Z_W : integer := ${z_w};
        Z_B : integer := ${z_b};
        Z_Y : integer := ${z_y};
        RESOURCE_OPTION : string := "${resource_option}"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        y      : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        done   : out std_logic
    );
end ${name};

architecture rtl of ${name} is
    signal conv1_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal conv2_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal shortcut_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal add_out : std_logic_vector(DATA_WIDTH - 1 downto 0);
begin
    -- First convolution layer
    conv1: entity ${work_library_name}.conv1d(rtl)
        port map (
            enable => enable,
            clock  => clock,
            x      => x,
            y      => conv1_out
        );

    -- Second convolution layer
    conv2: entity ${work_library_name}.conv1d(rtl)
        port map (
            enable => enable,
            clock  => clock,
            x      => conv1_out,
            y      => conv2_out
        );

    -- Shortcut layer
    shortcut: entity ${work_library_name}.conv1d(rtl)
        port map (
            enable => enable,
            clock  => clock,
            x      => x,
            y      => shortcut_out
        );

    -- Addition layer
    addition: entity ${work_library_name}.adder(rtl)
        port map (
            enable => enable,
            clock  => clock,
            x1     => conv2_out,
            x2     => shortcut_out,
            y      => add_out
        );

    -- Output
    relu: entity ${work_library_name}.relu(rtl)
        port map (
            enable => enable,
            clock  => clock,
            x      => add_out,
            y      => y,
            done   => done
        );
end architecture;
