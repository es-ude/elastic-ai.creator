library ieee;
use ieee.std_logic_1164.all;

entity ${name} is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(${x_address_width}-1 downto 0);
        y_address : in std_logic_vector(${y_address_width}-1 downto 0);

        x   : in std_logic_vector(${x_width}-1 downto 0);
        y  : out std_logic_vector(${y_width}-1 downto 0);

        done   : out std_logic
    );
end;

architecture rtl of ${name} is
    constant TOTAL_WIDTH : natural := ${x_width};
    constant FRAC_WIDTH : natural := ${frac_width};
    constant VECTOR_WIDTH : natural := ${vector_width};
    constant KERNEL_SIZE : natural := ${kernel_size};
    constant IN_CHANNELS : natural := ${in_channels};
    constant OUT_CHANNELS : natural := ${out_channels};
    constant X_ADDRESS_WIDTH : natural := ${x_address_width};
    constant Y_ADDRESS_WIDTH : natural := ${y_address_width};

    signal reset : std_logic;

begin

    reset <= not enable;

    ${name}_conv1d : entity work.conv1d_fxp_MAC_RoundToZero
        generic map(
            TOTAL_WIDTH => TOTAL_WIDTH,
            FRAC_WIDTH => FRAC_WIDTH,
            VECTOR_WIDTH => VECTOR_WIDTH,
            KERNEL_SIZE => KERNEL_SIZE,
            IN_CHANNELS => IN_CHANNELS,
            OUT_CHANNELS => OUT_CHANNELS,
            X_ADDRESS_WIDTH => X_ADDRESS_WIDTH,
            Y_ADDRESS_WIDTH => Y_ADDRESS_WIDTH
        )
        port map (
            clock => clock,
            enable => enable,
            reset => reset,
            x => x,
            x_address => x_address,
            y => y,
            y_address => y_address,
            done => done
        );
end rtl;
