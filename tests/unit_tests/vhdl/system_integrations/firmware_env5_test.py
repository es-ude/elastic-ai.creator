from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.identity.layer import BufferedIdentity
from elasticai.creator.nn.sequential.layer import Sequential
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5


def extract_skeleton_code(firmware) -> str:
    destination = InMemoryPath(name="build", parent=None)
    firmware.save_to(destination)
    srcs_dir = cast(InMemoryPath, destination["srcs"])
    actual_code_lines = cast(InMemoryFile, srcs_dir["skeleton"]).text
    actual_code = "\n".join(actual_code_lines)
    return actual_code


def test_firmware_generates_correct_skeleton_v1() -> None:
    model = Sequential(BufferedIdentity(num_input_features=10, total_bits=8))
    design = model.create_design("network")
    with pytest.warns(FutureWarning):
        firmware = FirmwareENv5(
            design, x_num_values=10, y_num_values=10, id=66, skeleton_version="v1"
        )

        actual_code = extract_skeleton_code(firmware)
    expected_code = """library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity skeleton is
    port (
        -- control interface
        clock                : in std_logic;
        clk_hadamard                : in std_logic;
        reset                : in std_logic; -- controls functionality (sleep)
        busy                : out std_logic; -- done with entire calculation
        wake_up             : out std_logic;
        -- indicate new data or request
        rd                    : in std_logic;    -- request a variable
        wr                 : in std_logic;     -- request changing a variable

        -- data interface
        data_in            : in std_logic_vector(7 downto 0);
        address_in        : in std_logic_vector(15 downto 0);
        data_out            : out std_logic_vector(7 downto 0);

        debug                : out std_logic_vector(7 downto 0);

        led_ctrl             : out std_logic_vector(3 DOWNTO 0)
    );
end;

architecture rtl of skeleton is
    constant DATA_WIDTH_IN : integer := 8;
    constant DATA_WIDTH_OUT : integer := 8;
    constant X_ADDR_WIDTH : integer := 4;
    constant X_NUM_VALUES : integer := 10;
    constant Y_ADDR_WIDTH : integer:= 4;
    constant Y_NUM_VALUES : integer := 10;

    signal network_enable :  std_logic;

    signal c_config_en :  std_logic;
    signal done :  std_logic;

    signal x :  std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal y : std_logic_vector(DATA_WIDTH_OUT-1 downto 0);
    signal x_address :  std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal y_address :  std_logic_vector(Y_ADDR_WIDTH-1 downto 0);

    type buf_data_in_t is array (0 to X_NUM_VALUES) of std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal data_buf_in : buf_data_in_t;
    type skeleton_id_data_t is array (0 to 0) of std_logic_vector(7 downto 0);
    signal skeleton_id_str : skeleton_id_data_t := (0 => "01000010");

    function pad_output_to_middleware(network_out : std_logic_vector(DATA_WIDTH_OUT-1 downto 0)) return std_logic_vector is
    variable k : std_logic_vector(7 downto 0);
    begin
        if DATA_WIDTH_OUT /= 8 then
            k(7 downto DATA_WIDTH_OUT) := (others => '0');
        end if;
        k(DATA_WIDTH_OUT-1 downto 0) := network_out;
        return k;
    end function;

begin

    i_network: entity work.network(rtl)
    port map (
        clock => clock,
        enable => network_enable,
        x => x,
        x_address => x_address,
        y => y,
        y_address => y_address,
        done => done
    );

    -- orignial implementation
    busy <= not done;
    wake_up <= done;

    receive_data_from_middleware: process (clock, wr, address_in)
    variable int_addr : integer range 0 to 20000;
    begin
        if rising_edge(clock) then
            if reset = '1' then
                network_enable <= '0';
            else
                int_addr := to_integer(unsigned(address_in));
                if int_addr < X_NUM_VALUES then
                    data_buf_in(int_addr) <= data_in(DATA_WIDTH_IN-1 downto 0);
                elsif int_addr = 100 then
                    network_enable <= data_in(0);
                end if;
            end if;
        end if;
    end process;

    sendback_data_to_middleware: process  (clock, rd, address_in)
    variable int_addr : integer range 0 to 20000;
    begin
        if rising_edge(clock) then
            int_addr := to_integer(unsigned(address_in));
            if int_addr <= Y_NUM_VALUES-1 then
                y_address <= address_in(y_address'length-1 downto 0);
                data_out(7 downto 0) <= pad_output_to_middleware(y);
            elsif int_addr = 2000  then
                data_out(7 downto 0) <= skeleton_id_str(int_addr-2000);
            end if;
        end if;
    end process;

    send_buf_to_network: process (clock, x_address)
    begin
        if rising_edge(clock) then
            x <= data_buf_in(to_integer(unsigned(x_address)));
        end if;
    end process;

end rtl;"""

    assert expected_code == actual_code


def test_firmware_generates_correct_skeleton_v2() -> None:
    model = Sequential(BufferedIdentity(num_input_features=10, total_bits=8))
    design = model.create_design("network")
    firmware = FirmwareENv5(
        design,
        x_num_values=10,
        y_num_values=10,
        id=list(range(0, 16)),
        skeleton_version="v2",
    )
    actual_code = extract_skeleton_code(firmware)
    expected_code = """library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity skeleton is
    port (
        -- control interface
        clock                : in std_logic;
        clk_hadamard                : in std_logic;
        reset                : in std_logic; -- controls functionality (sleep)
        busy                : out std_logic; -- done with entire calculation
        wake_up             : out std_logic;
        -- indicate new data or request
        rd                    : in std_logic;    -- request a variable
        wr                 : in std_logic;     -- request changing a variable

        -- data interface
        data_in            : in std_logic_vector(7 downto 0);
        address_in        : in std_logic_vector(15 downto 0);
        data_out            : out std_logic_vector(7 downto 0);

        debug                : out std_logic_vector(7 downto 0);

        led_ctrl             : out std_logic_vector(3 DOWNTO 0)
    );
end;

architecture rtl of skeleton is
    constant DATA_WIDTH_IN : integer := 8;
    constant DATA_WIDTH_OUT : integer := 8;
    constant X_ADDR_WIDTH : integer := 4;
    constant X_NUM_VALUES : integer := 10;
    constant Y_ADDR_WIDTH : integer:= 4;
    constant Y_NUM_VALUES : integer := 10;

    signal network_enable :  std_logic;

    signal c_config_en :  std_logic;
    signal done :  std_logic;

    signal x :  std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal y : std_logic_vector(DATA_WIDTH_OUT-1 downto 0);
    signal x_address :  std_logic_vector(X_ADDR_WIDTH-1 downto 0);
    signal y_address :  std_logic_vector(Y_ADDR_WIDTH-1 downto 0);

    type buf_data_in_t is array (0 to X_NUM_VALUES) of std_logic_vector(DATA_WIDTH_IN-1 downto 0);
    signal data_buf_in : buf_data_in_t;
    type skeleton_id_data_t is array (0 to 15) of std_logic_vector(7 downto 0);
    signal skeleton_id_str : skeleton_id_data_t := ("00000000", "00000001", "00000010", "00000011", "00000100", "00000101", "00000110", "00000111", "00001000", "00001001", "00001010", "00001011", "00001100", "00001101", "00001110", "00001111");

    function pad_output_to_middleware(network_out : std_logic_vector(DATA_WIDTH_OUT-1 downto 0)) return std_logic_vector is
    variable k : std_logic_vector(7 downto 0);
    begin
        if DATA_WIDTH_OUT /= 8 then
            k(7 downto DATA_WIDTH_OUT) := (others => '0');
        end if;
        k(DATA_WIDTH_OUT-1 downto 0) := network_out;
        return k;
    end function;

begin

    i_network: entity work.network(rtl)
    port map (
        clock => clock,
        enable => network_enable,
        x => x,
        x_address => x_address,
        y => y,
        y_address => y_address,
        done => done
    );

    busy <= not done;
    wake_up <= done;

    receive_data_from_middleware: process (clock)
    variable int_addr : integer range 0 to 18 + X_NUM_VALUES;
    begin
        if rising_edge(clock) then
            if reset = '1' then
                network_enable <= '0';
            else
                if wr = '1' then
                    int_addr := to_integer(unsigned(address_in));
                    if int_addr = 16 then
                        network_enable <= data_in(0);
                    elsif int_addr >= 18 and int_addr < 18 + X_NUM_VALUES then
                        data_buf_in(int_addr-18) <= data_in(DATA_WIDTH_IN-1 downto 0);
                    end if;
                end if;
            end if;
        end if;
    end process;

    sendback_data_to_middleware: process  (clock)
    variable int_addr : integer range 0 to 18 + Y_NUM_VALUES;
    begin
        if rising_edge(clock) then
            if rd = '1' then
                int_addr := to_integer(unsigned(address_in));
                if int_addr <= 15 then
                    data_out(7 downto 0) <= skeleton_id_str(int_addr);
                elsif int_addr >= 18 and int_addr < 18 + Y_NUM_VALUES then
                    y_address <= std_logic_vector(unsigned(address_in(y_address'length-1 downto 0))-18);
                    data_out(7 downto 0) <= pad_output_to_middleware(y);
                end if;
            end if;
        end if;
    end process;

    send_buf_to_network: process (clock)
    begin
        if falling_edge(clock) then
            x <= data_buf_in(to_integer(unsigned(x_address)));
        end if;
    end process;

end rtl;"""
    assert expected_code == actual_code
