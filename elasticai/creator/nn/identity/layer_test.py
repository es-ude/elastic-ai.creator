from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from tests.temporary_file_structure import get_savable_file_structure

from .layer import BufferedIdentity, BufferlessIdentity


def test_buffered_identity_generates_correct_vhdl_code() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;


entity identity is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(3-1 downto 0);
        y_address : in std_logic_vector(3-1 downto 0);
        x   : in std_logic_vector(16-1 downto 0);
        y  : out std_logic_vector(16-1 downto 0);
        done   : out std_logic
    );
end identity;

architecture rtl of identity is
begin
    y <= x;
    done <= enable;
    x_address <= y_address;
end rtl;
"""
    identity = BufferedIdentity(num_input_features=6, total_bits=16)
    assert expected == get_code(identity)


def test_bufferless_identity_generates_correct_vhdl_code() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;


entity identity is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x   : in std_logic_vector(8-1 downto 0);
        y  : out std_logic_vector(8-1 downto 0);
    );
end identity;

architecture rtl of identity is
begin
    y <= x;
end rtl;
"""
    identity = BufferlessIdentity(total_bits=8)
    assert expected == get_code(identity)


def get_code(layer: DesignCreatorModule) -> str:
    design = layer.create_design("identity")
    return get_savable_file_structure(design)["identity.vhd"]
