from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.identity.layer import BufferedIdentity, BufferlessIdentity


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
""".splitlines()
    identity = BufferedIdentity(num_input_features=6, total_bits=16)
    build_path = InMemoryPath("build", parent=None)
    design = identity.create_design("identity")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["identity"]).text
    assert actual == expected


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
""".splitlines()
    identity = BufferlessIdentity(total_bits=8)
    build_path = InMemoryPath("build", parent=None)
    design = identity.create_design("identity")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["identity"]).text
    assert actual == expected
