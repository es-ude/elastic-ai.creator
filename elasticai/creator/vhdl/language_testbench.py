"""
The module contains classes and functions for generating vhdl code similar to the language module
This module includes CodeGenerator that are only used by the vhdl testbenches
"""
from enum import Enum
from elasticai.creator.vhdl.language import _indent_and_filter_non_empty_lines


class Keywords(Enum):
    UUT = "uut"


class UUT:
    def __init__(self, identifier: str):
        self.identifier = identifier

    def __call__(self):
        #     uut: sigmoid
        #     port map (
        #         x => test_input,
        #         y => test_output
        #     );
        yield f"{Keywords.UUT.value}: {self.identifier}"
        # TODO: change with ahmads portmap
        yield f"port map ("
        yield from _indent_and_filter_non_empty_lines(
            ["x => test_input", "y => test_output"]
        )
        yield f");"
