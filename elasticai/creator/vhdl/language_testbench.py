"""
The module contains classes and functions for generating vhdl code similar to the language module
This module includes CodeGenerator that are only used by the vhdl testbenches
"""
from abc import ABC, abstractmethod
from typing import Iterator


class TestBenchBase(ABC):
    simulation_start_msg = 'report "======Simulation Start======" severity Note'
    simulation_end_msgs = (
        'report "======Simulation Success======" severity Note',
        'report "Please check the output message." severity Note',
        "wait",
    )

    @abstractmethod
    def _body(self) -> Iterator[str]:
        ...

    def __iter__(self) -> Iterator[str]:
        yield self.simulation_start_msg
        yield from self._body()
        yield from self.simulation_end_msgs
