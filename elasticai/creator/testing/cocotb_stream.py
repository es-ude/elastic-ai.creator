from collections.abc import Callable, Sequence
from typing import Any, overload

from cocotb.handle import LogicArrayObject, LogicObject, SimHandleBase
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray, Range


def bitstring_to_logic_array(value: str) -> LogicArray:
    return LogicArray(value, Range(len(value) - 1, "downto", 0))


def logic_value_to_bitstring(value: LogicArray) -> str:
    return str(value)


def set_from_bit_string(signal: LogicArrayObject, bitstring: str) -> None:
    signal.value = bitstring_to_logic_array(bitstring)


class ClockReset:
    def __init__(self, clk: LogicObject, rst: LogicObject):
        self.clk = clk
        self.rst = rst

    @classmethod
    def from_dut(
        cls,
        dut: SimHandleBase,
        *,
        clk_signal: str = "clk",
        rst_signal: str = "rst",
    ) -> "ClockReset":
        return cls(clk=getattr(dut, clk_signal), rst=getattr(dut, rst_signal))

    async def reset_active_high(self, *, reset_cycles: int = 1) -> None:
        self.rst.value = 1
        for _ in range(reset_cycles):
            await RisingEdge(self.clk)
        self.rst.value = 0
        await RisingEdge(self.clk)


class StreamInterface[TInput, TOutput]:
    def __init__(
        self,
        clk: LogicObject,
        data_in: LogicArrayObject,
        valid_in: LogicObject,
        data_out: LogicArrayObject,
        valid_out: LogicObject,
        ready_in: LogicObject,
        *,
        input_to_logic_array: Callable[[TInput], LogicArray],
        output_from_value: Callable[[LogicArray], TOutput],
    ):
        self.clk = clk
        self.data_in = data_in
        self.valid_in = valid_in
        self.data_out = data_out
        self.valid_out = valid_out
        self.ready_in = ready_in
        self._input_to_logic_array = input_to_logic_array
        self._output_from_value = output_from_value

    @classmethod
    @overload
    def from_dut(
        cls,
        dut: SimHandleBase,
        *,
        clk_signal: str = "clk",
        data_in_signal: str = "d_in",
        valid_in_signal: str = "src_valid",
        data_out_signal: str = "d_out",
        valid_out_signal: str = "valid",
        ready_in_signal: str = "dst_ready",
    ) -> "StreamInterface[str, str]": ...

    @classmethod
    @overload
    def from_dut(
        cls,
        dut: SimHandleBase,
        *,
        clk_signal: str = "clk",
        data_in_signal: str = "d_in",
        valid_in_signal: str = "src_valid",
        data_out_signal: str = "d_out",
        valid_out_signal: str = "valid",
        ready_in_signal: str = "dst_ready",
        input_to_logic_array: Callable[[TInput], LogicArray],
        output_from_value: Callable[[LogicArray], TOutput],
    ) -> "StreamInterface[TInput, TOutput]": ...

    @classmethod
    def from_dut(
        cls,
        dut: SimHandleBase,
        *,
        clk_signal: str = "clk",
        data_in_signal: str = "d_in",
        valid_in_signal: str = "src_valid",
        data_out_signal: str = "d_out",
        valid_out_signal: str = "valid",
        ready_in_signal: str = "dst_ready",
        input_to_logic_array: Callable[[Any], LogicArray] = bitstring_to_logic_array,
        output_from_value: Callable[[LogicArray], Any] = logic_value_to_bitstring,
    ) -> "StreamInterface[Any, Any]":
        return cls(
            clk=getattr(dut, clk_signal),
            data_in=getattr(dut, data_in_signal),
            valid_in=getattr(dut, valid_in_signal),
            data_out=getattr(dut, data_out_signal),
            valid_out=getattr(dut, valid_out_signal),
            ready_in=getattr(dut, ready_in_signal),
            input_to_logic_array=input_to_logic_array,
            output_from_value=output_from_value,
        )

    async def drive_chunks(
        self,
        chunks: Sequence[TInput],
    ) -> None:
        for chunk in chunks:
            self.valid_in.value = 1
            self.data_in.value = self._input_to_logic_array(chunk)
            await RisingEdge(self.clk)
        self.valid_in.value = 0

    async def collect_chunks(
        self,
        *,
        expected_count: int,
        max_cycles: int,
    ) -> list[TOutput]:
        self.ready_in.value = 1
        observed: list[TOutput] = []
        for _ in range(max_cycles):
            if str(self.valid_out.value) == "1":
                observed.append(self._output_from_value(self.data_out.value))
                if len(observed) == expected_count:
                    break
            await RisingEdge(self.clk)
        self.ready_in.value = 0
        return observed
