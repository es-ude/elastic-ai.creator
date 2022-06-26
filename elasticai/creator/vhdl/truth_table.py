from abc import ABC, abstractmethod
from string import Template
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import ToLogicEncoder
from elasticai.creator.vhdl.templates.utils import expand_multiline_template


# noinspection PyPropertyDefinition
class IOTable(Iterable[tuple[int, int]], Protocol):
    @property
    def input_bit_width(self) -> int:
        ...

    @property
    def output_bit_width(self) -> int:
        ...


class _EncodedIOTable(IOTable):
    def __iter__(self) -> Iterator[tuple[int, int]]:
        for i in range(len(self)):
            yield self[i]

    def __init__(
        self,
        input_encoder: ToLogicEncoder,
        output_encoder: ToLogicEncoder,
        inputs: Sequence[int],
        outputs: Sequence[int],
    ):
        self._input_encoder = input_encoder
        self._output_encoder = output_encoder
        self._inputs = inputs
        self._outputs = outputs

    def __getitem__(self, item) -> tuple[str, str]:
        encoded_input = self._input_encoder(self._inputs.__getitem__(item))
        encoded_output = self._output_encoder(self._outputs.__getitem__(item))
        return encoded_input, encoded_output

    def __len__(self):
        return min(len(self._inputs), len(self._outputs))

    @property
    def input_bit_width(self) -> int:
        return self._input_encoder.bit_width

    @property
    def output_bit_width(self) -> int:
        return self._output_encoder.bit_width


class TruthTableVHDLDesign(ABC):
    def __init__(self, io_table: IOTable, entity_name: str, architecture_name: str):
        self._table = io_table
        self.entity_name = entity_name
        self.architecture_name = architecture_name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @property
    def input_bitwidth(self) -> int:
        return self._table.input_bit_width

    @property
    def output_bitwidth(self) -> int:
        return self._table.output_bit_width

    def __iter__(self):
        yield from self.__call__()

    @classmethod
    def by_enumerating_output_sequence(
        cls, entity_name: str, architecture_name: str, outputs: Sequence[int]
    ):
        num_outputs = len(outputs)
        num_inputs = num_outputs
        inputs = range(num_inputs)

        input_encoder = ToLogicEncoder()
        output_encoder = ToLogicEncoder()
        output_encoder.register_symbols(np.unique(outputs))
        input_encoder.register_symbols(range(num_inputs))

        io_table = _EncodedIOTable(input_encoder, output_encoder, inputs, outputs)
        vhdl_lut_design = cls(io_table, entity_name, architecture_name)
        return vhdl_lut_design


class TruthTableVHDLDesignCaseWhen(TruthTableVHDLDesign):
    def __call__(self) -> Code:
        cases = [
            f'when "{input}" => output <= "{output}";' for input, output in self._table
        ]
        code = expand_multiline_template(
            read_text(
                "elasticai.creator.vhdl.templates.precomputed_convs",
                "truth_table.tpl.vhd",
            ),
            cases=cases,
        )

        def substitution(line: str):
            return Template(line).substitute(
                entity_name=self.entity_name,
                architecture_name=self.architecture_name,
                input_vector_start_bit=self._table.input_bit_width - 1,
                output_vector_start_bit=self._table.output_bit_width - 1,
            )

        yield from map(substitution, code)
