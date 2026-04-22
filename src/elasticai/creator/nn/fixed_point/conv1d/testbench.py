import math
from abc import abstractmethod
from collections import defaultdict
from typing import Protocol

from elasticai.creator.arithmetic import FxpConverter, FxpParams
from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.testing import Testbench
from elasticai.creator.vhdl.design.ports import Port


class Conv1dDesignProtocol(Protocol):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def input_signal_length(self) -> int: ...

    @property
    @abstractmethod
    def port(self) -> Port: ...

    @property
    @abstractmethod
    def kernel_size(self) -> int: ...

    @property
    @abstractmethod
    def in_channels(self) -> int: ...

    @property
    @abstractmethod
    def out_channels(self) -> int: ...


class Conv1dTestbench(Testbench):
    def __init__(self, name: str, uut: Conv1dDesignProtocol, fxp_params: FxpParams):
        self._converter = FxpConverter(
            FxpParams(
                total_bits=fxp_params.total_bits,
                frac_bits=fxp_params.frac_bits,
                signed=True,
            )
        )
        self._converter_for_batch = FxpConverter(
            FxpParams(total_bits=8, frac_bits=0, signed=True)
        )  # max for 255 lines of inputs
        self._name = name
        self._uut_name = uut.name
        self._input_signal_length = uut.input_signal_length
        self._in_channels = uut.in_channels
        self._out_channels = uut.out_channels
        self._x_address_width = uut.port["x_address"].width
        self._fxp_params = fxp_params
        self._kernel_size = uut.kernel_size
        self._output_signal_length = math.floor(
            self._input_signal_length - self._kernel_size + 1
        )
        self._y_address_width = uut.port["y_address"].width

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="testbench.tpl.vhd",
            parameters={
                "testbench_name": self.name,
                "input_signal_length": str(self._input_signal_length),
                "in_channels": str(self._in_channels),
                "out_channels": str(self._out_channels),
                "total_bits": str(self._fxp_params.total_bits),
                "x_address_width": str(self._x_address_width),
                "output_signal_length": str(self._output_signal_length),
                "y_address_width": str(self._y_address_width),
                "uut_name": self._uut_name,
            },
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

    @property
    def name(self) -> str:
        return self._name

    def prepare_inputs(self, *inputs) -> list[dict]:
        batches = inputs[0]
        prepared_inputs = []
        for batch in batches:
            prepared_inputs.append({})
            for channel_id, channel in enumerate(batch):
                for time_step_id, time_step_val in enumerate(channel):
                    prepared_inputs[-1][f"x_{channel_id}_{time_step_id}"] = (
                        self._converter.rational_to_binary_string_vhdl(
                            time_step_val
                        ).replace('"', "")
                    )

        return prepared_inputs

    def parse_reported_content(self, content: list[str]) -> list[list[list[float]]]:
        """
        This function parses the reported content, which is just a list of strings.
        All lines starting with 'output_text:' are considered as a result of the testbench.
        These results will be stacked for each batch.
        So you get a list[list[list[float]]] which is similar to batch[out channels[output neurons[float]]].
        For each item reported it is checked if the string starts with 'result: '.
        If so the remaining part will be split by ','. The first number gives the batch. The second the result.
        The channels are greedy guessed.
        We do so by taking the first X-values, where X is the number of values per channel.
        After we have enough values for the channel, we increase the channel number.
        If you report not enough values per channel, this will look like the last channel has not reported enough values.
        """

        def split_list(a_list):
            out_channel_length = len(a_list) // self._out_channels
            new_list = list()
            out_channel_counter = (
                -1
            )  # start with -1 because it will be increased in first iteration of loop
            for i, value in enumerate(a_list):
                if i % out_channel_length == 0:
                    new_list.append(list())
                    out_channel_counter += 1
                new_list[out_channel_counter].append(value)
            return new_list

        results_dict = defaultdict(list)
        for line in map(str.strip, content):
            if line.startswith("result: "):
                batch_text = line.split(":")[1].split(",")[0][1:]
                output_text = line.split(":")[1].split(",")[1][0:]
                print("output_text: ", output_text)
                batch = int(self._converter_for_batch.binary_to_rational(batch_text))
                if "U" not in line.split(":")[1].split(",")[1][1:]:
                    output = self._converter.binary_to_rational(output_text)
                else:
                    output = output_text
                results_dict[batch].append(output)
        results = list()
        for x in results_dict.items():
            results.append(split_list(x[1]))
        if len(results) == 0:
            raise Exception(content)
        return list(results)
