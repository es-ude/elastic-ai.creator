import math
from abc import abstractmethod
from collections import defaultdict
from typing import Protocol

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.fixed_point.number_converter import FXPParams, NumberConverter
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.simulated_layer import Testbench


class LinearDesignProtocol(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def port(self) -> Port:
        ...

    @property
    @abstractmethod
    def in_feature_num(self) -> int:
        ...

    @property
    @abstractmethod
    def out_feature_num(self) -> int:
        ...

    @property
    @abstractmethod
    def frac_width(self) -> int:
        ...

    @property
    @abstractmethod
    def data_width(self) -> int:
        ...


class LinearTestbench(Testbench):
    def __init__(self, name: str, uut: LinearDesignProtocol):
        self._converter_for_batch = NumberConverter(
            FXPParams(8, 0)
        )  # max for 255 lines of inputs
        self._name = name
        self._uut_name = uut.name
        self._input_signal_length = uut.in_feature_num
        self._x_address_width = uut.port["x_address"].width
        self._fxp_params = FXPParams(uut.data_width, uut.frac_width)
        self._converter = NumberConverter(self._fxp_params)
        self._output_signal_length = uut.out_feature_num
        self._y_address_width = uut.port["y_address"].width

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="testbench.tpl.vhd",
            parameters={
                "testbench_name": self.name,
                "input_signal_length": str(self._input_signal_length),
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
                        self._converter.rational_to_bits(time_step_val)
                    )
        return prepared_inputs

    def parse_reported_content(self, content: list[str]) -> list[list[list[float]]]:
        """
        This function parses the reported content, which is just a list of strings.
        All lines starting with 'output_text:' are considered as a result of the testbench.
        These results will be stacked for each batch.
        So you get a list[list[list[float]]] which is similar to batch[out channels[output neurons[float]]].
        For linear layer the output neurons is 1.
        For each item reported it is checked if the string starts with 'result: '.
        If so the remaining part will be split by ','. The first number gives the batch. The second the result.
        """

        def split_list(a_list):
            print("len(a_list): ", len(a_list))
            new_list = list()
            new_list.append(list())
            for i, value in enumerate(a_list):
                new_list[0].append(value)
            return new_list

        results_dict = defaultdict(list)

        print()
        for line in map(str.strip, content):
            if line.startswith("result: "):
                batch_text = line.split(":")[1].split(",")[0][1:]
                output_text = line.split(":")[1].split(",")[1][0:]
                print("output_text: ", output_text)
                batch = int(self._converter_for_batch.bits_to_rational(batch_text))
                if "U" not in line.split(":")[1].split(",")[1][1:]:
                    output = self._converter.bits_to_rational(output_text)
                else:
                    output = output_text
                results_dict[batch].append(output)
            else:
                print(line)
        results = list()
        for x in results_dict.items():
            results.append(split_list(x[1]))
        print("results: ", results)
        if len(results) is 0:
            raise Exception(content)
        return list(results)
