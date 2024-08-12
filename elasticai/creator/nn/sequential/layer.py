import abc
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design_creator import DesignCreator

from .design import Sequential as _SequentialDesign


class Sequential(DesignCreator, torch.nn.Sequential):
    def __init__(self, *submodules: DesignCreator):
        super().__init__(*cast(tuple[torch.nn.Module, ...], submodules))

    def create_design(self, name: str) -> Design:
        registry = _Registry()
        submodules: list[DesignCreator] = [
            cast(DesignCreator, m) for m in self.children()
        ]
        for module in submodules:
            registry.register(module.__class__.__name__.lower(), module)
        subdesigns = list(registry.build_designs())
        return _SequentialDesign(
            sub_designs=subdesigns,
            name=name,
        )


class IntForwardSubmission(torch.nn.Module, DesignCreator):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create_design(self, name: str) -> Design:
        ...

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def int_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


class IntegerSequential(Sequential):
    def __init__(self, *submodules: IntForwardSubmission):
        super().__init__(*submodules)

        self.submodules = submodules

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        given_input_QParams = None

        for submodule in self.submodules:
            x = submodule(x, given_input_QParams=given_input_QParams)
            given_input_QParams = submodule.output_QParams

        return x

    def int_forward(
        self, inputs: torch.Tensor, quant_data_file_dir: str = None
    ) -> torch.Tensor:
        assert not self.training, "int_forward() should only be called in eval mode"

        x = inputs

        # Save quantized input to file
        if x.dtype != torch.int32:
            q_x = self.submodules[0].input_QParams.quantizeProcess(x)
        else:
            q_x = x

        if quant_data_file_dir is not None:
            q_x_file_path = Path(quant_data_file_dir) / f"q_x.txt"
            self._save_to_file(q_x, q_x_file_path)

        for submodule in self.submodules:
            x = submodule.int_forward(x, quant_data_file_dir)

        if x.dtype == torch.int32 and quant_data_file_dir is not None:
            q_y_file_path = Path(quant_data_file_dir) / f"q_y.txt"
            self._save_to_file(x, q_y_file_path)

        x = self.submodules[-1].output_QParams.dequantizeProcess(x)

        return x

    def _save_to_file(self, tensor: torch.Tensor, file_path: str) -> None:
        tensor_numpy = tensor.int().numpy()
        with open(file_path, "a") as f:
            np.savetxt(f, tensor_numpy.reshape(-1, 1), fmt="%d")


class _Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, DesignCreator] = {}
        self._name_counters: dict[str, int] = {}

    def _make_name_unique(self, name: str) -> str:
        return f"{name}_{self._get_counter_for_name(name)}"

    def _get_counter_for_name(self, name: str) -> int:
        if name in self._name_counters:
            return self._name_counters[name]
        else:
            return 0

    def _increment_name_counter(self, name: str):
        self._name_counters[name] = 1 + self._get_counter_for_name(name)

    def register(self, name: str, d: DesignCreator):
        unique_name = self._make_name_unique(name)
        self._nodes[unique_name] = d
        self._increment_name_counter(name)

    def build_designs(self) -> Iterator[Design]:
        for name, module in self._nodes.items():
            yield module.create_design(name)
