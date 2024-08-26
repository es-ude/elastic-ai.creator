from pathlib import Path

from elasticai.creator.file_generation.v2.template import (
    InProjectTemplate,
    save_template,
)
from elasticai.creator.vhdl.design.design import Design


class LSTMTestBench:
    def __init__(self, name: str, uut: Design):
        self._uut = uut
        self.name = name

    def save_to(self, destination: Path):
        test_bench = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name=f"lstm_network_tb.tpl.vhd",
            parameters={"name": self.name, "uut_name": self._uut.name},
        )
        save_template(test_bench, destination / f"{self.name}.vhd")
