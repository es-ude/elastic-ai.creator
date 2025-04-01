from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design.design import Design


class LSTMTestBench:
    def __init__(self, name: str, uut: Design):
        self._uut = uut
        self.name = name

    def save_to(self, destination: Path):
        test_bench = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstm_network_tb.tpl.vhd",
            parameters={"name": self.name, "uut_name": self._uut.name},
        )
        destination.create_subpath(f"{self.name}").as_file(".vhd").write(test_bench)
