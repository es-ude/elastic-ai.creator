from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design.design import Design


class LSTMTestBench:
    def __init__(self, name: str, uut: Design):
        self._uut = uut
        self._name = name

    def save_to(self, destination: Path):
        test_bench = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=f"lstm_network_tb.tpl.vhd",
            parameters={"uut_name": self._uut.name},
        )
        destination.create_subpath(f"{self._name}_tb").as_file(".vhd").write(test_bench)
        self._uut.save_to(destination)
