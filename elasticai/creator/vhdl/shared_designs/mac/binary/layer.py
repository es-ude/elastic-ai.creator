from elasticai.creator.file_generation.savable import Savable
from elasticai.creator.nn.binary._math_operations import MathOperations

from .design import MacDesign
from .mactestbench import MacTestBench


class MacLayer:
    def __init__(self, vector_width: int):
        self.ops = MathOperations()
        self.name = "bin_mac"
        self._vector_width = vector_width

    def __call__(self, a, b):
        return self.ops.matmul(a, b)

    def create_design(self, name: str) -> Savable:
        return MacDesign(vector_width=self._vector_width, name=name)

    def create_testbench(self, name: str, design: MacDesign) -> MacTestBench:
        return MacTestBench(
            uut=design,
            uut_name="mac",
            name=name,
        )
