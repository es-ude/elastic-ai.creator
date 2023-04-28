from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Translatable


class FPBatchNormedLinear(Translatable):
    def translate(self, name: str) -> Design:
        ...
