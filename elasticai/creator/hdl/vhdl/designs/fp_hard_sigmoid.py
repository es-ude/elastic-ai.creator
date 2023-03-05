from elasticai.creator.hdl.design_base.network_blocks import NetworkBlock
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation.template import Template


class FPHardSigmoid(NetworkBlock):
    def __init__(
        self,
        *,
        zero_threshold: int,
        one_threshold: int,
        slope: int,
        one: int,
        y_intercept: int,
        total_bits: int,
        frac_bits: int,
        name="fp_hard_sigmoid",
    ):
        super().__init__(
            name=name,
            x_width=total_bits,
            y_width=total_bits,
        )
        d = dict(
            data_width=total_bits,
            frac_width=frac_bits,
            one=one,
            zero_threshold=zero_threshold,
            one_threshold=one_threshold,
            y_intercept=y_intercept,
            slope=slope,
            layer_name=self.name,
        )
        stringified_d = dict(((k, str(v)) for k, v in d.items()))
        self._template = Template(base_name="fp_hard_sigmoid", **stringified_d)

    def save_to(self, destination: Path) -> None:
        ...
