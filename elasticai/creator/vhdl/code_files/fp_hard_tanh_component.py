from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointConfig
from elasticai.creator.vhdl.templates import VHDLTemplate


class FPHardTanhComponent:
    def __init__(
        self,
        min_val: FixedPoint,
        max_val: FixedPoint,
        fixed_point_factory: FixedPointConfig,
        layer_id: str,
    ):
        d = dict(
            data_width=fixed_point_factory.total_bits,
            frac_width=fixed_point_factory.frac_bits,
            min_val=min_val.to_signed_int(),
            max_val=max_val.to_signed_int(),
            layer_name=layer_id,
        )
        stringified_d = dict(((k, str(v)) for k, v in d.items()))
        self._name = f"fp_hard_tanh_{layer_id}"
        self.template = VHDLTemplate(base_name="fp_hard_tanh", **stringified_d)

    @property
    def single_line_parameters(self):
        return self.template.single_line_parameters

    @property
    def name(self) -> str:
        return f"{self._name}.vhd"

    def lines(self) -> list[str]:
        return self.template.lines()
