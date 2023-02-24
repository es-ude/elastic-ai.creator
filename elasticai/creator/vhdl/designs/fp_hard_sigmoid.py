from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointConfig
from elasticai.creator.vhdl.templates import VHDLTemplate

from ._network_blocks import NetworkBlock
from .folder import Folder


class FPHardSigmoid(NetworkBlock):
    def __init__(
        self,
        zero_threshold: FixedPoint,
        one_threshold: FixedPoint,
        slope: FixedPoint,
        y_intercept: FixedPoint,
        fixed_point_factory: FixedPointConfig,
        name=None,
    ):
        super().__init__(
            name=name,
            x_width=fixed_point_factory.total_bits,
            y_width=fixed_point_factory.total_bits,
        )
        d = dict(
            data_width=fixed_point_factory.total_bits,
            frac_width=fixed_point_factory.frac_bits,
            one=fixed_point_factory(1).to_signed_int(),
            zero_threshold=zero_threshold.to_signed_int(),
            one_threshold=one_threshold.to_signed_int(),
            y_intercept=y_intercept.to_signed_int(),
            slope=slope.to_signed_int(),
            layer_name=self.name,
        )
        stringified_d = dict(((k, str(v)) for k, v in d.items()))
        self._template = VHDLTemplate(base_name="fp_hard_sigmoid", **stringified_d)

    def save_to(self, destination: Folder) -> None:
        destination.new_file(f"{self._name}.vhd", self._template.lines())
