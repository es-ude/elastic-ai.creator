import unittest

from elasticai.creator.vhdl.code_files.fp_hard_tanh_component import FPHardTanhComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.fp_hard_tanh_module import (
    FPHardTanhModule,
)


class FPHardTanhModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
        self.module = FPHardTanhModule(
            layer_id="0", fixed_point_factory=self.fp_factory
        )

    def test_components_return_only_one_component(self) -> None:
        components = list(self.module.files)
        self.assertEqual(len(components), 1)
        self.assertEqual(type(components[0]), FPHardTanhComponent)

    def test_components_component_args_are_correctly_set(self) -> None:
        component = list(self.module.files)[0]
        self.assertEqual(component.single_line_parameters["min_val"], str(self.fp_factory(-1).to_signed_int()))  # type: ignore
        self.assertEqual(component.single_line_parameters["max_val"], str(self.fp_factory(1).to_signed_int()))  # type: ignore
