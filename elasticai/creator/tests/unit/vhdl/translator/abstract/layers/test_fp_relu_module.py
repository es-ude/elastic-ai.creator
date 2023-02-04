import unittest

from elasticai.creator.vhdl.code_files.fp_relu_component import FPReLUComponent
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.fp_relu_module import (
    FPReLUModule,
)


class FPReluModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
        self.module = FPReLUModule(layer_id="0", fixed_point_factory=self.fp_factory)

    def test_components_return_only_one_component(self) -> None:
        components = list(self.module.files)
        self.assertEqual(len(components), 1)
        self.assertEqual(type(components[0]), FPReLUComponent)

    def test_components_component_args_are_correctly_set(self) -> None:
        component = list(self.module.files)[0]
        self.assertEqual(component.fixed_point_factory, self.fp_factory)  # type: ignore
