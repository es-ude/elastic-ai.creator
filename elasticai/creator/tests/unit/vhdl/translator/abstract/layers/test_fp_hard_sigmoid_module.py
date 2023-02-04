import unittest

from elasticai.creator.vhdl.code_files.fp_hard_sigmoid_file import FPHardSigmoidFile
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.translator.abstract.layers.fp_hard_sigmoid_module import (
    FPHardSigmoidModule,
)


class FPHardSigmoidModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fp_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
        self.module = FPHardSigmoidModule(
            layer_id="0", fixed_point_factory=self.fp_factory
        )

    def test_components_return_only_one_component(self) -> None:
        components = list(self.module.files)
        self.assertEqual(len(components), 1)
        self.assertEqual(type(components[0]), FPHardSigmoidFile)

    def test_components_component_args_are_correctly_set(self) -> None:
        component = list(self.module.files)[0]
        self.assertEqual(component.single_line_parameters["zero_threshold"], str(self.fp_factory(-3).to_signed_int()))  # type: ignore
        self.assertEqual(component.single_line_parameters["one_threshold"], str(self.fp_factory(3).to_signed_int()))  # type: ignore
        self.assertEqual(component.single_line_parameters["slope"], str(self.fp_factory(0.125).to_signed_int()))  # type: ignore
        self.assertEqual(component.single_line_parameters["y_intercept"], str(self.fp_factory(0.5).to_signed_int()))  # type: ignore
