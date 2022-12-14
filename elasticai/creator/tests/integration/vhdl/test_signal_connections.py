from elasticai.creator.tests.code_utilities_for_testing import CodeTestCase
from elasticai.creator.tests.integration.vhdl.models_for_testing import FirstModel
from elasticai.creator.vhdl.code import Code


class SignalConnectionsTest(CodeTestCase):
    def setUp(self) -> None:
        self.model = FirstModel()
        self.model.elasticai_tags.update(
            {
                "x_address_width": 1,
                "y_address_width": 1,
                "y_width": 16,
                "x_width": 16,
            }
        )
        code = CodeTestCase.unified_vhdl_from_module(self.model.translate())
        self.actual_connections: Code = self.extract_section_from_code(
            begin="begin",
            end="fp_linear : entity work.fp_linear(rtl)",
            lines=code,
        )[0]

    def test_x_is_connected_to_fp_linear_x(self):
        self.check_contains_all_expected_lines(
            expected=["fp_linear_x <= x;"], actual=self.actual_connections
        )
