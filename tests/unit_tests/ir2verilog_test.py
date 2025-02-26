from elasticai.creator.ir2verilog import TemplateDirector


class TestVerilogTemplate:
    def test_can_set_localparam(self):
        template = (
            TemplateDirector()
            .set_prototype("localparam DATA_WIDTH = 4'd8;")
            .localparam("DATA_WIDTH")
            .build()
        )

        assert (
            template.substitute({"DATA_WIDTH": "'h16"})
            == "localparam DATA_WIDTH = 'h16;"
        )

    def test_can_set_parameter_before_comma(self):
        template = (
            TemplateDirector()
            .set_prototype("parameter DATA_WIDTH = 4'd8,")
            .parameter("DATA_WIDTH")
            .build()
        )

        assert (
            template.substitute({"DATA_WIDTH": "'h16"})
            == "parameter DATA_WIDTH = 'h16,"
        )

    def test_keep_new_line_between_define_switches(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH\n`define DATA_LENGTH")
            .define_scoped_switch("DATA_WIDTH")
            .define_scoped_switch("DATA_LENGTH")
            .build()
        )

        assert (
            template.substitute(
                {
                    "DATA_WIDTH": True,
                    "DATA_LENGTH": True,
                }
            )
            == "`define DATA_WIDTH\n`define DATA_LENGTH"
        )

    def test_can_set_parameter_before_new_line(self):
        template = (
            TemplateDirector()
            .set_prototype("""parameter DATA_WIDTH = 4'd8
""")
            .parameter("DATA_WIDTH")
            .build()
        )

        assert (
            template.substitute({"DATA_WIDTH": "'h16"})
            == "parameter DATA_WIDTH = 'h16\n"
        )

    def test_can_undefine_data_width(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH")
            .build()
        )
        template.undef("DATA_WIDTH")
        assert (
            template.substitute({})
            == """// automatically disabled\n// `define DATA_WIDTH 8"""
        )

    def test_can_leave_data_width_defined(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH")
            .build()
        )
        template.define("DATA_WIDTH")
        assert template.substitute({}) == "`define DATA_WIDTH 8"

    def test_can_replace_module_name(self):
        template = (
            TemplateDirector()
            .set_prototype("module FILTER_FIR_HALF#(")
            .add_module_name()
            .build()
        )
        assert (
            template.substitute({"module_name": "FILTER_FIR_MY_NAME_HALF"})
            == "module FILTER_FIR_MY_NAME_HALF#("
        )

    def test_automatically_use_module_prefix_for_defines(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH")
            .build()
        )
        assert (
            template.substitute({"module_name": "FILTER_FIR_HALF"})
            == "`define FILTER_FIR_HALF_DATA_WIDTH 8"
        )

    def test_later_occurences_of_defined_names_are_scoped(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH 8\nDATA_WIDTH + 10")
            .define_scoped_switch("DATA_WIDTH")
            .build()
        )
        assert (
            template.substitute({"module_name": "FILTER_FIR_HALF"})
            == "`define FILTER_FIR_HALF_DATA_WIDTH 8\nFILTER_FIR_HALF_DATA_WIDTH + 10"
        )
