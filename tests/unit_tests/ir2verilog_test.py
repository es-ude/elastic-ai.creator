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

    def test_can_set_parameter_before_new_line(self):
        template = (
            TemplateDirector()
            .set_prototype("""parameter DATA_WIDTH = 4'd8""")
            .parameter("DATA_WIDTH")
            .build()
        )
        assert (
            template.substitute({"DATA_WIDTH": "'h16"}) == "parameter DATA_WIDTH = 'h16"
        )

    def test_can_define(self):
        template = (
            TemplateDirector()
            .set_prototype("//`define DATA_WIDTH")
            .define_scoped_switch("DATA_WIDTH", True)
            .build()
        )
        assert template.substitute({}) == "`define DATA_WIDTH"

    def test_can_undefine(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH")
            .define_scoped_switch("DATA_WIDTH", False)
            .build()
        )
        assert template.substitute({}) == "//`define DATA_WIDTH"

    def test_can_leave_defined(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH")
            .define_scoped_switch("DATA_WIDTH", True)
            .build()
        )
        assert template.substitute({}) == "`define DATA_WIDTH"

    def test_can_leave_undefined(self):
        template = (
            TemplateDirector()
            .set_prototype("//`define DATA_WIDTH")
            .define_scoped_switch("DATA_WIDTH", False)
            .build()
        )
        assert template.substitute({}) == "//`define DATA_WIDTH"

    def test_can_define_data_width(self):
        template = (
            TemplateDirector()
            .set_prototype("// `define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH", True)
            .build()
        )
        assert template.substitute({}) == "`define DATA_WIDTH 8"

    def test_can_undefine_data_width(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH", False)
            .build()
        )
        assert template.substitute({}) == """//`define DATA_WIDTH 8"""

    def test_can_leave_data_width_defined(self):
        template = (
            TemplateDirector()
            .set_prototype("`define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH", True)
            .build()
        )
        assert template.substitute({}) == "`define DATA_WIDTH 8"

    def test_can_leave_data_width_undefined(self):
        template = (
            TemplateDirector()
            .set_prototype("//`define DATA_WIDTH 8")
            .define_scoped_switch("DATA_WIDTH", False)
            .build()
        )
        assert template.substitute({}) == "//`define DATA_WIDTH 8"

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
            .define_scoped_switch("DATA_WIDTH", True)
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
            .define_scoped_switch("DATA_WIDTH", True)
            .build()
        )
        assert (
            template.substitute({"module_name": "FILTER_FIR_HALF"})
            == "`define FILTER_FIR_HALF_DATA_WIDTH 8\nFILTER_FIR_HALF_DATA_WIDTH + 10"
        )

    def test_can_replace_arrays(self):
        template = (
            TemplateDirector()
            .set_prototype("localparam DATA = {'d2, 'd3};")
            .localparam("DATA")
            .build()
        )
        assert template.substitute({"DATA": "'d5"}) == "localparam DATA = 'd5;"

    def test_replace_module_of_instance_with_parameter(self):
        template = (
            TemplateDirector()
            .set_prototype("""FILT_BIQUAD#(BITWIDTH_DATA, LENGTH) DUT(
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);""")
            .replace_module_of_instance("FILT_BIQUAD", "MyModule")
            .build()
        )
        assert (
            template.substitute({})
            == """FILT_BIQUAD#(BITWIDTH_DATA, LENGTH) MyModule(
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);"""
        )

    def test_replace_module_of_instance_without_parameter(self):
        template = (
            TemplateDirector()
            .set_prototype("""FILT_BIQUAD DUT(
    .CLK(clk_sys),
    .nRST(nrst),
    .EN(en_dut),
    .START_FLAG(clk_adc),
    .DATA_IN(filter_in),
    .DATA_OUT(dout),
    .DATA_VALID(filter_rdy)
);""")
            .replace_module_of_instance("FILT_BIQUAD", "MyModule")
            .build()
        )
        assert (
            template.substitute({})
            == """FILT_BIQUAD MyModule(
    .CLK(clk_sys),
    .nRST(nrst),
    .EN(en_dut),
    .START_FLAG(clk_adc),
    .DATA_IN(filter_in),
    .DATA_OUT(dout),
    .DATA_VALID(filter_rdy)
);"""
        )

    def test_replace_module_of_instance_with_space(self):
        template = (
            TemplateDirector()
            .set_prototype("""   FILT_BIQUAD #(4) DUT (
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);""")
            .replace_module_of_instance("FILT_BIQUAD", "MyModule")
            .build()
        )
        assert (
            template.substitute({})
            == """FILT_BIQUAD #(4) MyModule (
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);"""
        )

    def test_replace_instance_name_of_module(self):
        template = (
            TemplateDirector()
            .set_prototype("""   FILT_BIQUAD#(4) DUT (
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);""")
            .replace_instance_name("FILT_BIQUAD", "MyModule")
            .build()
        )
        assert (
            template.substitute({})
            == """   MyModule#(4) DUT (
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);"""
        )

    def test_replace_instance_name_of_module_with_space(self):
        template = (
            TemplateDirector()
            .set_prototype("""  FILT_BIQUAD #(4) DUT (
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);""")
            .replace_instance_name("FILT_BIQUAD", "MyModule")
            .build()
        )
        assert (
            template.substitute({})
            == """  MyModule #(4) DUT (
    .CLK(clk_sys),
	  .nRST(nrst),
	  .EN(en_dut),
	  .START_FLAG(clk_adc),
	  .DATA_IN(filter_in),
	  .DATA_OUT(dout),
	  .DATA_VALID(filter_rdy)
);"""
        )
