from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.generator.mac_async import MacAsync


class GenerateMacAsyncVhdTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        expected_code = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "entity mac_async is",
            "\tgeneric (",
            "\t\tDATA_WIDTH : integer := 16;",
            "\t\tFRAC_WIDTH : integer := 8",
            "\t);",
            "\tport (",
            "\t\tx1 : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tx2 : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tw1 : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tw2 : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tb : in signed(DATA_WIDTH-1 downto 0);",
            "\t\ty : out signed(DATA_WIDTH-1 downto 0)",
            "\t);",
            "end entity mac_async;",
            "architecture mac_async_rtl of mac_async is",
            "\tsignal product_1 : signed(DATA_WIDTH-1 downto 0);",
            "\tsignal product_2 : signed(DATA_WIDTH-1 downto 0);",
            "begin",
            "\tproduct_1 <= shift_right((x1 * w1), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
            "\tproduct_2 <= shift_right((x2 * w2), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
            "\ty <= product_1 + product_2 + b;",
            "end architecture mac_async_rtl;"
        ]
        component_name = "mac_async"
        data_width = 16
        frac_width = 8
        mac_async = MacAsync(component_name, data_width, frac_width)
        generated_code = list(mac_async())
        self.assertEqual(expected_code, generated_code)
