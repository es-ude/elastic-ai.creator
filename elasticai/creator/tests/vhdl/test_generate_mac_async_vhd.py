from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.generator.mac_async import MacAsync


class GenerateMacAsyncVhdTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        expected_code = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "entity mac_async is",
            "generic (",
            "DATA_WIDTH : integer := 16;",
            "FRAC_WIDTH : integer := 8",
            ");",
            "port (",
            "x1 : in signed(DATA_WIDTH-1 downto 0);",
            "x2 : in signed(DATA_WIDTH-1 downto 0);",
            "w1 : in signed(DATA_WIDTH-1 downto 0);",
            "w2 : in signed(DATA_WIDTH-1 downto 0);",
            "b : in signed(DATA_WIDTH-1 downto 0);",
            "y : out signed(DATA_WIDTH-1 downto 0)",
            ");",
            "end entity mac_async;",
            "architecture mac_async_rtl of mac_async is",
            "signal product_1 : signed(DATA_WIDTH-1 downto 0);",
            "signal product_2 : signed(DATA_WIDTH-1 downto 0);",
            "begin",
            "-- behavior: y=w1*x1+w2*x2+b",
            "product_1 <= shift_right((x1 * w1), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
            "product_2 <= shift_right((x2 * w2), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
            "y <= product_1 + product_2 + b;",
            "end architecture mac_async_rtl;"
        ]
        component_name = "mac_async"
        data_width = 16
        frac_width = 8
        mac_async = MacAsync(component_name, data_width, frac_width)
        generated_code = list(mac_async())
        self.assertEqual(expected_code, generated_code)
