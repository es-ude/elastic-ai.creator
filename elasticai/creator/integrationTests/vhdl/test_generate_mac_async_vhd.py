from elasticai.creator.integrationTests.vhdl.test_case import VHDLFileTest
from elasticai.creator.vhdl.generator.mac_async import MacAsync


class GenerateMacAsyncVhdTest(VHDLFileTest):
    def test_compare_files(self) -> None:
        component_name = "mac_async"
        data_width = 16
        frac_width = 8
        mac_async = MacAsync(component_name, data_width, frac_width)
        generated_code = list(mac_async())
        self.compareToFile("expected_mac_async.vhd", generated_code)
