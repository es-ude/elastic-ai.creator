from io import StringIO

from elasticai.creator.integrationTests.vhdl.test_case import VHDLFileTest

from elasticai.creator.vhdl.generator.lstm_cell import LstmCell


class GenerateLSTMCellVhdTest(VHDLFileTest):
    def test_compare_files(self) -> None:

        lstm_cell_code = LstmCell(
            component_name="lstm_cell", data_width=16, frac_width=8
        )
        lstm_cell_code = list(lstm_cell_code())
        self.compareToFile("expected_lstm_cell.vhd", lstm_cell_code)
