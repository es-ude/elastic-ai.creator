from itertools import filterfalse

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.generator.generator_functions import generate_signal_definitions_for_lstm
from elasticai.creator.vhdl.language import Code


class LstmCell:
    def __init__(self, component_name, data_width, frac_width):
        self.component_name = component_name
        self.data_width = data_width
        self.frac_width = frac_width

    def __call__(self) -> Code:
        template = read_text("elasticai.creator.vhdl.generator.templates", "lstm_cell.tpl.vhd")
        signal_defs = generate_signal_definitions_for_lstm(data_width=self.data_width, frac_width=self.frac_width)
        parameters = signal_defs | {'data_width': self.data_width, 'frac_width': self.frac_width}
        code = template.format(**parameters)

        def line_is_empty(line):
            return len(line) == 0
        yield from filterfalse(line_is_empty, map(str.strip, code.splitlines()))
