from typing import Iterable

from elasticai.creator.resource_utils import read_text


class MacAsync:
    def __init__(self, component_name, data_width, frac_width):
        self.component_name = component_name
        self.data_width = data_width
        self.frac_width = frac_width

    def __call__(self) -> Iterable[str]:
        template = read_text("elasticai.creator.vhdl.generator.templates", "mac_async.tpl.vhd")

        source_code = template.format(entity_name=self.component_name, data_width=self.data_width, frac_width=self.frac_width)
        yield from map(lambda s: s.strip(" "), source_code.splitlines())
