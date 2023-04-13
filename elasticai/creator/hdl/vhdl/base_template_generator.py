from typing import Literal

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)

"""
We want to programmatically use designs that are either hand-written or based on hand-written templates
these designs need to adhere to a well defined protocol. As a hardware designer you specify your protocol
and the expected version of the elasticai.creator in a file called `design_meta.toml` that lives in the
same folder as your `layer.py` that defines how the hdl code for the hardware design and the behaviour of
the corresponding neural network layer in software.

To help you as hardware designer stick to specific protocol you can generate a base template, that you
can use as a starting point to develop your design.
"""


class BaseTemplateGenerator:
    def __init__(self, pass_through: list[Literal["x", "enable", "y_address"]]):
        self.pass_through = pass_through

    def generate(self) -> str:
        return _generate_base_template_for_hw_block_protocol(
            pass_through=self.pass_through
        )


def _generate_base_template_for_hw_block_protocol(
    pass_through: list[Literal["enable", "y_address", "x"]]
) -> str:
    pass_through_map = {
        "x": "y",
        "y_address": "x_address",
        "enable": "done",
    }
    pass_through_lines = []
    for signal in pass_through:
        pass_through_lines.append(f"{pass_through_map[signal]} <- {signal};")
    config = TemplateConfig(
        package=module_to_package(
            _generate_base_template_for_hw_block_protocol.__module__
        ),
        file_name="base_template.tpl.vhd",
        parameters={"pass_through": "\n".join(pass_through_lines)},
    )
    return "\n".join(TemplateExpander(config).lines())
