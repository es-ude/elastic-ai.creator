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
        self._pass_through = pass_through
        self._pass_through_map = {
            "x": "y",
            "y_address": "x_address",
            "enable": "done",
        }
        self._base_config = TemplateConfig(
            package=module_to_package(self.__module__),
            file_name="base_template.tpl.vhd",
            parameters={},
        )

    def generate(self) -> str:
        self._check_validity()
        self._configure_base_template()
        return self._expand_base_template()

    def _configure_base_template(self):
        self._base_config.parameters["pass_through"] = self._create_pass_through_lines()

    def _create_pass_through_lines(self) -> str:
        pass_through_lines = []
        for signal in self._pass_through:
            pass_through_lines.append(f"{self._pass_through_map[signal]} <- {signal};")
        return "\n".join(pass_through_lines)

    def _expand_base_template(self) -> str:
        return "\n".join(TemplateExpander(self._base_config).lines())

    def _check_validity(self):
        if self._values_are_invalid():

            def to_string(values):
                return ", ".join(values)

            raise ValueError(
                f"found: {to_string(self._pass_through)}, expected one or more of"
                f" {to_string(sorted(self._valid_values))}"
            )

    @property
    def _valid_values(self) -> set[str]:
        return set(self._pass_through_map.keys())

    def _values_are_invalid(self) -> bool:
        actual_values = set(self._pass_through)
        return not actual_values.issubset(self._valid_values)
