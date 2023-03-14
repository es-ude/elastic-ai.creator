from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
)


class Template:
    def __init__(
        self,
        base_name: str,
        package: str = "elasticai.creator.hdl.vhdl.template_resources",
        suffix: str = ".tpl.vhd",
    ):
        self._template_name = base_name
        self._internal_template = TemplateExpander(
            TemplateConfig(
                file_name=f"{base_name}{suffix}", package=package, parameters=dict()
            )
        )

    def update_parameters(self, **parameters: str | list[str]):
        self._internal_template.config.parameters.update(parameters)

    def lines(self) -> list[str]:
        return self._internal_template.lines()
