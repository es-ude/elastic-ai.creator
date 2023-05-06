from collections.abc import Iterable

from elasticai.creator.hdl.code_generation.template import (
    InMemoryTemplate,
    InProjectTemplate,
    Template,
    TemplateExpander,
)


class _VHDLTemplateBase:
    def __init__(self, template: Template) -> None:
        self._internal_template = template

    def update_parameters(self, **parameters: str | list[str]):
        self._internal_template.parameters.update(parameters)

    def lines(self) -> list[str]:
        return TemplateExpander(self._internal_template).lines()


class InProjectVHDLTemplate(_VHDLTemplateBase):
    def __init__(
        self,
        base_name: str,
        package: str = "elasticai.creator.hdl.vhdl.template_resources",
        suffix: str = ".tpl.vhd",
    ) -> None:
        super().__init__(
            template=InProjectTemplate(
                file_name=f"{base_name}{suffix}", package=package, parameters=dict()
            )
        )


class InMemoryVHDLTemplate(_VHDLTemplateBase):
    def __init__(self, content: list[str]) -> None:
        super().__init__(template=InMemoryTemplate(content=content, parameters=dict()))
