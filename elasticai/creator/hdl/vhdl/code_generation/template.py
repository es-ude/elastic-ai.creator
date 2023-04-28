from collections.abc import Iterable

from elasticai.creator.hdl.code_generation.template import (
    InMemoryTemplateConfig,
    InProjectTemplateConfig,
    TemplateConfig,
    TemplateExpander,
)


class _TemplateBase:
    def __init__(self, config: TemplateConfig) -> None:
        self._internal_template = TemplateExpander(config=config)

    def update_parameters(self, **parameters: str | Iterable[str]):
        self._internal_template.config.parameters.update(parameters)

    def lines(self) -> list[str]:
        return self._internal_template.lines()


class InProjectTemplate(_TemplateBase):
    def __init__(
        self,
        base_name: str,
        package: str = "elasticai.creator.hdl.vhdl.template_resources",
        suffix: str = ".tpl.vhd",
    ) -> None:
        super().__init__(
            config=InProjectTemplateConfig(
                file_name=f"{base_name}{suffix}", package=package, parameters=dict()
            )
        )


class InMemoryTemplate(_TemplateBase):
    def __init__(self, template: Iterable[str]) -> None:
        super().__init__(
            config=InMemoryTemplateConfig(template=template, parameters=dict())
        )
