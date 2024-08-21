from pathlib import Path

from ._resource_utils import read_text
from ._template_expander import Template, TemplateExpander


def fill_template(template: Template) -> list[str]:
    expander = TemplateExpander(template)
    unfilled_variables = expander.unfilled_variables()
    if len(unfilled_variables) > 0:
        raise KeyError(
            "Template is not filled completly. The following variables are"
            f" unfilled: {', '.join(unfilled_variables)}."
        )
    return expander.lines()


def save_template(template: Template, destination: Path) -> None:
    lines = fill_template(template)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as out_file:
        out_file.writelines(f"{line}\n" for line in lines)


class InProjectTemplate:
    def __init__(
        self, package: str, file_name: str, parameters: dict[str, str | list[str]]
    ) -> None:
        self.package = package
        self.file_name = file_name
        self.parameters = parameters
        self.content = list(read_text(self.package, self.file_name))

    @staticmethod
    def module_to_package(module: str) -> str:
        return ".".join(module.split(".")[:-1])
