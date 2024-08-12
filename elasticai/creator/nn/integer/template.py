from pathlib import Path

from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    TemplateExpander,
)


class Template(InProjectTemplate):
    def __init__(
        self, package: str, file_name: str, parameters: dict[str, str | list[str]]
    ) -> None:
        super().__init__(package=package, file_name=file_name, parameters=parameters)

    def fill(self) -> list[str]:
        expander = TemplateExpander(self)
        unfilled_variables = expander.unfilled_variables()
        if len(unfilled_variables) > 0:
            raise KeyError(
                "Template is not filled completly. The following variables are"
                f" unfilled: {', '.join(unfilled_variables)}."
            )
        return expander.lines()


def save_code(destination: Path, code: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(mode="w") as out_file:
        out_file.write("\n".join(code))
