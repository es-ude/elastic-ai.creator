from itertools import chain
from typing import Iterable, Callable, Union

from elasticai.creator.resource_utils import read_text
from vhdl.code import Code, CodeFile
from vhdl.templates.utils import expand_template, expand_multiline_template


class VHDLFile(CodeFile):
    def save_to(self, prefix: str):
        pass

    _template_package = "elasticai.creator.vhdl.templates"

    def __init__(
        self,
        name: str,
        parameters: Union[
            dict[str, Union[str, Iterable[str]]],
            Callable[[], dict[str, Union[str, Iterable[str]]]],
        ] = lambda: {},
    ) -> None:
        self._name = name
        parameters = parameters() if callable(parameters) else parameters
        (
            self._parameters,
            self._multiline_parameters,
        ) = self._split_single_and_multiline_parameters(parameters)

    @staticmethod
    def _split_single_and_multiline_parameters(parameters: dict):
        single_line_parameters = dict(
            filter(lambda i: isinstance(i[1], str), parameters.items())
        )
        multiline_parameters = dict(
            filter(lambda i: not isinstance(i[1], str), parameters.items())
        )
        return single_line_parameters, multiline_parameters

    @property
    def single_line_parameters(self) -> dict[str, str]:
        return dict(**self._parameters)

    @property
    def parameters(self):
        return dict(chain(self._parameters.items(), self._multiline_parameters.items()))

    @property
    def multiline_parameters(self):
        return dict(**self._multiline_parameters)

    @property
    def name(self) -> str:
        return f"{self._name}.vhd"

    def code(self) -> Code:
        template = read_text(self._template_package, f"{self._name}.tpl.vhd")
        template = expand_template(template, **self._parameters)
        template = expand_multiline_template(template, **self._multiline_parameters)
        yield from template
