from collections.abc import Iterable, Iterator, Sequence
from itertools import repeat
from string import Template
from typing import Any, Union


def expand_multiline_template(
    template: Union[str, Iterable[str]], **kwargs: Iterable[str]
) -> Iterator[str]:
    """Expand a template field to multiple lines, while keeping indentation.
    Example:
        >>> template = "\\t$my_key"
        >>> values = ["hello,", "world", "!"]
        >>> "\\n".join(expand_multiline_template(template, my_key=values))
        '\\thello,\\n\\tworld\\n\\t!'
    """
    lines = _unify_template_datatype(template)
    for line in lines:
        contains_no_key = True
        for key in kwargs:
            if f"${key}" in line:
                contains_no_key = False
                for placeholder_line, value in zip(repeat(line), kwargs[key]):
                    t = Template(placeholder_line)
                    yield t.safe_substitute({key: value})
                break
        if contains_no_key:
            yield line


def _unify_template_datatype(template: Union[str, Iterable[str]]) -> Iterable[str]:
    if hasattr(template, "splitlines") and callable(template.splitlines):
        lines = template.splitlines()
    else:
        lines = template
    return lines


def expand_template(template: str | Iterable[str], **kwargs: str) -> Iterable[str]:
    if isinstance(template, str):
        yield Template(template).safe_substitute(kwargs)
    else:
        for line in template:
            yield Template(line).safe_substitute(kwargs)
