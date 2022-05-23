from itertools import repeat
from string import Template
from typing import Iterator, Sequence


def expand_multiline_template(template: str, **kwargs: Sequence) -> Iterator[str]:
    """Expand a template field to multiple lines, while keeping indentation.
    Example:
        >>> template = "\\t$my_key"
        >>> values = ["hello,", "world", "!"]
        >>> "\\n".join(expand_multiline_template(template, my_key=values))
        '\\thello,\\n\\tworld\\n\\t!'
    """
    lines = template.splitlines()
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
