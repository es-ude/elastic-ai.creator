from typing import Union

from elasticai.creator.vhdl.code import Code


def extract_section(
    begin: Union[str, Code], end: Union[str, Code], lines: Code
) -> list[list[str]]:
    extract = False
    content: list[list[str]] = []
    current_section: list[str] = []
    begin = list(begin) if not isinstance(begin, str) else [begin]
    end = list(end) if not isinstance(end, str) else [end]
    look_back = len(begin)
    look_ahead = len(end)
    lines = list(lines)
    i = 0
    last_i = len(lines)
    while i < last_i:
        look_ahead_window = lines[i : i + look_ahead]
        look_back_window = lines[i : i + look_back]

        if not extract and look_back_window == begin:
            extract = True
            i = i + look_back - 1
        elif extract and look_ahead_window == end:
            extract = False
            content.append(current_section)
            current_section = []
        elif extract:
            current_section.append(lines[i])
        i += 1

    if extract:
        raise ValueError(f"reached end of code before end: {end}")
    return content
