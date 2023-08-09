import re


def extract_rom_values(text: str | list[str]) -> tuple[str, ...]:
    if not isinstance(text, list):
        text = [text]
    values: tuple[str, ...] = tuple()
    for line in text:
        match = re.match(
            r'.*\("([a-f0-9]+(",\s?"[a-f0-9]+)*)"\)',
            line,
        )
        if match is not None:
            array = match.group(1)
            values = tuple(re.split(r'(?:",\s?")', array))

    return values
