import re
from typing import overload


@overload
def extract_rom_values(text: str) -> tuple[str, ...]:
    ...


@overload
def extract_rom_values(text: list[str]) -> tuple[str, ...]:
    ...


def extract_rom_values(text: str | list[str]) -> tuple[str, ...]:
    if not isinstance(text, list):
        text = [text]
    values: tuple[str, ...] = tuple()
    for line in text:
        match = re.match(
            r'.*\(x"([a-f0-9]+(",\s?x"[a-f0-9]+)*)"\)',
            line,
        )
        if match is not None:
            array = match.group(1)
            values = tuple(re.split(r'(?:",\s?x")', array))

    return values
