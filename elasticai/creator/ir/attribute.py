from typing import TypeAlias

SizeT: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]
Attribute: TypeAlias = (
    int | float | str | tuple["Attribute", ...] | dict[str, "Attribute"]
)
