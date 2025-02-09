from typing import TypeAlias

SizeT: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]
Attribute: TypeAlias = (
    int
    | float
    | str
    | list["Attribute"]
    | dict[str, "Attribute"]
    | list[int]
    | tuple["Attribute", ...]
)
