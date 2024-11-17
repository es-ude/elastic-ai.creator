from typing import TypeAlias

SizeT: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]
AttributeT: TypeAlias = (
    int | float | str | tuple["AttributeT", ...] | dict[str, "AttributeT"]
)
