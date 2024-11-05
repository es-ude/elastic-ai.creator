import dataclasses
from dataclasses import dataclass

from .attribute import AttributeT


@dataclass(eq=True, frozen=True)
class Edge:
    src: str
    sink: str
    attributes: dict[str, AttributeT] = dataclasses.field(default_factory=dict)

    @classmethod
    def _filter_attributes(cls, d: dict[str, AttributeT]) -> dict[str, AttributeT]:
        return dict((k, v) for k, v in d if k not in ("src", "sink"))

    def as_dict(self) -> dict[str, AttributeT]:
        return dict(src=self.src, sink=self.sink) | self.attributes

    @classmethod
    def from_dict(cls, data: dict[str, AttributeT]) -> "Edge":
        return cls(
            src=data["src"], sink=data["sink"], attributes=cls._filter_attributes(data)
        )
