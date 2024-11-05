import dataclasses
from dataclasses import dataclass

from .attribute import AttributeT


@dataclass
class Node:
    name: str
    type: str
    attributes: dict[str, AttributeT] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, AttributeT]:
        return dict(name=self.name, type=self.type) | self.attributes

    @classmethod
    def from_dict(cls, data: dict[str, AttributeT]) -> "Node":
        return Node(
            name=data["name"],
            type=data["type"],
            attributes=dict(
                (k, v) for k, v in data.items() if k not in ("name", "type")
            ),
        )
