from elasticai.creator.hdl.design_base.acceptor import Acceptor


class Signal(Acceptor):
    def __init__(self, name: str, width: int, accepted_names: list[str]):
        self._name = name
        self._width = width
        self._accepted_names = tuple(accepted_names)

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    def __hash__(self):
        return hash((self.name, self.width, self._accepted_names))

    def accepts(self, other: "Signal") -> bool:
        return other.name in self._accepted_names and self.width == other.width
