from typing import Optional

from ._signal_base import _SignalBase


class _VectorSignalImpl(_SignalBase):
    def __init__(
        self, *, accepted_names: list[str], id: str, width: int, default: Optional[str]
    ) -> None:
        super().__init__(accepted_names=accepted_names, id=id, default=default)
        self.width = width

    def definition(self, prefix: str = "") -> str:
        return (
            f"signal {prefix}{self.id()} :"
            f" std_logic_vector({self.width - 1} downto 0){self._default};"
        )

    def accepts(self, other: _SignalBase) -> bool:
        if isinstance(other, _VectorSignalImpl):
            return other.id() in self._accepted_names and self.width == other.width
        return False
