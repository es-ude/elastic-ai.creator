from typing import Any, Optional

from elasticai.creator.vhdl.signals.base_signal import (
    BaseInSignal,
    OutSignal,
    Reversible,
)
from elasticai.creator.vhdl.signals.basename_matching_signal import (
    BaseNameMatchingMixin,
    BaseNameMixin,
)
from elasticai.creator.vhdl.signals.std_logic_signal_definitions import (
    Vector,
    create_std_logic_definition,
    create_std_logic_vector_definition,
)


class _StdLogicDefinitionWithDefaultMixin(BaseNameMixin):
    def __init__(
        self,
        basename: str,
        prefix: Optional[str] = None,
        default_value: Optional[str] = None,
    ):
        super().__init__(basename=basename, prefix=prefix)
        self._default_value = default_value

    def definition(self) -> str:
        return create_std_logic_definition(self, self._default_value)


class LogicInSignal(
    _StdLogicDefinitionWithDefaultMixin,
    BaseNameMatchingMixin,
    BaseInSignal,
    Reversible["LogicOutSignal"],
):
    @staticmethod
    def _direction_matches(other: Any) -> bool:
        return isinstance(other, LogicOutSignal)

    def matches(self, other: Any) -> bool:
        return self._direction_matches(other) and super().matches(other)

    def reverse(self) -> "LogicOutSignal":
        return LogicOutSignal(
            basename=self._basename,
            prefix=self._prefix,
            default_value=self._default_value,
        )


class LogicOutSignal(
    _StdLogicDefinitionWithDefaultMixin,
    BaseNameMixin,
    OutSignal,
    Reversible[LogicInSignal],
):
    def reverse(self) -> "LogicInSignal":
        return LogicInSignal(
            basename=self._basename,
            prefix=self._prefix,
            default_value=self._default_value,
        )


class _LogicVectorDefinitionWithDefaultMixin(BaseNameMixin, Vector):
    def __init__(
        self,
        basename: str,
        width: int,
        prefix: Optional[str] = None,
        default_value: Optional[str] = None,
    ):
        super().__init__(basename, prefix)
        self._default_value = default_value
        self._width = width

    def width(self) -> int:
        return self._width

    def definition(self):
        return create_std_logic_vector_definition(self, self._default_value)


class LogicInVectorSignal(
    _LogicVectorDefinitionWithDefaultMixin,
    BaseNameMatchingMixin,
    Reversible["LogicOutVectorSignal"],
):
    def _width_matches(self, other: Any) -> bool:
        if isinstance(other, _LogicVectorDefinitionWithDefaultMixin):
            return self.width() == other.width()
        return False

    @staticmethod
    def _direction_matches(other: Any) -> bool:
        return isinstance(other, LogicOutVectorSignal)

    def matches(self, other: Any) -> bool:
        return (
            self._direction_matches(other)
            and self._width_matches(other)
            and super().matches(other)
        )

    def reverse(self) -> "LogicOutVectorSignal":
        return LogicOutVectorSignal(
            basename=self._basename,
            prefix=self._prefix,
            width=self._width,
            default_value=self._default_value,
        )


class LogicOutVectorSignal(
    _LogicVectorDefinitionWithDefaultMixin,
    BaseNameMixin,
    Reversible["LogicInVectorSignal"],
):
    def reverse(self) -> "LogicInVectorSignal":
        return LogicInVectorSignal(
            basename=self._basename,
            prefix=self._prefix,
            width=self._width,
            default_value=self._default_value,
        )
