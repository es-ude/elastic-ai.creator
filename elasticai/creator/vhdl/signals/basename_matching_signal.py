from typing import Any, Optional

from elasticai.creator.vhdl.typing import Identifiable


class BaseNameMixin(Identifiable):
    def __init__(self, basename: str, prefix: Optional[str] = None):
        super().__init__()
        self._basename = basename
        self._prefix = prefix

    def id(self) -> str:
        basename = self._basename
        prefix = self._prefix
        return basename if (prefix == "" or prefix is None) else f"{prefix}_{basename}"


class BaseNameMatchingMixin(BaseNameMixin):
    def matches(self, other: Any) -> bool:
        if isinstance(other, BaseNameMixin):
            return self._basename == other._basename
        return False
