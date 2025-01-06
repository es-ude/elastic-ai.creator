import re
from abc import ABC, abstractmethod


class ParametrizationListParser:
    tokens = {
        "PARAMLIST": r"ParametrizationList",
        "CPAR": r"\)",
        "OPAR": r"\(",
        "TEXT": r"[a-zA-Z0-9_,=]+\s?",
        "COLON": r":",
    }

    def __init__(self) -> None:
        self._state = _ReadingKeyOutsideParamList()

    def parse(self, serialized):
        lists_string = re.findall(r"ModuleDict\(\s*(.*)\s*\)\s*\Z", serialized)[0]
        regexp = re.compile(
            "|".join(
                [
                    "(?P<{}>{})".format(t_name, t_exp)
                    for t_name, t_exp in self.tokens.items()
                ]
            )
        )
        for match in re.finditer(regexp, lists_string):
            kind = match.lastgroup
            token = match.group()
            match kind:
                case "PARAMLIST":
                    self._state = self._state.parametrization_list()
                case "CPAR":
                    self._state = self._state.cpar()
                case "OPAR":
                    self._state = self._state.opar()
                case "TEXT":
                    self._state = self._state.text(token)
                case "COLON":
                    self._state = self._state.colon()
        result = self._state.parametrizations
        self._state = _ReadingKeyOutsideParamList()
        return result


class _State(ABC):
    def __init__(self) -> None:
        self.num_opened_pars = 0
        self.current_tensor_name = "unknown"
        self.current_repr: list[str] = []
        self.parametrizations: dict[str, list[str]] = {}

    @classmethod
    def from_other(cls, other: "_State") -> "_State":
        s = cls()
        s.num_opened_pars = other.num_opened_pars
        s.current_tensor_name = other.current_tensor_name
        s.parametrizations = other.parametrizations
        s.current_repr = other.current_repr
        return s

    def opar(self) -> "_State":
        self.num_opened_pars += 1
        return self

    def cpar(self) -> "_State":
        self.num_opened_pars -= 1
        return self

    @abstractmethod
    def colon(self) -> "_State": ...
    @abstractmethod
    def text(self, token: str) -> "_State": ...
    @abstractmethod
    def parametrization_list(self) -> "_State": ...


class _ReadingKeyInsideParamList(_State):
    def parametrization_list(self) -> "_State":
        raise ValueError()

    def colon(self):
        return _ReadingRepr.from_other(self)

    def cpar(self):
        self.num_opened_pars -= 1
        if self.num_opened_pars == 0:
            return _ReadingKeyOutsideParamList.from_other(self)
        return self

    def text(self, token: str):
        return self


class _ReadingRepr(_State):
    def _append(self, token: str):
        self.current_repr.append(token)

    def opar(self):
        self._append("(")
        return super().opar()

    def cpar(self):
        self._append(")")
        self.num_opened_pars -= 1
        if self.num_opened_pars == 1:
            self.parametrizations[self.current_tensor_name].append(
                "".join(self.current_repr)
            )
            self.current_repr.clear()
            return _ReadingKeyInsideParamList.from_other(self)
        return self

    def colon(self):
        self._append(":")
        return self

    def text(self, token: str):
        self._append(token)
        return self

    def parametrization_list(self) -> "_State":
        raise ValueError()


class _ReadingKeyOutsideParamList(_State):
    def colon(self) -> "_State":
        return self

    def text(self, token: str) -> "_State":
        self.current_tensor_name = token
        self.parametrizations[token] = []
        return self

    def parametrization_list(self) -> "_State":
        return _ReadingKeyInsideParamList.from_other(self)
