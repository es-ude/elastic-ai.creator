import re
from abc import abstractmethod
from dataclasses import dataclass, field


class _StringValue:
    def __init__(self) -> None:
        self._data: list[str] = []

    def append(self, data: str):
        self._data.append(data)

    def finalize(self) -> str:
        return "".join(self._data)


class _DictValue:
    def __init__(self) -> None:
        self._data = {}

    def __getitem__(self, item: str) -> str | dict:
        return self._data[item]

    def __setitem__(self, key: str, value: str | dict) -> None:
        self._data[key] = value

    def finalize(self) -> dict[str, str | dict]:
        return self._data


@dataclass
class _StateRecord:
    opened_pars: int = 0
    result: dict[str, str | dict] = field(default_factory=dict)
    key: str = "unknown"
    current_val: _StringValue | _DictValue = field(default_factory=_StringValue)


class _State:
    def __init__(self):
        self._record = _StateRecord()

    def finalize(self):
        self._record.result[self._record.key] = self._record.current_val.finalize()

    @property
    def current_value(self):
        return self._record.current_val

    @current_value.setter
    def current_value(self, value):
        self._record.current_val = value

    @property
    def key(self):
        return self._record.key

    @key.setter
    def key(self, value):
        self._record.key = value

    @property
    def opened_pars(self) -> int:
        return self._record.opened_pars

    @abstractmethod
    def do_opar(self): ...

    @abstractmethod
    def do_cpar(self): ...

    def opar(self, _):
        self._record.opened_pars += 1
        return self.do_opar()

    def cpar(self, _):
        self._record.opened_pars -= 1
        return self.do_cpar()

    @abstractmethod
    def text(self, token): ...

    @abstractmethod
    def colon(self, _): ...

    @classmethod
    def from_other(cls, other):
        s = cls()
        s._record = other._record
        return s

    def get_result(self):
        return self._record.result


class _ReadingModuleType(_State):
    def do_opar(self):
        self.current_value.append("(")
        return _ReadingParameters.from_other(self)

    def do_cpar(self):
        raise ValueError()

    def text(self, token):
        self.current_value.append(token)
        return self

    def colon(self, _):
        raise ValueError()


class _ReadingParameters(_State):
    def do_opar(self):
        self.current_value.append("(")
        return self

    def do_cpar(self):
        self.current_value.append(")")
        if self.opened_pars == 0:
            self.finalize()
            self.current_value = _StringValue()
            return _ReadingKey.from_other(self)
        else:
            return self

    def text(self, token):
        self.current_value.append(token)
        return self

    def colon(self, _):
        self.current_value.append(":")
        return self


class _ReadingKey(_State):
    def do_opar(self):
        return self

    def do_cpar(self):
        return self

    def text(self, token):
        self._record.key = token
        return self

    def colon(self, _):
        return _ReadingModuleType.from_other(self)


class SequentialParser:
    tokens = {
        "CPAR": r"\)",
        "OPAR": r"\(",
        "TEXT": r"[a-zA-Z0-9_,=\-\.]+\s?",
        "COLON": r":",
    }

    def __init__(self):
        self._state = _ReadingKey()

    def parse(self, serialized):
        lists_string = re.findall(r"Sequential\(\s*(.*)\s*\)\s*\Z", serialized)[0]
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
            fn = getattr(self._state, kind.lower())
            self._state = fn(token)
        result = self._state.get_result()
        self._state = _ReadingKey()
        return result
