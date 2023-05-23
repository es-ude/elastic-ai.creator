from dataclasses import dataclass
from re import finditer as find_regex
from typing import Iterable, cast


@dataclass(eq=True, frozen=True)
class Token:
    type: str
    value: str

    def matches(self, actual) -> bool:
        return (self.type == "CAPTURING_TERMINAL" and self.value == actual.type) or (
            self == actual
        )

    @classmethod
    def capturing_terminal(cls, type: str) -> "Token":
        return cls("CAPTURING_TERMINAL", type)


def tokenize(code: str | Iterable[str], token_pattern: str):
    if not isinstance(code, str):
        for line in code:
            yield from tokenize(line, token_pattern)
    else:
        for match in find_regex(token_pattern, code):
            token = Token(cast(str, match.lastgroup), match.group())
            if token.type != "SKIP":
                yield token
