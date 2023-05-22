import re
from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable, Iterator, cast

ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
SKIP = r"\s+|--.*"
OPERATOR = "|".join(["<=", "=>", "-"])
DELIMITER = r"[();:=']"
NUMBER = r"\d+"


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


def tokenize(code: str | Iterable[str]) -> Iterator[Token]:
    if not isinstance(code, str):
        for line in code:
            yield from tokenize(line)
    else:
        _tokens = (
            ("ID", ID),
            ("SKIP", SKIP),
            ("OPERATOR", OPERATOR),
            ("DELIMITER", DELIMITER),
            ("NUMBER", NUMBER),
        )
        regex = "|".join("(?P<{}>{})".format(t, v) for t, v in _tokens)
        for match in re.finditer(regex, code):
            token = Token(cast(str, match.lastgroup), match.group())
            if token.type != "SKIP":
                yield token


def tokenize_rule(r: str) -> tuple[Token]:
    """
    Converts a rule given as a string into a token sequence.
    Currently, it only supports terminals.
    The syntax is as follows:
     - Use `'abc'` to denote a literal terminal sequence abc, the symbol will be passed on to the vhdl tokenizer
     - Use `abc` to create a "capturing terminal" of type `"abc"`, these are terminals that match any token of the type `"abc"`
    """
    rule: Any = r.split()

    def convert(s: str) -> tuple[Token, ...]:
        if s.startswith("'") and s.endswith("'"):
            value = tuple(tokenize(s[1:-1]))
            return value
        else:
            return (Token.capturing_terminal(s),)

    rule = map(convert, rule)
    rule = tuple(chain.from_iterable(rule))
    return rule
