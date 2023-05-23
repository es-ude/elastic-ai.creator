from dataclasses import dataclass
from itertools import chain
from re import finditer as find_regex
from typing import Any, Iterable, Iterator, cast


@dataclass(eq=True, frozen=True)
class Token:
    type: str
    value: str

    def matches(self, actual) -> bool:
        return (self.type == "CAPTURING_TERMINAL" and self.value == actual.type) or (
            self == actual
        )

    @classmethod
    def intermediate(cls, name: str) -> "Token":
        return cls("INTERMEDIATE", name)

    @classmethod
    def capturing_terminal(cls, type: str) -> "Token":
        return cls("CAPTURING_TERMINAL", type)


Rule = tuple[Token, tuple[Token, ...]]


class Tokenizer:
    def __init__(self, token_pattern: str):
        self._token_pattern = token_pattern

    def tokenize(self, input: str | Iterable[str]) -> Iterator[Token]:
        if not isinstance(input, str):
            for line in input:
                yield from self.tokenize(line)
        else:
            for match in find_regex(self._token_pattern, input):
                token = Token(cast(str, match.lastgroup), match.group())
                if token.type != "SKIP":
                    yield token

    def tokenize_rule(self, input: str) -> Rule:
        """
        Converts a rule given as a string into a token sequence.
        Currently, it only supports terminals.
        The syntax is as follows:
         - Use `'abc'` to denote a literal terminal sequence abc, the symbol will be passed on to the vhdl tokenizer
         - Tokens are separated by a single space
         - There is no need for escaping: to denote a literal `'` just write `'''`
         - Use `abc` to create a "capturing terminal" of type `"abc"`, these are terminals that match any token of the type `"abc"`
        """
        intermediate, _, rest = input.partition(":")
        intermediate = intermediate.strip()
        rule: Any = rest.split()

        def convert(s: str) -> tuple[Token, ...]:
            if s.startswith("'") and s.endswith("'"):
                value = tuple(self.tokenize(s[1:-1]))
                return value
            if s.islower():
                return (Token.intermediate(s),)
            else:
                return (Token.capturing_terminal(s),)

        rule = map(convert, rule)
        rule = tuple(chain.from_iterable(rule))
        return Token.intermediate(intermediate), rule
