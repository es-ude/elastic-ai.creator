from dataclasses import dataclass
from itertools import chain
from re import finditer as find_regex
from typing import Callable, Iterable, Iterator, Sequence, cast


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


def parse(
    tokens: Iterator[Token],
    rules: tuple[tuple[Token, ...]],
    handle: Callable[[Sequence[Token]], None],
):
    """
    This is a rudimentary but robust parser.
    Given a tuple of rules, that consist entirely of terminals (see the `.tokens` module) it will call `handle`
    on the sequence of tokens that matches the shortest of the given rules and continue parsing.
    It is robust in the following sense: when encountering an unexpected token, the algorithm will start parsing
    from scratch with that token instead of raising an exception.

    It's primary use case is to extract simple patterns from vhdl code without having to parse an entire file
    or specify a full vhdl grammar.
    """
    tokens = chain(tokens, (Token("END", ""),))

    try:
        seen_tokens: list[Token] = []
        token = next(tokens)
        active_rules = set(rules)

        def reset():
            seen_tokens.clear()
            nonlocal active_rules
            active_rules = set(rules)

        def determine_followup_rules(token: Token):
            nonlocal active_rules
            new_active_rules: set[tuple[Token, ...]] = set()
            num_seen_symbols = len(seen_tokens)
            for rule in active_rules:
                if rule[num_seen_symbols].matches(token):
                    new_active_rules.add(rule)
            there_is_a_rule_for_next_token = len(new_active_rules) > 0
            if there_is_a_rule_for_next_token:
                active_rules = new_active_rules
            else:
                reset()

        def a_rule_has_completed():
            num_seen_symbols = len(seen_tokens)
            for rule in active_rules:
                rule_length = len(rule)
                if num_seen_symbols == rule_length:
                    return True
            return False

        while True:
            if a_rule_has_completed():
                handle(seen_tokens)
                reset()
            else:
                determine_followup_rules(token)
                seen_tokens.append(token)
                token = next(tokens)

    except StopIteration:
        pass
