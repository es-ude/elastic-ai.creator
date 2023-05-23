from itertools import chain
from typing import Callable, Iterator, Sequence

from elasticai.creator.hdl.code_generation.tokens import Rule, Token


def parse(
    input: Iterator[Token],
    rules: tuple[Rule],
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
    input = chain(input, (Token("END", ""),))

    seen_tokens: list[Token] = []
    token = next(input)

    active_rules = set(rules)

    def reset():
        seen_tokens.clear()
        nonlocal active_rules
        active_rules = set(rules)

    def determine_followup_rules(token: Token):
        nonlocal active_rules
        new_active_rules: set[Rule] = set()
        num_seen_symbols = len(seen_tokens)
        for rule in active_rules:
            rhs = rule[1]
            if rhs[num_seen_symbols].matches(token):
                new_active_rules.add(rule)
        there_is_a_rule_for_next_token = len(new_active_rules) > 0
        if there_is_a_rule_for_next_token:
            active_rules = new_active_rules
        else:
            reset()

    def a_rule_has_completed():
        num_seen_symbols = len(seen_tokens)
        for rule in active_rules:
            rhs = rule[1]
            rule_length = len(rhs)
            if num_seen_symbols == rule_length:
                return True
        return False

    try:
        while True:
            if a_rule_has_completed():
                handle(seen_tokens)
                reset()
            else:
                determine_followup_rules(token)
                seen_tokens.append(token)
                token = next(input)

    except StopIteration:
        pass
