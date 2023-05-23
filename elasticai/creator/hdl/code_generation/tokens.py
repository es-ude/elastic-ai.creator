from itertools import chain
from typing import Any

from elasticai.creator.hdl.code_generation.parsing import Token
from elasticai.creator.hdl.vhdl.code_generation.tokens import tokenize


def tokenize_rule(r: str) -> tuple[Token]:
    """
    Converts a rule given as a string into a token sequence.
    Currently, it only supports terminals.
    The syntax is as follows:
     - Use `'abc'` to denote a literal terminal sequence abc, the symbol will be passed on to the vhdl tokenizer
     - Tokens are separated by a single space
     - There is no need for escaping: to denote a literal `'` just write `'''`
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
