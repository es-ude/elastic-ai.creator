from typing import Iterable, Iterator

from elasticai.creator.hdl.code_generation.tokens import Rule, Token, Tokenizer

ID = r"[a-zA-Z_][a-zA-Z0-9_]*"
SKIP = r"\s+|--.*"
OPERATOR = "|".join(["<=", "=>", "-"])
DELIMITER = r"[();:=']"
NUMBER = r"\d+"
_tokens = (
    ("ID", ID),
    ("SKIP", SKIP),
    ("OPERATOR", OPERATOR),
    ("DELIMITER", DELIMITER),
    ("NUMBER", NUMBER),
)
_regex = "|".join("(?P<{}>{})".format(t, v) for t, v in _tokens)

_tokenizer = Tokenizer(_regex)


def tokenize(code: str | Iterable[str]) -> Iterator[Token]:
    return _tokenizer.tokenize(code)


def tokenize_rule(r: str) -> Rule:
    return _tokenizer.tokenize_rule(r)
