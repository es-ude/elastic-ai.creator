from typing import Iterable, Iterator

from elasticai.creator.hdl.code_generation.parsing import Token
from elasticai.creator.hdl.code_generation.parsing import tokenize as _tokenize

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


def tokenize(code: str | Iterable[str]) -> Iterator[Token]:
    return _tokenize(code, _regex)
