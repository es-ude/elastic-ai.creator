from typing import Any, Protocol


class Evaluator(Protocol):
    def run(self) -> Any:
        ...
