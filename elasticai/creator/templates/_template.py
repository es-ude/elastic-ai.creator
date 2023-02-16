from abc import ABC, abstractmethod


class Template(ABC):
    @abstractmethod
    def update_parameters(self, **parameters: str | tuple[str] | list[str]) -> None:
        ...

    @property
    @abstractmethod
    def single_line_parameters(self) -> dict[str, str]:
        ...

    @property
    @abstractmethod
    def multi_line_parameters(self) -> dict[str, tuple[str, ...] | list[str]]:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, str | tuple[str, ...] | list[str]]:
        ...

    @abstractmethod
    def lines(self) -> list[str]:
        ...
