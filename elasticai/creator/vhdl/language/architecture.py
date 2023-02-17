from abc import ABC, abstractmethod

from elasticai.creator.vhdl.language.entity import Entity


class Architecture(ABC):
    """
    Represents a vhdl architecture block, such as
    ```
     architecture my_architecture of my_entity is
       -- some vhdl code
     begin
       -- some more vhdl code
     end my_architecture
    ```
    for above code block we should find
    > a = parse_architecture_from(above_code_block)
    > a.name == "my_architecture"
    > a.implemented_entity.name == "my_entity"
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def implemented_entity(self) -> Entity:
        ...

    @abstractmethod
    def lines(self) -> list[str]:
        ...
