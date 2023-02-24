import math
from typing import Iterable, Iterator


class ToLogicEncoder:
    """
    Throughout our implementations we have to deal with two different levels of representations for numbers:
    During training we typically need to apply mathematical operations and we do not care too much about how our numbers are encoded.
    E.g. in a scenario where we want to use two bit on hardware_description_language to represent our numbers, in our machine learning framework we
    might decide it is beneficial to use the numbers -3 and 4 for some reason. However, especially in the context of precomputed
    results, the hardware_description_language implementation does not need to know the numeric values, but instead just needs to be able to keep a
    consistent and correct mapping. The NumericToLogicEncoder takes care of performing the translations from numeric representation
    to the bit vector used in the hardware_description_language implementation. We encode bit vectors just as unsigned integers.
    """

    def __init__(self):
        self._symbols = set()
        self._mapping = dict()

    def _update_mapping(self) -> None:
        sorted_numerics = sorted(self._symbols)
        mapping = {value: index for index, value in enumerate(sorted_numerics)}
        self._mapping.update(mapping)

    def register_symbol(self, numeric_representation: int) -> None:
        self._symbols.add(numeric_representation)
        self._update_mapping()

    def register_symbols(self, symbols: Iterable[int]) -> None:
        for symbol in symbols:
            self._symbols.add(symbol)
        self._update_mapping()

    @staticmethod
    def _int_to_bin(number: int, num_bits: int) -> str:
        return f"{number:0{num_bits}b}"

    @property
    def bit_width(self) -> int:
        return math.floor(math.log2(len(self._symbols)))

    def __len__(self):
        return len(self._symbols)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._symbols == other._symbols and self._mapping == other._mapping
        else:
            return False

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for symbol, encoded_symbol in self._mapping.values():
            yield symbol, encoded_symbol

    def __getitem__(self, item: int) -> int:
        return self._mapping[item]

    def __call__(self, number: int) -> str:
        if number not in self._symbols:
            raise ValueError
        return self._int_to_bin(self._mapping[number], self.bit_width)
