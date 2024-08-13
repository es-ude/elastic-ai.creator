class Rom:
    def __init__(self, name: str, data_width: int,
                 values: list[int]) -> None:
        self._name = name
        self._data_width = data_width