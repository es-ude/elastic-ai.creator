from torch.nn import Dropout as _Dropout


class Dropout(_Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)
