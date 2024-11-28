from elasticai.creator.base_modules.dropout import Dropout as DropoutBase


class Dropout(DropoutBase):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            inplace=inplace,
        )
