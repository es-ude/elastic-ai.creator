from torch.nn import Identity
from torch import Tensor


class Identity(Identity):
    @staticmethod
    def right_inverse(x: Tensor) -> Tensor:
        return x
