import numpy as np
from torch import Tensor, nn

from elasticai.creator.precomputation.precomputation import (
    get_precomputations_from_direct_children,
    precomputable,
)

model = nn.Sequential(
    precomputable(
        nn.Sigmoid(),
        input_shape=(-1,),
        input_generator=lambda input_shape: Tensor(
            np.linspace(-10, 10, num=20).reshape(input_shape)
        ),
    )
)

for child in model.named_children():
    print(child)

precomputations = get_precomputations_from_direct_children(model)
for precomputation in precomputations:
    precomputation()
    print("x: ", precomputation[0], "\ny: ", precomputation[1])
