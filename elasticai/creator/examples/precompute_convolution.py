from functools import partial
from io import StringIO

from torch.nn import Conv1d, MaxPool1d, Sequential

from elasticai.creator.precomputation.input_domains import (
    create_codomain_for_1d_conv,
    create_codomain_for_depthwise_1d_conv,
)
from elasticai.creator.precomputation.precomputation import (
    get_precomputations_from_direct_children,
    precomputable,
)
from elasticai.creator.qat.layers import Binarize, QConv1d

model = Sequential(
    QConv1d(
        in_channels=2,
        out_channels=2,
        kernel_size=(3,),
        quantizer=Binarize(),
    ),
    MaxPool1d(kernel_size=2),
    Binarize(),
    precomputable(
        Sequential(
            Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=(4,),
                groups=2,
            ),
            Binarize(),
        ),
        input_generator=partial(
            create_codomain_for_depthwise_1d_conv, shape=(2, 4), items=[-1, 1]
        ),
        input_shape=(2, 2),
    ),
    MaxPool1d(kernel_size=2),
    precomputable(
        Sequential(
            Conv1d(
                in_channels=2,
                out_channels=3,
                kernel_size=(1,),
            ),
            Binarize(),
        ),
        input_generator=partial(
            create_codomain_for_1d_conv, shape=(1, 2), items=[-1, 1]
        ),
        input_shape=(6, 1),
    ),
)

model.eval()

with StringIO() as f:
    for precompute in get_precomputations_from_direct_children(model):
        precompute()
        precompute.dump_to(f)
    print(f.getvalue())
